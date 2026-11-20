"""The ``dreambot`` command."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from dreambot import __version__
from dreambot.config import Config, ConfigError, load_config

app = typer.Typer(
    help="Dreambot — a free, open-source, fully local Midjourney alternative for the Mac, powered by MLX.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
models_app = typer.Typer(help="See, download and inspect models.", no_args_is_help=False)
app.add_typer(models_app, name="models")
loras_app = typer.Typer(
    help="LoRAs: small fine-tunes that give a model one look (anime, pixel art, ...).",
    no_args_is_help=False,
)
app.add_typer(loras_app, name="loras")
service_app = typer.Typer(help="Run Dreambot in the background (starts at login).", no_args_is_help=True)
app.add_typer(service_app, name="service")
access_app = typer.Typer(help="Control who may use your Dreambot.", no_args_is_help=False)
app.add_typer(access_app, name="access")

console = Console()
err = Console(stderr=True)


def _plain(text: object) -> str:
    """Words that came from somewhere else (a message, a name the person typed), to be printed as they are.

    Rich reads square brackets as its own markup. Put into a coloured line as it is, a hint that
    names a section of the config file (``[web] port``) loses the section, and a name that
    holds something like ``[/red]`` stops the command with a traceback.
    """
    return escape(str(text))


LOCAL_USER = "console:me"
LOCAL_CHANNEL = "console:main"


def _config() -> Config:
    try:
        return load_config()
    except ConfigError as exc:
        err.print(f"[red]Config error:[/red] {_plain(exc)}")
        raise typer.Exit(2) from exc


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _version(value: bool) -> None:
    if value:
        console.print(f"dreambot {__version__}")
        raise typer.Exit()


@app.callback()
def main_options(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    pass


# --------------------------------------------------------------------------------------
# imagine / chat
# --------------------------------------------------------------------------------------


def _quiet_worker_log(config: Config):
    """Where the inference worker's own output (progress bars, model warnings) goes."""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    return open(config.log_dir / "worker.log", "a", buffering=1)


@app.command()
def imagine(
    prompt: Annotated[list[str], typer.Argument(help="The prompt, with any --parameters, in quotes.")],
    attach: Annotated[
        list[Path] | None, typer.Option("--attach", "-a", help="Image to attach (repeatable).", exists=True)
    ] = None,
    show: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the result in Preview when done.")
    ] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate images from a prompt, exactly as you would message it to the bot.

    Example: dreambot imagine "a red fox in the snow --ar 16:9 --chaos 20"
    """
    _logging(verbose)
    code = asyncio.run(_imagine(" ".join(prompt), [str(path) for path in attach or []], show, verbose))
    raise typer.Exit(code)


async def _imagine(prompt: str, attachments: list[str], show: bool, verbose: bool) -> int:
    from dreambot.core.jobs import EventType, JobEvent, JobStatus
    from dreambot.core.render import job_caption
    from dreambot.core.service import Dreambot, DreambotError
    from dreambot.engine.client import Engine

    config = _config()
    log = None if verbose else _quiet_worker_log(config)
    engine = Engine(backend=config.engine.backend, idle_timeout=None, stderr=log)
    done: dict[str, asyncio.Event] = {}
    failed = False

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        tasks: dict[str, int] = {}

        def on_event(event: JobEvent) -> None:
            job = event.job
            if job.id not in tasks:
                tasks[job.id] = progress.add_task(f"#{job.id} queued", total=1.0)
            task = tasks[job.id]
            if event.type is EventType.LOADING:
                progress.update(task, description=f"#{job.id} loading {job.model}")
            elif event.type is EventType.THINKING:
                styled = job.params.get("sref") or job.params.get("personalize") or job.detail.get("style")
                progress.update(
                    task, description=f"#{job.id} {'working out the look' if styled else 'art-directing'}"
                )
            elif event.type in (EventType.STARTED, EventType.PROGRESS):
                progress.update(task, description=f"#{job.id} {job.model}", completed=event.fraction)
            elif event.type in (EventType.FINISHED, EventType.FAILED, EventType.CANCELLED):
                progress.update(task, completed=1.0)
                done.setdefault(job.id, asyncio.Event()).set()

        async with Dreambot(config, engine=engine) as service:
            service.subscribe(on_event)
            try:
                jobs = await service.imagine(
                    user=LOCAL_USER,
                    channel=LOCAL_CHANNEL,
                    prompt=prompt,
                    attachments=attachments,
                    allow_paths=True,
                )
            except DreambotError as exc:
                err.print(f"[red]{_plain(exc)}[/red]")
                return 2
            try:
                for job in jobs:
                    await done.setdefault(job.id, asyncio.Event()).wait()
            except (KeyboardInterrupt, asyncio.CancelledError):
                await service.cancel(user=LOCAL_USER)
                return 130

    for job in jobs:
        if job.status is JobStatus.DONE:
            console.print(_plain(job_caption(job)))
            for image in job.images:
                console.print(f"  {image.index}. {_plain(image.path)}")
            if job.grid:
                console.print(f"  grid: {_plain(job.grid)}")
            if show:
                subprocess.run(["/usr/bin/open", job.grid or job.images[0].path], check=False)
        else:
            failed = True
            err.print(f"[red]#{job.id} {job.status.value}:[/red] {_plain(job.error or '')}")
    return 1 if failed else 0


@app.command()
def chat(
    show: Annotated[bool, typer.Option("--open/--no-open", help="Open each result in Preview.")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Talk to Dreambot in the terminal: the same commands and shortcuts as the chat bots."""
    _logging(verbose)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_chat(verbose, show))


async def _chat(verbose: bool, open_results: bool = True) -> None:
    from dreambot.core.commands import Commands, Reply, job_reply
    from dreambot.core.jobs import EventType, JobEvent
    from dreambot.core.service import Dreambot
    from dreambot.engine.client import Engine

    config = _config()
    log = None if verbose else _quiet_worker_log(config)
    engine = Engine(
        backend=config.engine.backend, idle_timeout=config.engine.idle_unload_seconds or None, stderr=log
    )

    def show(reply: Reply) -> None:
        if reply.text:
            console.print(reply.text, markup=False, highlight=False)
        for image in reply.images:
            console.print(f"  [dim]{_plain(image)}[/dim]")
        if reply.images and open_results:
            subprocess.run(["/usr/bin/open", "-g", reply.images[0]], check=False)

    def on_event(event: JobEvent) -> None:
        job = event.job
        if event.type is EventType.QUEUED and event.position > 0:
            console.print(f"[dim]#{job.id} queued, {event.position} ahead[/dim]")
        elif event.type is EventType.LOADING:
            console.print(f"[dim]#{job.id} loading {job.model}…[/dim]")
        elif event.type in (EventType.FINISHED, EventType.FAILED, EventType.CANCELLED):
            show(job_reply(job))
            console.print()

    async with Dreambot(config, engine=engine) as service:
        service.subscribe(on_event)
        commands = Commands(service)
        console.print(
            "[bold]Dreambot[/bold] — type a prompt, a shortcut like U2, or /help. Ctrl-D to quit.\n"
        )
        while True:
            try:
                line = await asyncio.to_thread(console.input, "[bold cyan]›[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                break
            text, attachments = _split_attachments(line)
            if not text and not attachments:
                continue
            if text.lower() in ("quit", "exit", "/quit", "/exit"):
                break
            replies = await commands.handle(
                user=LOCAL_USER,
                channel=LOCAL_CHANNEL,
                text=text,
                attachments=attachments,
                direct=True,
                allow_paths=True,
            )
            for reply in replies:
                show(reply)
        while service.queue_length:
            await asyncio.sleep(0.2)
        await asyncio.sleep(0.1)


def _split_attachments(line: str) -> tuple[str, list[str]]:
    """Leading tokens that name existing image files (e.g. dragged into the terminal) are attachments."""
    from dreambot.core.commands import split_command

    words = split_command(line.strip())
    attachments: list[str] = []
    consumed = 0
    for word in words:
        candidate = Path(word).expanduser()
        if candidate.is_file() and candidate.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".heic",
            ".gif",
            ".tiff",
            ".bmp",
        }:
            attachments.append(str(candidate))
            consumed += 1
        else:
            break
    if not attachments:
        return line.strip(), []
    return " ".join(words[consumed:]), attachments


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------


@models_app.callback(invoke_without_command=True)
def models_default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        models_list()


@models_app.command("list")
def models_list(
    all_models: Annotated[
        bool, typer.Option("--all", "-a", help="Include models that are not installed.")
    ] = False,
) -> None:
    """List models: what is installed, and (with --all) everything Dreambot supports."""
    from dreambot.core.planner import Catalog
    from dreambot.engine import weights
    from dreambot.engine.registry import REGISTRY

    config = _config()
    catalog = Catalog(config)
    table = Table(show_lines=False, header_style="bold")
    for column in ("Model", "Installed", "Can do", "License", "About"):
        table.add_column(column, overflow="fold")
    shown = 0
    for spec in REGISTRY:
        found = weights.installed(spec)
        if not found and not all_models:
            continue
        shown += 1
        best = catalog.weights(spec)
        installed = ", ".join(f"{w.label} {w.size_gb:.0f}GB" for w in found) if found else "[dim]—[/dim]"
        if best is not None and len(found) > 1:
            installed += f" [dim](uses {best.label})[/dim]"
        license_text = spec.license if spec.permissive else f"[yellow]{spec.license}[/yellow]"
        mark = " [green]✓[/green]" if spec.verified else ""
        table.add_row(
            f"[bold]{spec.id}[/bold]{mark}",
            installed,
            ", ".join(op.value for op in spec.ops) or "[dim]specialty[/dim]",
            license_text,
            spec.summary,
        )
    if shown == 0:
        console.print(
            "No models are installed yet. Get started with:  [bold]dreambot models pull z-image-turbo[/bold]"
        )
        console.print("See everything available with:  dreambot models list --all")
        return
    console.print(table)
    console.print(
        "[green]✓[/green] = verified end-to-end by Dreambot's test-suite. "
        "[yellow]Yellow[/yellow] licenses restrict use (usually non-commercial); read them before relying on a model."
    )
    if not all_models:
        console.print(
            f"[dim]{len(REGISTRY) - shown} more models are available: dreambot models list --all[/dim]"
        )


@models_app.command("pull")
def models_pull(
    model: Annotated[str, typer.Argument(help="Model name, e.g. z-image-turbo, flux2-klein-4b, seedvr2-3b.")],
    quant: Annotated[
        str | None,
        typer.Option(
            "--quant", "-q", help="Quantization: q3, q4, q5, q6, q8 or bf16. Default: best for this Mac."
        ),
    ] = None,
) -> None:
    """Download a model's weights into the shared Hugging Face cache."""
    from dreambot.engine import weights
    from dreambot.engine.registry import REGISTRY, UnknownModel

    env = {**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
    env.pop("HF_HUB_OFFLINE", None)
    if model.lower() in ("describe", "vision"):
        # Not an image model: the vision-language model behind /describe and /shorten.
        console.print(f"[bold]Vision model[/bold] for /describe and /shorten: {weights.DEFAULT_VISION_MODEL}")
        args = [sys.executable, "-m", "dreambot.engine.prepare", "describe"]
        raise typer.Exit(subprocess.run(args, env=env, check=False).returncode)
    try:
        spec = REGISTRY.get(model)
    except UnknownModel as exc:
        err.print(f"[red]{_plain(exc)}[/red]")
        raise typer.Exit(2) from exc
    repo, level = weights.recommended_repo(spec, [quant] if quant else None)
    console.print(f"[bold]{spec.title}[/bold] — {spec.summary}")
    console.print(f"Source: https://huggingface.co/{repo}" + (f" ({level})" if level else ""))
    if not spec.permissive:
        console.print(
            f"[yellow]License: {spec.license}.[/yellow] Review it at https://huggingface.co/{spec.repo}"
        )
    args = [sys.executable, "-m", "dreambot.engine.prepare", spec.id, *([quant] if quant else [])]
    raise typer.Exit(subprocess.run(args, env=env, check=False).returncode)


@models_app.command("info")
def models_info(model: Annotated[str, typer.Argument(help="Model name.")]) -> None:
    """Show everything Dreambot knows about one model."""
    from dreambot.engine import weights
    from dreambot.engine.registry import REGISTRY, UnknownModel

    try:
        spec = REGISTRY.get(model)
    except UnknownModel as exc:
        err.print(f"[red]{_plain(exc)}[/red]")
        raise typer.Exit(2) from exc
    console.print(f"[bold]{spec.title}[/bold] ({spec.id})")
    console.print(spec.summary)
    console.print(f"Operations: {', '.join(op.value for op in spec.ops) or 'none wired up yet'}")
    console.print(
        f"Defaults: {spec.steps} steps"
        + (f", guidance {spec.guidance:g}" if spec.guidance is not None else "")
    )
    console.print(f"Negative prompts: {'yes' if spec.cfg else 'no (best-effort)'}")
    console.print(f"License: {spec.license}" + ("" if spec.permissive else " — check before commercial use"))
    console.print(f"Upstream: https://huggingface.co/{spec.repo}" + (" (gated)" if spec.gated else ""))
    if spec.aliases:
        console.print(f"Also known as: {', '.join(spec.aliases)}")
    found = weights.installed(spec)
    if found:
        for item in found:
            console.print(
                f"Installed: {_plain(item.repo)} — {item.label}, {item.size_gb:.1f} GB\n  {_plain(item.path)}"
            )
    else:
        repo, level = weights.recommended_repo(spec)
        console.print(
            f"Not installed. `dreambot models pull {spec.id}` would fetch {repo}"
            + (f" ({level})" if level else "")
        )
    console.print("Verified by Dreambot's tests: " + ("yes" if spec.verified else "not yet"))


# --------------------------------------------------------------------------------------
# loras
# --------------------------------------------------------------------------------------


@loras_app.callback(invoke_without_command=True)
def loras_default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        loras_list()


@loras_app.command("list")
def loras_list() -> None:
    """List the LoRAs in your library."""
    from dreambot.core.planner import Catalog
    from dreambot.engine.loras import Library
    from dreambot.engine.registry import REGISTRY

    config = _config()
    catalog = Catalog(config)
    found, problems = Library(config.lora_dir).entries()
    if not found:
        console.print("No LoRAs yet. A LoRA is a small fine-tune that gives a model one look. Add one with:")
        console.print(
            "  [bold]dreambot loras add tarn59/pixel_art_style_lora_z_image_turbo --name pixel-art[/bold]"
        )
        console.print("The source is a LoRA on Hugging Face, or a .safetensors file on this Mac.")
    else:
        table = Table(show_lines=False, header_style="bold")
        for column in ("LoRA", "Trained on", "Trigger words", "Scale", "Size"):
            table.add_column(column, overflow="fold")
        for entry in found:
            base = REGISTRY.find(entry.base) if entry.base else None
            if base is None:
                trained_on = "[dim]not known[/dim]"
            elif catalog.is_installed(base):
                trained_on = base.id
            else:
                trained_on = f"{base.id} [yellow](not installed)[/yellow]"
            table.add_row(
                f"[bold]{entry.name}[/bold]",
                trained_on,
                entry.trigger or "[dim]—[/dim]",
                f"{entry.scale:g}",
                f"{Path(entry.path).stat().st_size / 1000**2:.0f} MB",
            )
        console.print(table)
        console.print(
            "Use one in a prompt with [bold]--lora <name>[/bold], or always with `/settings lora <name>`."
        )
    for name, problem in sorted(problems.items()):
        err.print(f"[yellow]{name}[/yellow] is left out: {problem}.")


@loras_app.command("add")
def loras_add(
    source: Annotated[
        str,
        typer.Argument(
            help="A LoRA on Hugging Face (owner/name, or its link), or a .safetensors file on this Mac."
        ),
    ],
    name: Annotated[str | None, typer.Option("--name", "-n", help="What to call it, e.g. pixel-art.")] = None,
    base: Annotated[
        str | None,
        typer.Option("--base", "-b", help="The model it was trained on. Default: what its page says."),
    ] = None,
    trigger: Annotated[
        str | None,
        typer.Option(
            "--trigger",
            "-t",
            help="Words put in front of the prompt. With {prompt} in them, where the prompt goes. "
            "Default: what its page says.",
        ),
    ] = None,
    scale: Annotated[float, typer.Option("--scale", min=-4, max=4, help="How strongly it is applied.")] = 1.0,
    file: Annotated[
        str | None, typer.Option("--file", help="Which file, when a repository holds several.")
    ] = None,
    title: Annotated[str | None, typer.Option("--title", help="A name for people to read.")] = None,
    replace: Annotated[
        bool, typer.Option("--replace", help="Replace the LoRA that already has this name.")
    ] = False,
) -> None:
    """Add a LoRA to your library.

    Example: dreambot loras add tarn59/pixel_art_style_lora_z_image_turbo --name pixel-art
    """
    import json

    from dreambot.core.planner import Catalog
    from dreambot.engine.loras import Library
    from dreambot.engine.registry import REGISTRY

    config = _config()
    folder = config.lora_dir
    before = {path.name: path.stat().st_mtime_ns for path in folder.glob("*.toml")} if folder.is_dir() else {}
    request = {
        "source": source,
        "name": name,
        "base": base,
        "title": title,
        "trigger": trigger,
        "scale": scale,
        "file": file,
        "replace": replace,
    }
    env = {**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
    env.pop("HF_HUB_OFFLINE", None)
    args = [sys.executable, "-m", "dreambot.engine.prepare", "lora", str(folder), json.dumps(request)]
    code = subprocess.run(args, env=env, check=False).returncode
    if code != 0:
        raise typer.Exit(code)
    library = Library(folder)
    for notes in sorted(folder.glob("*.toml")):
        if before.get(notes.name) == notes.stat().st_mtime_ns:
            continue
        entry = library.find(notes.stem)
        if entry is None:
            continue
        if entry.trigger:
            console.print(f"Trigger words, put in front of the prompt: [bold]{entry.trigger}[/bold]")
        else:
            console.print(
                "[dim]No trigger words are known for it. If its page names some, write them into "
                f'{notes} as `trigger = "..."`.[/dim]'
            )
        trained_on = REGISTRY.find(entry.base) if entry.base else None
        if trained_on is not None and not trained_on.permissive:
            console.print(
                f"[yellow]{trained_on.title} is licensed {trained_on.license}[/yellow], and so is what is made with it."
            )
        if trained_on is not None and not Catalog(config).is_installed(trained_on):
            console.print(
                f"[yellow]It was trained on {trained_on.id}, which is not installed:[/yellow] "
                f"dreambot models pull {trained_on.id}"
            )
        console.print(f'Try it:  [bold]dreambot imagine "a fox in a forest --lora {entry.name}"[/bold]')
        console.print(
            f"For every prompt: `/settings lora {entry.name}` in chat, or the LoRA button by the prompt "
            "in the web app."
        )


@loras_app.command("info")
def loras_info(name: Annotated[str, typer.Argument(help="The LoRA's name.")]) -> None:
    """Show everything about one LoRA."""
    from dreambot.engine.loras import Library, LoraError
    from dreambot.engine.registry import REGISTRY

    config = _config()
    try:
        entry = Library(config.lora_dir).find(name)
    except LoraError as exc:
        err.print(f"[red]{_plain(name)} can't be used: {_plain(exc)}.[/red]")
        raise typer.Exit(2) from exc
    if entry is None:
        err.print(f"[red]There is no LoRA called {_plain(name)}.[/red] See `dreambot loras`.")
        raise typer.Exit(2)
    console.print(f"[bold]{entry.title or entry.name}[/bold] ({entry.name})")
    base = REGISTRY.find(entry.base) if entry.base else None
    console.print(
        f"Trained on: {f'{base.title} ({base.id})' if base else 'not known — it is tried on whichever model a prompt uses'}"
    )
    console.print(f"Trigger words: {entry.trigger or 'none'}")
    console.print(f"Scale: {entry.scale:g}")
    console.print(f"File: {entry.path}  [dim]({Path(entry.path).stat().st_size / 1000**2:.0f} MB)[/dim]")
    if entry.source:
        console.print(f"Source: {entry.source}")
    console.print(
        f"License: {entry.license or 'not stated'}"
        + (f"; {base.title} itself is {base.license}" if base else "")
    )


@loras_app.command("remove")
def loras_remove(
    name: Annotated[str, typer.Argument(help="The LoRA's name.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask first.")] = False,
) -> None:
    """Remove a LoRA from your library: its file and its notes."""
    from dreambot.engine import loras

    folder = _config().lora_dir
    try:
        name = loras.check_name(name)
    except loras.LoraError as exc:
        err.print(f"[red]{_plain(exc)}[/red]")
        raise typer.Exit(2) from exc
    there = [path for path in (folder / f"{name}.safetensors", folder / f"{name}.toml") if path.is_file()]
    if not there:
        err.print(f"[red]There is no LoRA called {_plain(name)}.[/red]")
        raise typer.Exit(2)
    if not yes and not typer.confirm(f"Remove the LoRA {name} ({', '.join(path.name for path in there)})?"):
        raise typer.Exit(1)
    loras.remove(folder, name)
    console.print(f"Removed {name}. Pictures made with it stay where they are.")


# --------------------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------------------


@app.command()
def params() -> None:
    """List every prompt parameter and how faithfully it reproduces Midjourney's."""
    from dreambot.core.render import params_text

    console.print(params_text(), markup=False, highlight=False)


@app.command()
def doctor() -> None:
    """Check that this Mac is ready to run Dreambot."""
    from dreambot.core.planner import Catalog
    from dreambot.core.settings import auto_enhance
    from dreambot.engine import weights

    config = _config()
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        mark = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"{mark} {label}" + (f" [dim]— {detail}[/dim]" if detail else ""))

    console.print(f"[bold]Dreambot {__version__}[/bold]\n")
    check(
        "Apple Silicon Mac",
        platform.system() == "Darwin" and platform.machine() == "arm64",
        platform.platform(),
    )
    memory = weights.physical_memory_gb()
    check("Memory", memory >= 16, f"{memory:.0f} GB (16 GB minimum; 32 GB+ recommended)")
    check("Python", sys.version_info >= (3, 11), sys.version.split()[0])
    try:
        from importlib.metadata import version

        check("MLX", True, f"mlx {version('mlx')}, mflux {version('mflux')}")
    except Exception as exc:
        check("MLX", False, str(exc))

    installed = Catalog(config).installed()
    check(
        "Models installed",
        bool(installed),
        ", ".join(spec.id for spec in installed) or "run: dreambot models pull z-image-turbo",
    )
    vision = weights.find_vision_model(config.assist.vision_model)
    language = (
        vision.parent.parent.name.removeprefix("models--").replace("--", "/")
        if vision is not None and vision.parent.parent.name.startswith("models--")
        else (str(vision) if vision else None)
    )
    enhancing = auto_enhance() if config.defaults.enhance is None else config.defaults.enhance
    if language is None:
        console.print(
            "[dim]·[/dim] Language model: none [dim]— `dreambot models pull describe` adds /describe, "
            "smarter /shorten and the prompt enhancer[/dim]"
        )
    else:
        state = "on" if enhancing else "off by default (`/settings enhance on` to use it)"
        console.print(f"[green]✓[/green] Language model [dim]— {language}; prompt enhancer {state}[/dim]")
    console.print(f"  [dim]Hugging Face cache: {weights.hf_cache_dir()}[/dim]")
    console.print(
        f"  [dim]Config: {config.config_path}{'' if config.config_path.exists() else ' (not created; defaults in use)'}[/dim]"
    )
    console.print(f"  [dim]Images are saved to: {config.output_dir}[/dim]")
    console.print()
    console.print("[bold]Chat frontends[/bold]")
    from dreambot import launchd, secrets
    from dreambot.frontends.imessage.chatdb import ChatDB, ChatDBError

    for name, enabled in (("telegram", config.telegram.enabled), ("discord", config.discord.enabled)):
        source = secrets.token_source(name)
        state = "enabled" if enabled else "off"
        token = f"token in {source}" if source else "no token yet"
        mark = (
            "[green]✓[/green]"
            if enabled and source
            else ("[yellow]![/yellow]" if enabled else "[dim]·[/dim]")
        )
        console.print(f"{mark} {name.title()}: {state}, {token}")
    if not config.imessage.enabled:
        # The Messages database is only opened once iMessage has been switched on in setup.
        console.print("[dim]·[/dim] iMessage: off")
    else:
        try:
            reader = ChatDB()
            reader.open()
            reader.close()
            readable, problem = True, ""
        except ChatDBError as exc:
            readable, problem = False, str(exc)
        state = f"enabled ({config.imessage.mode} mode)"
        if readable:
            console.print(f"[green]✓[/green] iMessage: {state}, Messages database is readable")
        elif launchd.is_loaded():
            # This check runs in the terminal; the background service has permissions of its own.
            console.print(
                f"[yellow]![/yellow] iMessage: {state}; this terminal cannot read the Messages database, "
                "but the background service may be allowed to"
            )
            console.print(f"  [dim]Its log says: {config.log_dir / 'dreambot.log'}[/dim]")
        else:
            ok = False
            console.print(f"[red]✗[/red] iMessage: {state}, cannot read the Messages database yet")
            console.print(f"  [dim]{problem}[/dim]")
        console.print(f"  [dim]For iMessage, grant Full Disk Access to: {launchd.interpreter()}[/dim]")
        console.print("  [dim](and to your terminal app when you run `dreambot run` by hand)[/dim]")
    if config.web.enabled:
        console.print(
            f"[green]✓[/green] Web app: on, at http://127.0.0.1:{config.web.port}/ whenever Dreambot runs"
        )
    else:
        console.print("[dim]·[/dim] Web app: off — `dreambot web` opens it, `dreambot setup web` keeps it on")
    service = launchd.status()
    console.print(
        "[dim]· Background service: "
        + (
            "running"
            if service["loaded"]
            else "installed but not loaded"
            if service["installed"]
            else "not installed"
        )
        + "[/dim]"
    )
    console.print()
    console.print("[green]Ready.[/green]" if ok else "[red]Some checks failed.[/red]")
    raise typer.Exit(0 if ok else 1)


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


@app.command()
def run(
    telegram: Annotated[
        bool | None, typer.Option("--telegram/--no-telegram", help="Override the config for this run.")
    ] = None,
    discord: Annotated[
        bool | None, typer.Option("--discord/--no-discord", help="Override the config for this run.")
    ] = None,
    imessage: Annotated[
        bool | None, typer.Option("--imessage/--no-imessage", help="Override the config for this run.")
    ] = None,
    web: Annotated[
        bool | None, typer.Option("--web/--no-web", help="Override the config for this run.")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Start the bot: serve iMessage, Discord, Telegram and the web app (whichever are set up)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "telegram", "discord.gateway", "discord.client", "discord.http"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Dreambot has no voice features; discord.py warns at every start that it lacks the libraries for them.
    logging.getLogger("discord.client").addFilter(lambda record: "voice will NOT" not in record.getMessage())
    config = _config()
    want = {
        "telegram": config.telegram.enabled if telegram is None else telegram,
        "discord": config.discord.enabled if discord is None else discord,
        "imessage": config.imessage.enabled if imessage is None else imessage,
        "web": config.web.enabled if web is None else web,
    }
    if not any(want.values()):
        console.print("No chat frontend is enabled yet. Run [bold]dreambot setup[/bold] to connect one,")
        console.print("or try it right here with [bold]dreambot chat[/bold] or [bold]dreambot web[/bold].")
        raise typer.Exit(0)
    code = asyncio.run(_run(config, want, verbose))
    raise typer.Exit(code)


async def _run(config: Config, want: dict[str, bool], verbose: bool) -> int:
    import signal

    from dreambot import secrets
    from dreambot.core.service import Dreambot
    from dreambot.engine.client import Engine
    from dreambot.frontends.base import ChatFrontend

    log = logging.getLogger("dreambot")
    worker_log = None if verbose else _quiet_worker_log(config)
    engine = Engine(
        backend=config.engine.backend,
        idle_timeout=config.engine.idle_unload_seconds or None,
        stderr=worker_log,
        env=(
            {"DREAMBOT_MEMORY_BUDGET_GB": str(config.engine.memory_budget_gb)}
            if config.engine.memory_budget_gb
            else None
        ),
    )
    started: list[ChatFrontend] = []
    async with Dreambot(config, engine=engine) as service:
        if want["telegram"]:
            token = secrets.get_token("telegram")
            if not token:
                log.error("Telegram is enabled but has no token. Run `dreambot setup`.")
            else:
                from dreambot.frontends.telegram_bot import TelegramFrontend

                await _start(TelegramFrontend(service, token), started, log)
        if want["discord"]:
            token = secrets.get_token("discord")
            if not token:
                log.error("Discord is enabled but has no token. Run `dreambot setup`.")
            else:
                from dreambot.frontends.discord_bot import DiscordFrontend

                await _start(DiscordFrontend(service, token, guild_ids=config.discord.guilds), started, log)
        if want["imessage"]:
            from dreambot.frontends.imessage.bot import IMessageFrontend

            await _start(IMessageFrontend(service), started, log)
        site = None
        if want.get("web"):
            from dreambot.frontends.web.server import WebFrontend, clean_cache

            clean_cache(config.home)
            site = WebFrontend(service)
            await _start(site, started, log)

        if not started:
            log.error("No frontend could be started.")
            return 1
        # The web app answers this Mac only and is its owner's: it is the chat apps that need claiming.
        chats = [front for front in started if front.name != "web"]
        if chats and not service.access.has_owner and not (len(chats) == 1 and chats[0].name == "imessage"):
            code = service.access.new_pairing_code()
            console.print(
                f"\nNobody owns this Dreambot yet. Send this to the bot to claim it:  [bold]/pair {code}[/bold]\n"
            )
        log.info("Dreambot is running: %s. Press Ctrl-C to stop.", ", ".join(front.name for front in started))
        if site is not None and site in started:
            log.info("The web app is at %s", site.url)

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, stop.set)
        await stop.wait()
        log.info("Stopping…")
        for front in reversed(started):
            with contextlib.suppress(Exception):
                await front.stop()
    return 0


async def _start(front, started: list, log: logging.Logger) -> None:
    try:
        await front.start()
    except Exception as exc:
        log.error("Could not start %s: %s", front.name, exc)
        return
    started.append(front)


@app.command()
def web(
    port: Annotated[
        int | None, typer.Option("--port", help="Listen on this port instead of the configured one.")
    ] = None,
    show: Annotated[bool, typer.Option("--open/--no-open", help="Open it in your browser.")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Open the web app: create, organise, Lookbooks and styles, in your browser."""
    _logging(verbose)
    config = _config()
    from dreambot import launchd

    address = f"http://127.0.0.1:{port or config.web.port}/"
    if launchd.is_loaded() and config.web.enabled and port is None:
        # The background service is already serving it, with the chat bots' jobs in its feed.
        console.print(f"The background service is hosting the web app: [bold]{address}[/bold]")
        if show:
            import webbrowser

            webbrowser.open(address)
        raise typer.Exit(0)
    if launchd.is_loaded():
        console.print(
            "[dim]The background service is running without the web app. `dreambot setup web` has it "
            "host this too, so that pictures made in the chat apps appear here as they are made.[/dim]"
        )
    with contextlib.suppress(KeyboardInterrupt):
        code = asyncio.run(_web(config, port, show, verbose))
        raise typer.Exit(code)


async def _web(config: Config, port: int | None, show: bool, verbose: bool) -> int:
    import signal

    from dreambot.core.service import Dreambot
    from dreambot.engine.client import Engine
    from dreambot.frontends.web.server import WebFrontend, clean_cache

    worker_log = None if verbose else _quiet_worker_log(config)
    engine = Engine(
        backend=config.engine.backend,
        idle_timeout=config.engine.idle_unload_seconds or None,
        stderr=worker_log,
        env=(
            {"DREAMBOT_MEMORY_BUDGET_GB": str(config.engine.memory_budget_gb)}
            if config.engine.memory_budget_gb
            else None
        ),
    )
    clean_cache(config.home)
    async with Dreambot(config, engine=engine) as service:
        site = WebFrontend(service, port=port)
        try:
            await site.start()
        except RuntimeError as exc:
            err.print(f"[red]The web app could not start: {_plain(exc)}[/red]")
            return 1
        console.print(f"Dreambot's web app is at [bold]{site.url}[/bold]  (Ctrl-C to stop)")
        if show:
            import webbrowser

            webbrowser.open(site.url)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, stop.set)
        await stop.wait()
        console.print("Stopping…")
        await site.stop()
    return 0


# --------------------------------------------------------------------------------------
# setup, pairing, access
# --------------------------------------------------------------------------------------


def _update_config(config: Config, changes: dict[str, dict]) -> None:
    """Merge ``changes`` (section -> values) into config.toml, keeping everything else in it."""
    import tomllib

    import tomli_w

    path = config.config_path
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for section, values in changes.items():
        data.setdefault(section, {}).update(values)
    data.pop("home", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")


def _discord_invite(token: str) -> str | None:
    """Build the add-to-server link from the application id embedded in a bot token."""
    import base64

    try:
        first = token.split(".")[0]
        application_id = int(base64.b64decode(first + "=" * (-len(first) % 4)).decode())
    except (ValueError, IndexError, UnicodeDecodeError):
        return None
    from dreambot.frontends.discord_bot import PERMISSIONS

    return (
        f"https://discord.com/oauth2/authorize?client_id={application_id}"
        f"&permissions={PERMISSIONS.value}&scope=bot+applications.commands"
    )


def _check_telegram_token(token: str) -> tuple[bool | None, str]:
    """Ask Telegram whose token this is.

    Returns ``(True, username)``, ``(False, why)`` when Telegram rejects it, or ``(None, why)``
    when Telegram could not be reached. Error text never includes the request URL: the token
    is part of it.
    """
    import re

    import httpx

    if not re.fullmatch(r"\d{5,}:[A-Za-z0-9_-]{30,}", token):
        return False, "bot tokens look like 123456789:AAE… (numbers, a colon, then about 35 characters)"
    try:
        response = httpx.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        data = response.json()
    except httpx.HTTPError as exc:
        return None, type(exc).__name__
    except ValueError:
        return None, "an unreadable answer"
    if isinstance(data, dict) and data.get("ok"):
        return True, str(data.get("result", {}).get("username", ""))
    reason = data.get("description") if isinstance(data, dict) else None
    return False, str(reason or f"HTTP {response.status_code}")


def _check_discord_token(token: str) -> tuple[bool | None, str]:
    """Ask Discord whose token this is: ``(True, name)``, ``(False, why)`` or ``(None, why)`` if unreachable."""
    import re

    import httpx

    if not re.fullmatch(r"[\w-]{18,}\.[\w-]{4,}\.[\w-]{20,}", token):
        return False, "bot tokens are three runs of letters and digits separated by dots"
    try:
        response = httpx.get(
            "https://discord.com/api/v10/users/@me",
            headers={"Authorization": f"Bot {token}", "User-Agent": f"DiscordBot (dreambot, {__version__})"},
            timeout=10,
        )
        data = response.json()
    except httpx.HTTPError as exc:
        return None, type(exc).__name__
    except ValueError:
        return None, "an unreadable answer"
    if response.status_code == 200 and isinstance(data, dict) and data.get("username"):
        return True, str(data["username"])
    reason = data.get("message") if isinstance(data, dict) else None
    return False, str(reason or f"HTTP {response.status_code}")


def _discord_needs_code_grant(token: str) -> bool:
    """Whether the application has "Requires OAuth2 Code Grant" switched on.

    With it on, Discord refuses the plain add-to-server link ("Integration requires code
    grant"), which looks like the link being wrong. False when Discord cannot be asked.
    """
    import httpx

    try:
        response = httpx.get(
            "https://discord.com/api/v10/applications/@me",
            headers={"Authorization": f"Bot {token}", "User-Agent": f"DiscordBot (dreambot, {__version__})"},
            timeout=10,
        )
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    return isinstance(data, dict) and bool(data.get("bot_require_code_grant"))


def _setup_telegram(config: Config) -> bool:
    """Returns whether anything was changed."""
    from dreambot import secrets

    console.print("[bold]Telegram[/bold]")
    console.print("   In Telegram, message @BotFather, send /newbot, and copy the token it gives you.")
    if secrets.token_source("telegram"):
        console.print("   [green]A token is already stored.[/green] Press Enter to keep it.")
    token = typer.prompt("   Telegram bot token", default="", hide_input=True, show_default=False).strip()
    if not token:
        return False
    valid, detail = _check_telegram_token(token)
    if valid is False:
        console.print(f"   [red]Telegram did not accept that token: {detail}. Nothing was saved.[/red]")
        return False
    secrets.set_token("telegram", token)
    _update_config(config, {"telegram": {"enabled": True}})
    if valid:
        console.print(f"   [green]That is @{detail}. Token saved to your Keychain; Telegram enabled.[/green]")
        console.print(f"   Your bot: https://t.me/{detail}")
    else:
        console.print(
            f"   [yellow]Saved to your Keychain, but Telegram could not be reached to check it ({detail}).[/yellow]"
        )
    return True


def _setup_discord(config: Config) -> bool:
    from dreambot import secrets

    console.print("[bold]Discord[/bold]")
    console.print("   At https://discord.com/developers/applications create an application, open Bot,")
    console.print("   press Reset Token and copy it. No privileged intents are needed.")
    if secrets.token_source("discord"):
        console.print("   [green]A token is already stored.[/green] Press Enter to keep it.")
    token = typer.prompt("   Discord bot token", default="", hide_input=True, show_default=False).strip()
    changed = False
    if token:
        valid, detail = _check_discord_token(token)
        if valid is False:
            console.print(f"   [red]Discord did not accept that token: {detail}. Nothing was saved.[/red]")
            return False
        secrets.set_token("discord", token)
        _update_config(config, {"discord": {"enabled": True}})
        changed = True
        if valid:
            console.print(
                f"   [green]That is {detail}. Token saved to your Keychain; Discord enabled.[/green]"
            )
        else:
            console.print(
                f"   [yellow]Saved to your Keychain, but Discord could not be reached to check it ({detail}).[/yellow]"
            )
    stored = secrets.get_token("discord")
    if stored and (invite := _discord_invite(stored)):
        console.print(
            "   Add the bot to a server you manage (a private one is fine), then use /imagine there"
        )
        console.print(f"   or message the bot directly: {invite}")
        console.print("   (A bot cannot be added as a friend; sharing a server with it is how you reach it.)")
        if _discord_needs_code_grant(stored):
            console.print(
                "   [yellow]This application has “Requires OAuth2 Code Grant” switched on, so Discord will "
                "refuse that link. Switch it off on the Bot page of the developer portal, and save.[/yellow]"
            )
    return changed


def _setup_imessage(config: Config) -> bool:
    from dreambot import launchd
    from dreambot.frontends.imessage.chatdb import ChatDB, ChatDBError

    console.print("[bold]iMessage[/bold]")
    console.print(
        "   Dreambot reads new messages from the Messages database and replies through the Messages app."
    )
    console.print(
        "   [bold]self[/bold]: Messages on this Mac uses your Apple Account — you text yourself to use the bot."
    )
    console.print(
        "   [bold]dedicated[/bold]: Messages on this Mac is signed into an Apple Account just for the bot"
    )
    console.print("   (Messages > Settings > iMessage; the rest of the Mac stays on your own account).")
    mode = (
        typer.prompt("   Mode (self / dedicated, Enter to skip)", default="", show_default=False)
        .strip()
        .lower()
    )
    if mode not in ("self", "dedicated"):
        if mode:
            console.print("   [yellow]Skipped (type self or dedicated).[/yellow]")
        return False
    _update_config(config, {"imessage": {"enabled": True, "mode": mode}})
    console.print(f"   [green]iMessage enabled in {mode} mode.[/green]")
    if launchd.is_loaded():
        # The service reads the database, not this terminal, so it is the service that needs access.
        console.print(
            "   The background service reads the Messages database. Give Full Disk Access to its Python"
        )
        console.print("   (System Settings → Privacy & Security → Full Disk Access → +, then ⌘⇧G and paste):")
        console.print(f"   [bold]{launchd.interpreter()}[/bold]")
        console.print(
            "   Its log says whether that worked: `dreambot service status` shows where the log is."
        )
    else:
        try:
            reader = ChatDB()
            reader.open()
            reader.close()
            console.print("   [green]The Messages database is readable from this terminal.[/green]")
        except ChatDBError as exc:
            console.print(f"   [yellow]{exc}[/yellow]")
        console.print(
            f"   For the background service, also give Full Disk Access to: {launchd.interpreter()}"
        )
    console.print("   When Dreambot starts, macOS asks whether it may control Messages — choose Allow.")
    if mode == "self":
        console.print(
            "   Then text [bold]your own phone number[/bold] from your iPhone: that thread is your Dreambot."
        )
    else:
        console.print(
            "   Then text the bot's Apple Account from your iPhone and send it the code from `dreambot pair`."
        )
    return True


def _setup_web(config: Config) -> bool:
    """Returns whether anything was changed."""
    address = f"http://127.0.0.1:{config.web.port}/"
    console.print("[bold]Web app[/bold]")
    console.print(
        "   Midjourney's website for this Mac: create, organise, Lookbooks and styles in a browser."
    )
    console.print("   It answers this Mac only, and `dreambot web` opens it without any setup.")
    if config.web.enabled:
        console.print(
            f"   [green]It is on:[/green] {address} whenever Dreambot runs. Type `off` to switch it off."
        )
        question = "   Keep it on? (Enter to keep, `off` to switch off)"
    else:
        question = "   Have `dreambot run` and the background service host it too? (yes to switch on)"
    answer = typer.prompt(question, default="", show_default=False).strip().lower()
    if answer in ("y", "yes", "on") and not config.web.enabled:
        _update_config(config, {"web": {"enabled": True}})
        console.print(f"   [green]Switched on.[/green] Once Dreambot is running, open {address}")
        return True
    if answer in ("n", "no", "off") and config.web.enabled:
        _update_config(config, {"web": {"enabled": False}})
        console.print("   [green]Switched off.[/green] `dreambot web` still opens it when you want it.")
        return True
    return False


_SETUP_STEPS = {
    "telegram": _setup_telegram,
    "discord": _setup_discord,
    "imessage": _setup_imessage,
    "web": _setup_web,
}


@app.command()
def setup(
    frontend: Annotated[
        str | None, typer.Argument(help="Set up just one: telegram, discord, imessage or web.")
    ] = None,
) -> None:
    """Connect Dreambot to Telegram, Discord, iMessage and the web app, step by step."""
    config = _config()
    chosen = frontend.lower() if frontend else None
    if chosen is not None and chosen not in _SETUP_STEPS:
        err.print(f"[red]I can set up {', '.join(_SETUP_STEPS)} — not `{_plain(frontend)}`.[/red]")
        raise typer.Exit(2)
    from dreambot import launchd

    console.print("[bold]Dreambot setup[/bold] — press Enter to skip any step.\n")
    changed = False
    for name, step in _SETUP_STEPS.items():
        if chosen in (None, name):
            changed = step(config) or changed
            console.print()

    if changed and launchd.is_loaded():
        # The background service read its configuration when it started.
        _wait_for_jobs(config)
        launchd.restart()
        console.print("[green]The background service was restarted and is using the new setup.[/green]")
        console.print("   dreambot pair                print the code that makes you the bot's owner")
        return
    console.print("[bold]Start it[/bold]")
    console.print("   dreambot run                 run in this terminal")
    console.print("   dreambot service install     run in the background and at login")
    console.print("   dreambot pair                print the code that makes you the bot's owner")


@app.command()
def pair() -> None:
    """Print a one-time code. Send `/pair <code>` to the bot to become its owner."""
    from dreambot.core.access import Access
    from dreambot.core.store import Store

    config = _config()
    store = Store(config.database_path)
    try:
        code = Access(config, store).new_pairing_code()
    finally:
        store.close()
    console.print(f"Send this to your Dreambot within 15 minutes:  [bold]/pair {code}[/bold]")


@access_app.callback(invoke_without_command=True)
def access_default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        access_list()


@access_app.command("list")
def access_list() -> None:
    """Show who owns and who may use this Dreambot."""
    from dreambot.core.access import Access
    from dreambot.core.store import Store

    config = _config()
    store = Store(config.database_path)
    try:
        access = Access(config, store)
        console.print("[bold]Owners[/bold]")
        for user in sorted(access.owners()) or ["(none yet — run `dreambot pair`)"]:
            console.print(f"  {user}")
        console.print("[bold]Allowed[/bold]")
        for user in sorted(access.members()) or ["(nobody else)"]:
            console.print(f"  {user}")
        if config.access.open:
            console.print("[yellow]access.open is on: anyone who can reach the bot may use it.[/yellow]")
    finally:
        store.close()


@access_app.command("allow")
def access_allow(
    user: Annotated[str, typer.Argument(help="e.g. telegram:123456789, discord:42, imessage:+15551234567")],
) -> None:
    """Let someone use this Dreambot."""
    from dreambot.core.access import Access, normalize_user
    from dreambot.core.store import Store

    config = _config()
    store = Store(config.database_path)
    try:
        Access(config, store).allow(user)
    finally:
        store.close()
    console.print(f"{normalize_user(user)} may now use this Dreambot.")


@access_app.command("revoke")
def access_revoke(user: Annotated[str, typer.Argument(help="The id to remove.")]) -> None:
    """Remove someone's access."""
    from dreambot.core.access import Access
    from dreambot.core.store import Store

    config = _config()
    store = Store(config.database_path)
    try:
        removed = Access(config, store).revoke(user)
    finally:
        store.close()
    console.print("Removed." if removed else "They were not on the list (people named in config.toml stay).")


# --------------------------------------------------------------------------------------
# background service
# --------------------------------------------------------------------------------------


#: How long a restart waits for jobs that are running or queued before going ahead anyway.
RESTART_PATIENCE = 300.0


def _wait_for_jobs(config: Config, *, now: bool = False, patience: float = RESTART_PATIENCE) -> None:
    """Before stopping a running service, let the jobs it is working on finish.

    Stopping it mid-job throws the job away, and the person who asked is left looking at a
    progress message that never moves again.
    """
    import time

    from dreambot import launchd
    from dreambot.core.store import Store

    if now or not launchd.is_loaded() or not config.database_path.exists():
        return
    store = Store(config.database_path)
    try:
        waiting = len(store.interrupted_jobs())
        if not waiting:
            return
        console.print(
            f"The service is busy with {waiting} job(s). Waiting for it to finish before restarting "
            "(`--now` restarts at once)…"
        )
        deadline = time.monotonic() + patience
        while store.interrupted_jobs():
            if time.monotonic() >= deadline:
                console.print("[yellow]Still busy after waiting; restarting anyway.[/yellow]")
                return
            time.sleep(2)
    finally:
        store.close()


NowOption = Annotated[
    bool, typer.Option("--now", help="Do not wait for running jobs to finish; they are lost.")
]


@service_app.command("install")
def service_install(now: NowOption = False) -> None:
    """Install and start the background service (a per-user LaunchAgent)."""
    from dreambot import launchd

    config = _config()
    _wait_for_jobs(config, now=now)
    folder = launchd.code_is_guarded()
    if folder:
        console.print(
            f"Dreambot's code is in {folder if folder[0].islower() else f'your {folder} folder'}, "
            "which macOS does not let a background service read without asking on screen."
        )
        console.print("Making a copy for the service to run from (this takes a moment) …")
    try:
        path, runtime = launchd.install(config.log_dir, config.home)
    except RuntimeError as exc:
        err.print(f"[red]Could not start the service:[/red] {_plain(exc)}")
        raise typer.Exit(1) from exc
    console.print(f"Installed and started: {path}")
    if runtime is not None:
        console.print(f"It runs the copy in: {runtime}")
        console.print(
            "[bold]After changing or updating Dreambot, run `dreambot service install` again.[/bold]"
        )
    console.print(f"Logs: {config.log_dir / 'dreambot.log'}")
    console.print(f"For iMessage, grant Full Disk Access to: [bold]{launchd.interpreter()}[/bold]")
    console.print(
        "(System Settings → Privacy & Security → Full Disk Access → +, then press ⌘⇧G and paste that path.)"
    )


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Stop the background service and remove it."""
    from dreambot import launchd

    console.print("Removed." if launchd.uninstall(_config().home) else "The service was not installed.")


@service_app.command("restart")
def service_restart(now: NowOption = False) -> None:
    """Restart the background service (after changing config.toml, for example)."""
    from dreambot import launchd

    if not launchd.is_loaded():
        console.print("The service is not running. Start it with: dreambot service install")
        raise typer.Exit(1)
    _wait_for_jobs(_config(), now=now)
    launchd.restart()
    console.print("Restarted.")


@service_app.command("status")
def service_status() -> None:
    """Show whether the background service is running."""
    from dreambot import launchd

    config = _config()
    info = launchd.status()
    if not info["installed"]:
        console.print("Not installed. Run: dreambot service install")
    elif not info["loaded"]:
        console.print("Installed but not loaded. Run: dreambot service install")
    else:
        console.print(f"Running (state: {info['state'] or 'unknown'}, pid: {info['pid'] or '—'})")
    console.print(f"Logs: {config.log_dir / 'dreambot.log'}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
