"""The text interface: what a typed message means.

iMessage, Telegram and the local console all funnel plain text through :class:`Commands`, so the
three behave identically. (Discord has native slash commands and buttons, and calls the service
directly for those, but falls back to this router for typed shortcuts.)

A message is, in order of precedence:

1. a slash command — ``/imagine``, ``/describe``, ``/blend``, ``/settings``, ``/prefer`` ...;
2. a shortcut acting on an earlier image — ``U2``, ``V3``, ``zoom 2x``, ``pan left`` ...;
3. in a direct conversation: a reply to a picture is an edit instruction for that picture,
   and anything else is a prompt.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from dreambot.core import imaging, lookbook, render
from dreambot.core.actions import Action, ActionKind, parse_action
from dreambot.core.jobs import Job, JobImage, JobStatus
from dreambot.core.params import PromptError, find_param, parse_ratio
from dreambot.core.service import Dreambot, DreambotError
from dreambot.core.settings import (
    MAX_OPTION_SETS,
    STYLIZE_PRESETS,
    VARIATION_MODES,
    VIDEO_BATCHES,
    UserSettings,
)
from dreambot.core.store import MessageLink

BLEND_SHAPES = {"portrait": (2, 3), "square": (1, 1), "landscape": (3, 2)}
_ON = {"on", "yes", "true", "1", "enable", "enabled"}
_OFF = {"off", "no", "false", "0", "disable", "disabled"}
_NO_OPS = {
    "fast": "Local jobs always run at full speed — there are no fast hours to spend.",
    "relax": "Local jobs always run at full speed — there is no relax queue.",
    "turbo": "Local jobs always run at full speed — there is no turbo mode to pay for.",
    "stealth": "Everything Dreambot makes stays on this Mac. It is always private.",
    "public": "Everything Dreambot makes stays on this Mac; there is no public gallery.",
    "subscribe": "Dreambot is free and runs on your own Mac. There is nothing to subscribe to.",
}


@dataclass
class Reply:
    """Something to send back. Frontends turn it into a message in their own idiom."""

    text: str = ""
    #: Files to send with the text: pictures, or a video.
    images: list[str] = field(default_factory=list)
    #: The job (and image) this reply shows — lets a frontend attach the matching buttons.
    job: Job | None = None
    image: JobImage | None = None
    #: Jobs this message started. Their results arrive later as job events, not in this reply.
    queued: list[Job] = field(default_factory=list)


class Commands:
    def __init__(self, service: Dreambot):
        self.service = service

    async def handle(
        self,
        *,
        user: str,
        channel: str,
        text: str,
        attachments: Sequence[str | Path] = (),
        direct: bool = True,
        allow_paths: bool = False,
        target: MessageLink | None = None,
        slash: str = "/",
    ) -> list[Reply]:
        """Interpret one incoming message.

        ``direct`` says the conversation is with the bot alone, so bare text is a prompt.
        ``target`` is the picture the message replies to, if the platform has replies.
        Problems are returned as replies, never raised: the person should always get an answer.
        """
        try:
            return await self._handle(
                user=user,
                channel=channel,
                text=text.strip(),
                attachments=list(attachments),
                direct=direct,
                allow_paths=allow_paths,
                target=target,
                slash=slash,
            )
        except (DreambotError, PromptError) as exc:
            return [Reply(text=f"⚠️ {exc}")]

    async def _handle(
        self,
        *,
        user: str,
        channel: str,
        text: str,
        attachments: list[str | Path],
        direct: bool,
        allow_paths: bool,
        target: MessageLink | None,
        slash: str,
    ) -> list[Reply]:
        service = self.service
        access = service.access
        is_command = text.startswith("/") and not text.startswith("//")
        if not access.is_allowed(user):
            if is_command and text[1:].partition(" ")[0].partition("@")[0].lower() == "pair":
                return await self._cmd_pair(user=user, rest=text.partition(" ")[2].strip())
            # Stay silent in group conversations unless someone addresses the bot directly.
            return [Reply(text=access.refusal(user))] if direct or is_command else []
        if is_command:
            word, _, rest = text[1:].partition(" ")
            # Telegram appends the bot's name to commands in groups: /imagine@my_bot
            word = word.partition("@")[0].lower()
            handler = getattr(self, f"_cmd_{word}", None)
            if handler is not None:
                return await handler(
                    user=user,
                    channel=channel,
                    rest=rest.strip(),
                    attachments=attachments,
                    allow_paths=allow_paths,
                    target=target,
                    slash=slash,
                    direct=direct,
                )
            if word in _NO_OPS:
                return [Reply(text=_NO_OPS[word])]

        if text.lower().lstrip("/") in ("region", "vary region", "cells"):
            return await self._show_cells(channel=channel, target=target)

        action = parse_action(text)
        if action is not None:
            if target is not None:
                action = action.on(target.job, target.image if action.index is None else None)
            return await self._act(
                user=user, channel=channel, action=action, allow_paths=allow_paths, attachments=attachments
            )

        if text.startswith("/"):
            raise DreambotError(f"I don't know that command. Send {slash}help to see what I can do.")
        if not direct:
            return []
        if target is not None and text:
            # Replying to a picture with words: change that picture as described.
            action = Action(ActionKind.EDIT, target.job, target.image, prompt=text)
            return await self._act(user=user, channel=channel, action=action, allow_paths=allow_paths)
        if not text and not attachments:
            return []
        if not text and attachments:
            hint = (
                f"Nice picture. Tell me what to do with it: add a description to use it as an image prompt, "
                f"say `{slash}describe` to get prompts for it, or `{slash}edit <what to change>`."
            )
            if len(attachments) >= 2:
                hint += f" With several images you can also `{slash}blend` them."
            return [Reply(text=hint)]
        jobs = await service.imagine(
            user=user, channel=channel, prompt=text, attachments=attachments, allow_paths=allow_paths
        )
        return [Reply(queued=jobs)]

    # ------------------------------------------------------------------ actions ----

    async def _act(
        self,
        *,
        user: str,
        channel: str,
        action: Action,
        allow_paths: bool = False,
        attachments: Sequence[str | Path] = (),
    ) -> list[Reply]:
        result = await self.service.act(
            user=user, channel=channel, action=action, allow_paths=allow_paths, attachments=attachments
        )
        if result.jobs:
            return [Reply(queued=result.jobs)]
        if action.kind is ActionKind.SELECT and result.job and result.image:
            return [picked_reply(result.job, result.image)]
        return [Reply(text=result.message or "Done.", job=result.job, image=result.image)]

    async def _show_cells(self, *, channel: str, target: MessageLink | None) -> list[Reply]:
        """Send a picture with the cell grid drawn on it, for choosing a region."""
        service = self.service
        job = service.find_job(target.job) if target else service.store.last_job(channel=channel)
        if job is None or not job.images:
            raise DreambotError("There's no image here to pick a region of yet.")
        index = target.image if target and target.image else None
        if index is None and len(job.images) > 1:
            raise DreambotError("Pick one image first (for example `U2`), then send `region`.")
        image = job.image(index or job.images[0].index)
        overlay = service.config.scratch_dir / "cells" / f"{job.id}_{image.index}.jpg"
        imaging.region_overlay(image.path, overlay)
        return [
            Reply(
                text=(
                    "Name the cells to repaint and what to put there, e.g.\n"
                    f"region {image.index} B2-D4: a red balloon\n"
                    "Or describe the part instead: region the sky: an aurora"
                ),
                images=[str(overlay)],
                job=job,
                image=image,
            )
        ]

    # ----------------------------------------------------------------- commands ----

    async def _cmd_imagine(self, *, user, channel, rest, attachments, allow_paths, **_) -> list[Reply]:
        if rest.lower().startswith("prompt:"):
            rest = rest[len("prompt:") :].strip()
        jobs = await self.service.imagine(
            user=user, channel=channel, prompt=rest, attachments=attachments, allow_paths=allow_paths
        )
        return [Reply(queued=jobs)]

    _cmd_i = _cmd_imagine

    async def _cmd_edit(self, *, user, channel, rest, attachments, allow_paths, target, **_) -> list[Reply]:
        if not rest:
            raise DreambotError("Say what to change, e.g. `/edit make it night` with an image attached.")
        if attachments:
            jobs = await self.service.imagine(
                user=user,
                channel=channel,
                prompt=f"{rest} --edit",
                attachments=attachments,
                allow_paths=allow_paths,
            )
            return [Reply(queued=jobs)]
        action = Action(
            ActionKind.EDIT, target.job if target else None, target.image if target else None, prompt=rest
        )
        return await self._act(user=user, channel=channel, action=action)

    async def _cmd_describe(self, *, user, channel, attachments, target, **_) -> list[Reply]:
        source: str | Path | None = attachments[0] if attachments else None
        if source is None and target is not None:
            job = self.service.find_job(target.job)
            if job.images:
                source = job.image(target.image or job.images[0].index).path
        if source is None:
            raise DreambotError("Attach the image you want described (or reply to one).")
        job = await self.service.describe(user=user, channel=channel, image=source)
        return [Reply(queued=[job])]

    async def _cmd_blend(self, *, user, channel, rest, attachments, **_) -> list[Reply]:
        shape = rest.strip().lower()
        aspect: tuple[int, int] | None = None
        if shape in BLEND_SHAPES:
            aspect = BLEND_SHAPES[shape]
        elif shape:
            aspect = parse_ratio(shape.removeprefix("--ar").strip())
        if len(attachments) < 2:
            raise DreambotError(
                "Attach 2 to 5 images to blend. Add `portrait`, `square` or `landscape` to set the shape."
            )
        job = await self.service.blend(user=user, channel=channel, images=attachments, aspect=aspect)
        return [Reply(queued=[job])]

    async def _cmd_cancel(self, *, user, channel, rest, **_) -> list[Reply]:
        action = Action(ActionKind.CANCEL, rest.strip() or None)
        return await self._act(user=user, channel=channel, action=action)

    async def _cmd_show(self, *, rest, user, **_) -> list[Reply]:
        if not rest:
            recent = self.service.store.recent_jobs(user=user, limit=8)
            if not recent:
                return [Reply(text="No jobs yet.")]
            lines = [f"#{job.id}  {job.status.value:9s} {render.job_title(job, 70)}" for job in recent]
            return [Reply(text="Recent jobs (send /show <id> to see one again):\n" + "\n".join(lines))]
        job = self.service.find_job(rest.split()[0])
        return [job_reply(job)]

    async def _cmd_info(self, *, user, **_) -> list[Reply]:
        info = self.service.info(user)
        lines = [
            "Dreambot — running locally on this Mac",
            f"Default model: {info['model'] or 'none installed'}",
            f"Installed models: {', '.join(info['installed_models']) or 'none'}",
            f"Jobs completed: {info['jobs_done']} (failed: {info['jobs_failed']})",
            f"Queue: {info['queued']} waiting"
            + (f", #{info['running']} running" if info["running"] else ", idle"),
            f"Your jobs in progress: {info['your_jobs_in_progress']}",
            f"User ID: {info['user']}",
            "Cost: free. Visibility: private (nothing leaves this Mac except the messages you see here).",
        ]
        return [Reply(text="\n".join(lines))]

    async def _cmd_models(self, **_) -> list[Reply]:
        service = self.service
        installed = service.catalog.installed()
        if not installed:
            return [
                Reply(text="No models are installed yet. On the Mac, run: dreambot models pull z-image-turbo")
            ]
        lines = ["Installed models (use with --model <name>):"]
        for spec in installed:
            ops = ", ".join(op.value for op in spec.ops) or "specialty"
            lines.append(f"• {spec.id} — {spec.summary} [{ops}]")
        lines.append("\nMore can be added on the Mac with: dreambot models pull <name>")
        if styled := service.planner.loras.entries()[0]:
            lines.append("\nLoRAs — one fine-tuned look each (use with --lora <name>):")
            for entry in styled:
                made_for = f", for {entry.base}" if entry.base else ""
                lines.append(f"• {entry.name} — {entry.title or entry.name}{made_for}")
        lines.append("LoRAs (anime, pixel art, ...) are added on the Mac with: dreambot loras add <source>")
        return [Reply(text="\n".join(lines))]

    async def _cmd_params(self, **_) -> list[Reply]:
        return [Reply(text=render.params_text())]

    async def _cmd_help(self, *, direct, slash, **_) -> list[Reply]:
        return [Reply(text=render.help_text(direct=direct, slash=slash))]

    _cmd_start = _cmd_help
    _cmd_ask = _cmd_help

    async def _cmd_shorten(self, *, user, channel, rest, **_) -> list[Reply]:
        job = await self.service.shorten(user=user, channel=channel, prompt=rest)
        return [Reply(queued=[job])]

    # ------------------------------------------------------------------- access ----

    async def _cmd_pair(self, *, user, rest, **_) -> list[Reply]:
        access = self.service.access
        if access.is_owner(user):
            return [Reply(text="You already own this Dreambot.")]
        if not rest:
            raise DreambotError("Send the code from `dreambot pair`, like: /pair 123456")
        if access.pair(user, rest.split()[0]):
            return [Reply(text="Paired. You're now an owner of this Dreambot — send me a prompt, or /help.")]
        raise DreambotError(
            "That code isn't valid (codes expire after 15 minutes). Run `dreambot pair` again."
        )

    async def _cmd_allow(self, *, user, rest, **_) -> list[Reply]:
        access = self.service.access
        if not access.is_owner(user):
            raise DreambotError("Only an owner can let people in.")
        if not rest or ":" not in rest:
            raise DreambotError("Say who, using the id they were shown, e.g. /allow telegram:123456789")
        access.allow(rest.split()[0])
        return [Reply(text=f"{rest.split()[0]} can now use this Dreambot.")]

    async def _cmd_revoke(self, *, user, rest, **_) -> list[Reply]:
        access = self.service.access
        if not access.is_owner(user):
            raise DreambotError("Only an owner can remove people.")
        if not rest:
            raise DreambotError("Say who, e.g. /revoke telegram:123456789")
        removed = access.revoke(rest.split()[0])
        return [
            Reply(
                text="Removed." if removed else "They weren't on the list (people set in config.toml stay)."
            )
        ]

    async def _cmd_access(self, *, user, **_) -> list[Reply]:
        access = self.service.access
        if not access.is_owner(user):
            raise DreambotError("Only an owner can see who has access.")
        owners = "\n".join(f"  {name}" for name in sorted(access.owners())) or "  (none)"
        members = "\n".join(f"  {name}" for name in sorted(access.members())) or "  (none)"
        return [Reply(text=f"Owners:\n{owners}\nAllowed:\n{members}")]

    async def _cmd_whoami(self, *, user, **_) -> list[Reply]:
        return [Reply(text=f"You are {user} ({self.service.access.role(user) or 'not allowed'}).")]

    # ------------------------------------------------------- settings & prefer ----

    async def _cmd_settings(self, *, user, rest, slash, **_) -> list[Reply]:
        service = self.service
        settings = service.settings(user)
        words = rest.split()
        if not words:
            return [Reply(text=self._settings_text(user, slash))]
        key, value = words[0].lower(), " ".join(words[1:]).strip()
        if key == "reset":
            service.save_settings(user, UserSettings())
            return [Reply(text="Settings reset to the defaults.")]
        if not value:
            raise DreambotError(
                f"`{slash}settings {key}` needs a value. Send `{slash}settings` to see the options."
            )
        lowered = value.lower()
        if key == "model":
            settings.model = None if lowered in ("default", "auto", "none") else value
        elif key in ("lora", "loras"):
            # Checked when saved: every name has to be a LoRA of the library.
            settings.loras = [] if lowered in ("off", "none", "default") else value.replace(",", " ").split()
        elif key == "stylize":
            if lowered in STYLIZE_PRESETS:
                settings.stylize = STYLIZE_PRESETS[lowered]
            elif lowered.isdigit() and 0 <= int(lowered) <= 1000:
                settings.stylize = int(lowered)
            else:
                raise DreambotError("Stylize is low, med, high, very high, or a number from 0 to 1000.")
        elif key in ("variation", "variability"):
            if lowered not in VARIATION_MODES:
                raise DreambotError("Variation mode is `strong` or `subtle`.")
            settings.variation = lowered
        elif key in ("remix", "raw", "hd", "enhance", "personalize"):
            setattr(settings, key, _on_off(lowered, key))
        elif key in ("videos", "bs"):
            if not lowered.isdigit() or int(lowered) not in VIDEO_BATCHES:
                raise DreambotError("Videos per prompt is 1, 2 or 4.")
            settings.videos = int(lowered)
        elif key in ("video-hd", "videohd"):
            settings.video_hd = _on_off(lowered, key)
        elif key in ("picture-first", "picturefirst"):
            settings.picture_first = _on_off(lowered, key)
        elif key in ("count", "images"):
            if not lowered.isdigit() or not 1 <= int(lowered) <= 16:
                raise DreambotError("Images per job is a number from 1 to 16.")
            settings.count = int(lowered)
        else:
            raise DreambotError(f"I don't have a setting called `{key}`. Send `{slash}settings` to see them.")
        service.save_settings(user, settings)
        return [Reply(text=self._settings_text(user, slash))]

    def _settings_text(self, user: str, slash: str) -> str:
        service = self.service
        resolved = service.resolved_settings(user)
        try:
            model = service.planner.default_model(resolved).id
        except Exception:
            model = resolved.model or "none installed"
        stylize_name = next(
            (name for name, value in STYLIZE_PRESETS.items() if value == resolved.stylize), None
        )
        stylize = f"{resolved.stylize}" + (f" ({stylize_name})" if stylize_name else "")
        on = lambda flag: "on" if flag else "off"  # noqa: E731
        lines = [
            "Your settings:",
            f"  model       {model}",
            f"  lora        {', '.join(resolved.loras) or 'none'}",
            f"  stylize     {stylize}",
            f"  raw         {on(resolved.raw)}",
            f"  variation   {resolved.variation}",
            f"  remix       {on(resolved.remix)}",
            f"  hd          {on(resolved.hd)}",
            f"  enhance     {self._enhance_state(resolved)}",
            f"  personalize {self._personalize_state(resolved)}",
            f"  count       {resolved.count} images per job",
            f"  videos      {resolved.videos or service.config.video.count} per prompt",
            f"  video-hd    {on(service.config.video.hd if resolved.video_hd is None else resolved.video_hd)}",
            f"  picture-first {on(self._picture_first(resolved))} (a video from words starts on a picture of them)",
        ]
        if resolved.suffix:
            lines.append(f"  suffix      {resolved.suffix}")
        if resolved.option_sets:
            lines.append("  option sets " + ", ".join(f"--{name}" for name in sorted(resolved.option_sets)))
        lines.append(
            f"\nChange one with `{slash}settings <name> <value>`, e.g. `{slash}settings stylize high`, "
            f"`{slash}settings model qwen-image`, `{slash}settings lora pixel-art` (or `off`), "
            f"`{slash}settings variation subtle`. "
            f"`{slash}settings reset` restores the defaults."
        )
        return "\n".join(lines)

    def _picture_first(self, resolved) -> bool:
        """Whether this person's videos from words alone start on a picture made from the words."""
        if resolved.picture_first is not None:
            return resolved.picture_first
        return self.service.config.video.picture_first

    def _personalize_state(self, resolved) -> str:
        """`on`/`off`, and which Lookbooks every prompt is then given."""
        if not resolved.personalize:
            return "off"
        books = [book for book in map(self.service.store.get_lookbook, resolved.looks) if book]
        names = ", ".join(lookbook.slug(book["name"]) for book in books if not book["deleted"])
        return f"on — every prompt uses {names or 'your favourites (no default Lookbook is chosen)'}"

    async def _cmd_looks(self, *, user, rest, slash, **_) -> list[Reply]:
        """List the Lookbooks, or choose which ones a bare `--p` stands for."""
        service = self.service
        words = rest.replace(",", " ").split()
        if words and words[0].lower() == "default":
            names = words[1:]
            if not names:
                raise DreambotError(
                    f"Say which: `{slash}looks default <name> [<name> ...]`, or `{slash}looks default none`."
                )
            chosen: list[str] = []
            if [name.lower() for name in names] != ["none"]:
                for name in names:
                    book = service.find_lookbook(name)
                    if book is None:
                        raise DreambotError(f"I don't know a Lookbook called `{name}`.")
                    chosen.append(book["id"])
            service.set_default_lookbooks(user=user, lookbook_ids=chosen)
        elif words:
            raise DreambotError(
                f"Send `{slash}looks` to list your Lookbooks, or `{slash}looks default <name>`."
            )
        books = service.lookbooks(user)
        if not books:
            return [
                Reply(
                    text=(
                        "There are no Lookbooks yet. A Lookbook is a set of pictures that stands for a look. "
                        "Reply to a picture with `look: <name>` to start one with it, or make one in the web "
                        "app (`dreambot web` on the Mac)."
                    )
                )
            ]
        lines = ["Lookbooks (★ = used by a bare --p):"]
        for book in books:
            line = f"  {'★' if book['default'] else '·'} {book['slug']} — {book['count']} picture(s)"
            if book["code"]:
                line += f", code {book['code']}"
            if book["style"]:
                line += f"\n      {book['style']['summary']}"
            lines.append(line)
        lines.append(
            f"\nUse one with `--p <name>`; several, each with an optional `::weight`, blend. `--stylize` sets "
            f"how strongly. `{slash}looks default <name>` chooses what a bare `--p` means, and "
            f"`{slash}settings personalize on` applies it to every prompt. Reply to a picture with "
            "`look: <name>` to add it to a Lookbook."
        )
        return [Reply(text="\n".join(lines))]

    def _enhance_state(self, resolved) -> str:
        """`on`/`off`, and why it is not doing anything when it is on but cannot run."""
        if not resolved.enhance:
            return "off"
        if resolved.raw:
            return "on (but raw mode uses prompts as written)"
        if self.service.catalog.vision_model() is None:
            return "on (needs a language model: dreambot models pull describe)"
        return "on — a local language model art-directs short prompts"

    async def _cmd_prefer(self, *, user, rest, slash, **_) -> list[Reply]:
        service = self.service
        settings = service.settings(user)
        words = rest.split()
        if not words:
            raise DreambotError(
                f"Use `{slash}prefer suffix <text>`, `{slash}prefer option set <name> <value>`, "
                f"`{slash}prefer option list`, `{slash}prefer remix` or `{slash}prefer variability`."
            )
        topic = words[0].lower()
        if topic == "suffix":
            suffix = rest[len(words[0]) :].strip()
            if suffix:
                # Validate now so a broken suffix cannot poison every later prompt.
                from dreambot.core.prompt import parse_prompt

                parse_prompt(f"x {suffix}", option_sets=settings.option_sets)
            settings.suffix = suffix
            service.save_settings(user, settings)
            return [
                Reply(
                    text=f"Suffix set: every prompt now ends with `{suffix}`."
                    if suffix
                    else "Suffix cleared."
                )
            ]
        if topic == "remix":
            settings.remix = not service.resolved_settings(user).remix
            service.save_settings(user, settings)
            return [Reply(text=f"Remix mode is now {'on' if settings.remix else 'off'}.")]
        if topic == "variability":
            current = service.resolved_settings(user).variation
            settings.variation = "subtle" if current == "strong" else "strong"
            service.save_settings(user, settings)
            return [Reply(text=f"V buttons now make {settings.variation} variations.")]
        if topic == "option" and len(words) >= 2:
            verb = words[1].lower()
            if verb == "list":
                if not settings.option_sets:
                    return [
                        Reply(
                            text=f"No option sets yet. Create one with `{slash}prefer option set <name> <value>`."
                        )
                    ]
                return [
                    Reply(
                        text="\n".join(
                            f"--{name}: {value}" for name, value in sorted(settings.option_sets.items())
                        )
                    )
                ]
            if verb == "set" and len(words) >= 3:
                name = words[2].lstrip("-").lower()
                value = rest.split(None, 3)[3].strip() if len(words) >= 4 else ""
                if not name.isidentifier():
                    raise DreambotError("Option names use letters, digits and underscores only.")
                if find_param(name) is not None:
                    raise DreambotError(f"`--{name}` is already a built-in parameter; pick another name.")
                if not value:
                    removed = settings.option_sets.pop(name, None)
                    service.save_settings(user, settings)
                    return [Reply(text=f"Removed `--{name}`." if removed else f"There was no `--{name}`.")]
                if len(settings.option_sets) >= MAX_OPTION_SETS and name not in settings.option_sets:
                    raise DreambotError(f"You can keep up to {MAX_OPTION_SETS} option sets.")
                from dreambot.core.prompt import parse_prompt

                parse_prompt(f"x {value}")
                settings.option_sets[name] = value
                service.save_settings(user, settings)
                return [Reply(text=f"`--{name}` now means `{value}`.")]
        raise DreambotError(f"I didn't understand that. Send `{slash}prefer` to see the forms it takes.")


def _on_off(value: str, name: str) -> bool:
    if value in _ON:
        return True
    if value in _OFF:
        return False
    raise DreambotError(f"`{name}` is `on` or `off`.")


def job_reply(job: Job) -> Reply:
    """A finished (or failed) job as a reply: what every frontend sends when a job ends."""
    if job.status is not JobStatus.DONE:
        if job.status.finished:
            return Reply(text=render.failure_text(job), job=job)
        return Reply(text=f"#{job.id} is {job.status.value}: {render.job_title(job, 120)}", job=job)
    if job.texts:
        numbered = "\n\n".join(f"{n}. {text}" for n, text in enumerate(job.texts, start=1))
        notes = "".join(f"\nⓘ {note}" for note in job.notes)
        return Reply(text=f"{numbered}\n\nSend any of these back to me to imagine it.{notes}", job=job)
    caption = render.job_caption(job)
    if job.is_video:
        # Videos have no grid: each is sent as itself (see `ChatFrontend.deliver`).
        return Reply(
            text=caption,
            images=[job.media(image.index) for image in job.images],
            job=job,
            image=job.images[0] if len(job.images) == 1 else None,
        )
    if job.grid and len(job.images) > 1:
        return Reply(text=caption, images=[job.grid], job=job)
    image = job.images[0] if job.images else None
    return Reply(text=caption, images=[image.path] if image else [], job=job, image=image)


def picked_reply(job: Job, image: JobImage) -> Reply:
    """One result pulled out of a job (``U2``): the picture or the video, with its own buttons."""
    what = "Video" if job.clip(image.index) else "Image"
    caption = f"{what} {image.index} of #{job.id} — {render.job_title(job, 160)}"
    return Reply(text=caption, images=[job.media(image.index)], job=job, image=image)


def split_command(text: str) -> list[str]:
    """Shell-like splitting that never raises on unbalanced quotes."""
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()
