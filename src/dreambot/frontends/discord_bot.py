"""Discord frontend: slash commands, buttons and modals, the way Midjourney's own bot works.

The bot connects out to Discord's gateway, so it runs from a Mac behind a home router with no
public address. It needs no privileged intents: slash commands and buttons work everywhere,
plain-text prompts work in direct messages, and in servers you can also @mention it.

As with the other platforms, behaviour lives in
:class:`~dreambot.frontends.base.ChatFrontend`; this file maps Discord's interaction model
onto it. The one Discord-specific trick: while a slash command or button is being handled, the
first message the base class sends is delivered *as the interaction's response*, so the job's
status line appears exactly where Discord expects the reply.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import discord
from discord import app_commands, ui
from PIL import Image

from dreambot.core.actions import Action, ActionKind, decode_action
from dreambot.core.imaging import flatten
from dreambot.core.service import Dreambot, DreambotError
from dreambot.core.settings import STYLIZE_PRESETS, UserSettings
from dreambot.engine.protocol import Op
from dreambot.frontends.base import Buttons, ChatFrontend, Incoming

log = logging.getLogger(__name__)

#: Discord's default upload limit for bots.
UPLOAD_LIMIT = 9_500_000
SETTINGS_PREFIX = "set"
MESSAGE_LIMIT = 1900

_interaction: ContextVar[discord.Interaction | None] = ContextVar(
    "dreambot_discord_interaction", default=None
)
_ephemeral: ContextVar[bool] = ContextVar("dreambot_discord_ephemeral", default=False)

PERMISSIONS = discord.Permissions(
    view_channel=True,
    send_messages=True,
    send_messages_in_threads=True,
    attach_files=True,
    embed_links=True,
    read_message_history=True,
    add_reactions=True,
)


class _Client(discord.Client):
    def __init__(self, frontend: DiscordFrontend, **kwargs: Any):
        super().__init__(**kwargs)
        self.frontend = frontend

    async def setup_hook(self) -> None:
        await self.frontend._sync_commands()

    async def on_ready(self) -> None:
        log.info("discord: connected as %s (in %d server(s))", self.user, len(self.guilds))

    async def on_message(self, message: discord.Message) -> None:
        await self.frontend._on_message(message)

    async def on_interaction(self, interaction: discord.Interaction) -> None:
        if interaction.type is discord.InteractionType.component:
            await self.frontend._on_component(interaction)

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if self.user is not None and payload.user_id == self.user.id:
            return
        await self.frontend.handle_reaction(
            str(payload.user_id), str(payload.channel_id), str(payload.message_id), payload.emoji.name or ""
        )


class _TextModal(ui.Modal):
    """A modal with one or two text boxes that hands the answers back to the frontend."""

    def __init__(
        self, frontend: DiscordFrontend, action: Action, title: str, fields: list[tuple[str, ui.TextInput]]
    ):
        super().__init__(title=title[:45], timeout=600)
        self.frontend = frontend
        self.action = action
        #: (label, input) pairs, in display order.
        self.fields = fields
        for label, text_input in fields:
            self.add_item(ui.Label(text=label[:45], component=text_input))

    async def on_submit(self, interaction: discord.Interaction) -> None:
        values = [str(text_input.value or "").strip() for _, text_input in self.fields]
        await self.frontend._on_modal(interaction, self.action, values)


class DiscordFrontend(ChatFrontend):
    name = "discord"
    has_buttons = True
    can_edit = True
    grid_as_one_image = True
    max_caption = MESSAGE_LIMIT
    max_text = MESSAGE_LIMIT

    def __init__(self, service: Dreambot, token: str, *, guild_ids: Sequence[int] = ()):
        super().__init__(service)
        self._token = token
        self._guild_ids = list(guild_ids)
        self.client = _Client(self, intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self.client)
        self._sent: dict[str, Any] = {}
        self._downloads = service.config.scratch_dir / "discord"
        self._register_commands()

    # ------------------------------------------------------------- lifecycle ----

    async def start(self) -> None:
        await super().start()
        await self.client.login(self._token)
        self._runner = asyncio.get_running_loop().create_task(
            self.client.connect(reconnect=True), name="dreambot-discord"
        )

    async def stop(self) -> None:
        with contextlib.suppress(Exception):
            await self.client.close()
        await super().stop()

    async def _sync_commands(self) -> None:
        try:
            for guild_id in self._guild_ids:
                guild = discord.Object(id=guild_id)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
            await self.tree.sync()
        except discord.HTTPException as exc:
            log.error("discord: could not register slash commands: %s", exc)

    def invite_url(self) -> str | None:
        """The link that adds this bot to a server (available once connected)."""
        application_id = self.client.application_id
        if application_id is None:
            return None
        return discord.utils.oauth_url(
            application_id, permissions=PERMISSIONS, scopes=("bot", "applications.commands")
        )

    # ------------------------------------------------------------- transport ----

    def _view(self, buttons: Buttons | None) -> ui.View | None:
        if not buttons:
            return None
        view = ui.View(timeout=None)
        for row_number, row in enumerate(buttons[:5]):
            for label, payload in row[:5]:
                view.add_item(
                    ui.Button(
                        label=label[:80],
                        custom_id=payload[:100],
                        row=row_number,
                        style=discord.ButtonStyle.secondary,
                    )
                )
        return view

    def _file(self, path: str) -> discord.File:
        source = Path(path)
        if source.stat().st_size <= UPLOAD_LIMIT:
            return discord.File(source, filename=source.name)
        # Too large to upload as a PNG: send a high-quality JPEG instead (the PNG stays on the Mac).
        self._downloads.mkdir(parents=True, exist_ok=True)
        target = self._downloads / f"{source.stem}-{uuid.uuid4().hex[:6]}.jpg"
        with Image.open(source) as image:
            flatten(image).save(target, quality=93)
        return discord.File(target, filename=f"{source.stem}.jpg")

    async def _channel(self, conversation: str) -> Any:
        channel_id = int(conversation)
        return self.client.get_channel(channel_id) or await self.client.fetch_channel(channel_id)

    async def send(
        self,
        conversation: str,
        text: str = "",
        images: Sequence[str] = (),
        buttons: Buttons | None = None,
        *,
        reply_to: str | None = None,
        ask_reply: bool = False,
    ) -> list[str]:
        content = text[:MESSAGE_LIMIT] if text else None
        files = [self._file(path) for path in images[:10]]
        view = self._view(buttons)
        extras: dict[str, Any] = {}
        if files:
            extras["files"] = files
        if view is not None:
            extras["view"] = view

        interaction = _interaction.get()
        if (
            interaction is not None
            and str(interaction.channel_id) == conversation
            and not interaction.is_expired()
        ):
            # Private replies for information and errors; job status and results are public.
            ephemeral = _ephemeral.get() or (text.startswith("⚠️") and not images)
            if not interaction.response.is_done():
                await interaction.response.send_message(content=content, ephemeral=ephemeral, **extras)
                message = await interaction.original_response()
            else:
                message = await interaction.followup.send(
                    content=content or "​", ephemeral=ephemeral, wait=True, **extras
                )
        else:
            channel = await self._channel(conversation)
            if reply_to:
                extras["reference"] = discord.MessageReference(
                    message_id=int(reply_to), channel_id=int(conversation), fail_if_not_exists=False
                )
            message = await channel.send(content=content, **extras)
        self._sent[str(message.id)] = message
        if len(self._sent) > 500:
            for key in list(self._sent)[:250]:
                del self._sent[key]
        return [str(message.id)]

    async def _message(self, conversation: str, ref: str) -> Any:
        if ref in self._sent:
            return self._sent[ref]
        channel = await self._channel(conversation)
        return channel.get_partial_message(int(ref))

    async def edit(self, conversation: str, ref: str, text: str, buttons: Buttons | None = None) -> None:
        message = await self._message(conversation, ref)
        await message.edit(content=text[:MESSAGE_LIMIT], view=self._view(buttons))

    async def delete(self, conversation: str, ref: str) -> None:
        message = await self._message(conversation, ref)
        self._sent.pop(ref, None)
        await message.delete()

    # -------------------------------------------------------------- incoming ----

    async def _save(self, attachments: Sequence[discord.Attachment | None]) -> list[str]:
        saved: list[str] = []
        for attachment in attachments:
            if attachment is None:
                continue
            kind = (attachment.content_type or "").lower()
            suffix = Path(attachment.filename).suffix.lower()
            if not kind.startswith("image/") and suffix not in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".heic",
                ".gif",
            }:
                raise DreambotError(f"`{attachment.filename}` isn't an image.")
            self._downloads.mkdir(parents=True, exist_ok=True)
            target = self._downloads / f"in-{uuid.uuid4().hex[:12]}{suffix or '.img'}"
            await attachment.save(target)
            saved.append(str(target))
        return saved

    async def _on_message(self, message: discord.Message) -> None:
        if message.author.bot or self.client.user is None:
            return
        direct = message.guild is None
        mentioned = self.client.user in message.mentions
        if not direct and not mentioned:
            return
        text = message.content or ""
        for form in (f"<@{self.client.user.id}>", f"<@!{self.client.user.id}>"):
            text = text.replace(form, "")
        try:
            attachments = await self._save(message.attachments)
        except DreambotError as exc:
            await message.channel.send(f"⚠️ {exc}")
            return
        reference = message.reference.message_id if message.reference else None
        await self.handle_message(
            Incoming(
                user=str(message.author.id),
                conversation=str(message.channel.id),
                text=text.strip(),
                attachments=attachments,
                direct=True,  # a DM, or an explicit @mention: either way it is addressed to the bot
                reply_to=str(reference) if reference else None,
            )
        )

    async def _run_text(
        self,
        interaction: discord.Interaction,
        text: str,
        attachments: Sequence[discord.Attachment | None] = (),
        *,
        private: bool = False,
    ) -> None:
        """Handle a slash command by feeding its text form through the shared command router."""
        # People who are not allowed in are told so privately, whatever the command.
        private = private or not self.service.access.is_allowed(self.user_id(str(interaction.user.id)))
        outer = _interaction.set(interaction)
        flag = _ephemeral.set(private)
        try:
            try:
                saved = await self._save(attachments)
            except DreambotError as exc:
                await interaction.response.send_message(f"⚠️ {exc}", ephemeral=True)
                return
            await self.handle_message(
                Incoming(
                    user=str(interaction.user.id),
                    conversation=str(interaction.channel_id),
                    text=text,
                    attachments=saved,
                    direct=True,
                )
            )
            if not interaction.response.is_done():
                await interaction.response.send_message("Done.", ephemeral=True)
        finally:
            _interaction.reset(outer)
            _ephemeral.reset(flag)

    async def _on_component(self, interaction: discord.Interaction) -> None:
        payload = str((interaction.data or {}).get("custom_id", ""))
        user, conversation = str(interaction.user.id), str(interaction.channel_id)
        if payload.startswith(f"{SETTINGS_PREFIX}:"):
            await self._on_settings(interaction, payload)
            return
        action = decode_action(payload)
        full_user = self.user_id(user)
        if action is not None and self.service.access.is_allowed(full_user):
            modal = self._modal_for(full_user, action)
            if modal is not None:
                await interaction.response.send_modal(modal)
                return
        outer = _interaction.set(interaction)
        try:
            notice = await self.handle_button(user, conversation, payload)
            if interaction.response.is_done():
                if notice:
                    await interaction.followup.send(notice, ephemeral=True)
            elif notice:
                await interaction.response.send_message(notice, ephemeral=True)
            else:
                await interaction.response.defer()
        finally:
            _interaction.reset(outer)

    def _modal_for(self, user: str, action: Action) -> _TextModal | None:
        """Discord can ask for text in a modal, which beats "reply to this message"."""
        if self._question_for(user, action) is None:
            return None
        kind = action.kind
        try:
            prompt = self.service.find_job(action.job or "").prompt
        except DreambotError:
            prompt = ""
        paragraph = discord.TextStyle.paragraph
        if kind is ActionKind.VARY_REGION:
            fields = [
                (
                    "Which part?",
                    ui.TextInput(
                        placeholder="Cells like B2-D4, or a description like: the sky", max_length=200
                    ),
                ),
                (
                    "What should be there?",
                    ui.TextInput(style=paragraph, placeholder="a red balloon", max_length=1500),
                ),
            ]
            return _TextModal(self, action, "Vary (Region)", fields)
        if kind is ActionKind.EDIT:
            fields = [
                (
                    "What should change?",
                    ui.TextInput(style=paragraph, placeholder="make it night", max_length=1500),
                )
            ]
            return _TextModal(self, action, "Edit", fields)
        if kind is ActionKind.ZOOM:
            fields = [
                ("Zoom out by (1.0 - 2.0)", ui.TextInput(default="1.5", max_length=5)),
                (
                    "Prompt",
                    ui.TextInput(style=paragraph, default=prompt[:1500], required=False, max_length=1500),
                ),
            ]
            return _TextModal(self, action, "Custom Zoom", fields)
        fields = [("New prompt", ui.TextInput(style=paragraph, default=prompt[:1500], max_length=1500))]
        return _TextModal(self, action, "Remix Prompt", fields)

    async def _on_modal(self, interaction: discord.Interaction, action: Action, values: list[str]) -> None:
        if action.kind is ActionKind.VARY_REGION:
            text = f"{values[0]}: {values[1]}"
        elif action.kind is ActionKind.ZOOM:
            text = f"{values[0]} {values[1]}".strip()
        else:
            text = values[0]
        outer = _interaction.set(interaction)
        try:
            message = Incoming(user=str(interaction.user.id), conversation=str(interaction.channel_id))
            await self._complete(message, action, text)
            if not interaction.response.is_done():
                await interaction.response.defer()
        finally:
            _interaction.reset(outer)

    # -------------------------------------------------------------- settings ----

    def _settings_view(self, user: str) -> tuple[str, ui.View]:
        resolved = self.service.resolved_settings(user)
        view = ui.View(timeout=None)
        on, off = discord.ButtonStyle.success, discord.ButtonStyle.secondary

        def button(label: str, key: str, active: bool, row: int) -> None:
            view.add_item(
                ui.Button(
                    label=label, custom_id=f"{SETTINGS_PREFIX}:{key}", style=on if active else off, row=row
                )
            )

        models = [spec for spec in self.service.catalog.installed(Op.TXT2IMG)][:25]
        if models:
            try:
                current = self.service.planner.default_model(resolved).id
            except Exception:
                current = None
            select = ui.Select(
                custom_id=f"{SETTINGS_PREFIX}:model",
                placeholder="Model",
                row=0,
                options=[
                    discord.SelectOption(
                        label=spec.title[:100],
                        value=spec.id,
                        description=spec.summary[:100],
                        default=spec.id == current,
                    )
                    for spec in models
                ],
            )
            view.add_item(select)
        button("RAW Mode", "raw", resolved.raw, 1)
        for name, value in STYLIZE_PRESETS.items():
            button(f"Stylize {name}", f"stylize:{value}", resolved.stylize == value, 1)
        button("Remix mode", "remix", resolved.remix, 2)
        button("Strong Variation Mode", "variation:strong", resolved.variation == "strong", 2)
        button("Subtle Variation Mode", "variation:subtle", resolved.variation == "subtle", 2)
        button("HD (2K)", "hd", resolved.hd, 3)
        button("Enhance prompts", "enhance", resolved.enhance, 3)
        view.add_item(
            ui.Button(
                label="Reset Settings",
                custom_id=f"{SETTINGS_PREFIX}:reset",
                style=discord.ButtonStyle.danger,
                row=3,
            )
        )
        suffix = f"\nSuffix: `{resolved.suffix}`" if resolved.suffix else ""
        return f"Adjust your settings here. Everything runs locally and privately on this Mac.{suffix}", view

    async def _on_settings(self, interaction: discord.Interaction, payload: str) -> None:
        user = self.user_id(str(interaction.user.id))
        if not self.service.access.is_allowed(user):
            await interaction.response.send_message(self.service.access.refusal(user), ephemeral=True)
            return
        _, key, *rest = payload.split(":")
        settings = self.service.settings(user)
        resolved = self.service.resolved_settings(user)
        try:
            if key == "reset":
                settings = UserSettings(suffix=settings.suffix, option_sets=settings.option_sets)
            elif key == "model":
                values = (interaction.data or {}).get("values") or []
                settings.model = values[0] if values else None
            elif key == "stylize":
                settings.stylize = int(rest[0])
            elif key == "variation":
                settings.variation = rest[0]
            elif key in ("raw", "remix", "hd", "enhance"):
                setattr(settings, key, not getattr(resolved, key))
            self.service.save_settings(user, settings)
        except (DreambotError, ValueError, IndexError) as exc:
            await interaction.response.send_message(f"⚠️ {exc}", ephemeral=True)
            return
        content, view = self._settings_view(user)
        await interaction.response.edit_message(content=content, view=view)

    # -------------------------------------------------------------- commands ----

    def _register_commands(self) -> None:
        # Annotations below must be resolvable from module globals (discord.py evaluates them),
        # hence the fully qualified `discord.Attachment`.
        tree = self.tree

        @tree.command(name="imagine", description="Create images from a prompt")
        @app_commands.describe(
            prompt="What to imagine, with any --parameters (--ar 16:9, --chaos 20, ...)",
            image="Optional image: an image prompt, or the target of --edit / --sref / --oref",
            image2="A second optional image",
            image3="A third optional image",
            image4="A fourth optional image",
        )
        async def imagine(
            interaction: discord.Interaction,
            prompt: str,
            image: discord.Attachment | None = None,
            image2: discord.Attachment | None = None,
            image3: discord.Attachment | None = None,
            image4: discord.Attachment | None = None,
        ) -> None:
            await self._run_text(interaction, f"/imagine {prompt}", [image, image2, image3, image4])

        @tree.command(name="describe", description="Write four prompts for an image")
        @app_commands.describe(image="The image to describe")
        async def describe(interaction: discord.Interaction, image: discord.Attachment) -> None:
            await self._run_text(interaction, "/describe", [image])

        @tree.command(name="blend", description="Blend 2 to 5 images into new ones")
        @app_commands.describe(dimensions="Shape of the result (default: square)")
        @app_commands.choices(
            dimensions=[
                app_commands.Choice(name="Portrait (2:3)", value="portrait"),
                app_commands.Choice(name="Square (1:1)", value="square"),
                app_commands.Choice(name="Landscape (3:2)", value="landscape"),
            ]
        )
        async def blend(
            interaction: discord.Interaction,
            image1: discord.Attachment,
            image2: discord.Attachment,
            image3: discord.Attachment | None = None,
            image4: discord.Attachment | None = None,
            image5: discord.Attachment | None = None,
            dimensions: app_commands.Choice[str] | None = None,
        ) -> None:
            shape = dimensions.value if dimensions else ""
            await self._run_text(
                interaction, f"/blend {shape}".strip(), [image1, image2, image3, image4, image5]
            )

        @tree.command(name="edit", description="Change an image by instruction")
        @app_commands.describe(instruction="What to change, e.g. make it night", image="The image to edit")
        async def edit(interaction: discord.Interaction, instruction: str, image: discord.Attachment) -> None:
            await self._run_text(interaction, f"/edit {instruction}", [image])

        @tree.command(name="shorten", description="Suggest shorter versions of a prompt")
        @app_commands.describe(prompt="The prompt to shorten")
        async def shorten(interaction: discord.Interaction, prompt: str) -> None:
            await self._run_text(interaction, f"/shorten {prompt}")

        @tree.command(name="settings", description="View and adjust your settings")
        async def settings(interaction: discord.Interaction) -> None:
            user = self.user_id(str(interaction.user.id))
            if not self.service.access.is_allowed(user):
                await interaction.response.send_message(self.service.access.refusal(user), ephemeral=True)
                return
            content, view = self._settings_view(user)
            await interaction.response.send_message(content, view=view, ephemeral=True)

        @tree.command(name="show", description="Show one of your jobs again")
        @app_commands.describe(job_id="The job id, e.g. 7QK2MD (leave empty to list recent jobs)")
        async def show(interaction: discord.Interaction, job_id: str | None = None) -> None:
            await self._run_text(interaction, f"/show {job_id or ''}".strip(), private=job_id is None)

        def simple(name: str, description: str) -> None:
            @tree.command(name=name, description=description)
            async def command(interaction: discord.Interaction) -> None:
                await self._run_text(interaction, f"/{name}", private=True)

        simple("info", "Your usage and the queue")
        simple("models", "List the installed models")
        simple("help", "How to use Dreambot")
        simple("params", "Every prompt parameter and how it works locally")
        simple("cancel", "Cancel your running jobs here")
        for name in ("fast", "relax", "turbo", "stealth", "public"):
            simple(name, "Not needed on a local Dreambot (explains why)")

        @tree.command(name="pair", description="Claim this Dreambot with the code from `dreambot pair`")
        async def pair(interaction: discord.Interaction, code: str) -> None:
            await self._run_text(interaction, f"/pair {code}", private=True)

        @tree.command(name="allow", description="Owner: let someone use this Dreambot")
        async def allow(interaction: discord.Interaction, user: discord.User) -> None:
            await self._run_text(interaction, f"/allow discord:{user.id}", private=True)

        @tree.command(name="revoke", description="Owner: remove someone's access")
        async def revoke(interaction: discord.Interaction, user: discord.User) -> None:
            await self._run_text(interaction, f"/revoke discord:{user.id}", private=True)

        prefer = app_commands.Group(name="prefer", description="Personal shortcuts and defaults")

        @prefer.command(
            name="suffix", description="Text added to the end of every prompt (leave empty to clear)"
        )
        async def prefer_suffix(interaction: discord.Interaction, new_value: str | None = None) -> None:
            await self._run_text(interaction, f"/prefer suffix {new_value or ''}".strip(), private=True)

        @prefer.command(name="remix", description="Toggle remix mode")
        async def prefer_remix(interaction: discord.Interaction) -> None:
            await self._run_text(interaction, "/prefer remix", private=True)

        @prefer.command(
            name="variability", description="Switch V buttons between strong and subtle variations"
        )
        async def prefer_variability(interaction: discord.Interaction) -> None:
            await self._run_text(interaction, "/prefer variability", private=True)

        option = app_commands.Group(name="option", description="Custom options", parent=prefer)

        @option.command(
            name="set", description="Create a custom option, used as --name (leave value empty to delete)"
        )
        async def option_set(interaction: discord.Interaction, option: str, value: str | None = None) -> None:
            await self._run_text(
                interaction, f"/prefer option set {option} {value or ''}".strip(), private=True
            )

        @option.command(name="list", description="List your custom options")
        async def option_list(interaction: discord.Interaction) -> None:
            await self._run_text(interaction, "/prefer option list", private=True)

        tree.add_command(prefer)
