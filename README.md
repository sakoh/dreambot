# Dreambot

A free, open-source alternative to Midjourney that runs **completely locally and privately on your Mac**.
Text it from **iMessage**, **Discord** or **Telegram**; it answers with images made on your own GPU by
[MLX](https://github.com/ml-explore/mlx) models.

```
you:       a red fox in the snow, golden hour --ar 16:9 --chaos 20
dreambot:  [grid of four images]   U1 U2 U3 U4 🔄   V1 V2 V3 V4
you:       V3            → four variations of image 3
you:       upscale 2     → image 2 at twice the size
you:       zoom 2x · pan left · region the sky: an aurora · edit: make it night
```

No account, no subscription, no cloud. Prompts and images never leave the Mac except through the chat
app you choose to talk to it with.

> **Status: early (0.1).** The core, the command line and the image pipeline are verified on real models.
> The three chat frontends are complete and tested against simulated platforms, but have **not yet been
> run against the live services**. See [what is verified](#what-is-verified).

## Contents

- [Requirements](#requirements) · [Install](#install) · [Quick start](#quick-start)
- [Models](#models) · [Prompts and parameters](#prompts-and-parameters) · [Working with results](#working-with-results)
- [Chat frontends](#chat-frontends): [Telegram](#telegram) · [Discord](#discord) · [iMessage](#imessage)
- [Privacy](#privacy) · [How it works](#how-it-works) · [Midjourney parity](#midjourney-parity) · [Development](#development)

## Requirements

- A Mac with Apple Silicon. 16 GB of memory runs the small models; 32 GB or more is comfortable.
- macOS 14 or newer, and [uv](https://docs.astral.sh/uv/) (Python 3.11+ is fetched by uv if needed).
- Disk space for models: from 5 GB for a small quantized model to 30 GB+ for the largest.

## Install

```sh
git clone <this repository> dreambot && cd dreambot
uv sync
uv run dreambot doctor        # checks the Mac, MLX and what is installed
```

Everything below is shown as `dreambot ...`; inside the checkout that is `uv run dreambot ...`.

## Quick start

```sh
dreambot models pull z-image-turbo        # ~11 GB; skip if you already use mflux with it
dreambot imagine "an astronaut tending a rooftop garden above a neon city at dusk --ar 16:9"
dreambot chat                             # the full chat experience in your terminal
```

Images are saved to `~/Pictures/Dreambot/<date>/`. On an M5 Ultra a grid of four 1-megapixel images takes
about 30 seconds with Z-Image Turbo.

Dreambot shares the standard Hugging Face cache with [mflux](https://github.com/mflux-community/mflux), so
models you already downloaded for mflux are picked up automatically and nothing is stored twice.

## Models

Dreambot drives **every image model mflux implements** — 52 of them — through one interface, and picks a
suitable installed model for each kind of request. `dreambot models list --all` shows the full table with
licenses; the families are:

| Family | Models | Used for |
| --- | --- | --- |
| Z-Image | turbo, base, ControlNet union | default text-to-image, variations |
| FLUX.2 klein | 4B, 9B, 9B-KV, base | text-to-image, **edits and image references** |
| FLUX 3 klein | distilled, base | text-to-image, edits with up to 10 references |
| Qwen-Image | 2512, Edit 2509/2511, 2.1, 2.1 Turbo, 2.1 ControlNet | text rendering, edits, **mask inpainting** |
| FLUX.1 | schnell, dev, Krea dev, Kontext, Fill, Redux, Depth, ControlNets | LoRA ecosystem, inpainting, blending |
| Krea 2, ERNIE-Image, MOSS-Image, Ideogram 4, Boogu, Iris, Lens | | text-to-image |
| Anima | turbo, aesthetic, base | anime (`--niji`) |
| Ming-Image Design | | graphic design with transparency |
| FIBO, FIBO 2 | generate, edit, background removal | structured prompting |
| SeedVR2 | 3B, 7B | faithful upscaling |

```sh
dreambot models                      # what is installed
dreambot models list --all           # everything supported, with licenses
dreambot models info flux2-klein-4b  # details for one model
dreambot models pull flux2-klein-4b  # download (picks a quantization that suits this Mac)
dreambot models pull seedvr2-3b      # the upscaler behind Upscale (Subtle) and --hd
dreambot models pull describe        # the vision model behind /describe and /shorten
```

A good starting set is **`z-image-turbo`** (generation), **`flux2-klein-4b`** (edits, references, zoom and pan)
and **`seedvr2-3b`** (upscaling): about 27 GB together, all Apache-2.0.

**Licenses.** Dreambot is MIT, but each model has its own license, and many are non-commercial
(`dreambot models list --all` marks them). Every operation has at least one Apache/MIT option.

Other MLX image libraries (for example the generators in `mlx-vlm`, or SDXL ports) are not integrated yet.

## Prompts and parameters

Prompts work the way they do in Midjourney: description first, parameters at the end.

```
a lighthouse at dusk, oil painting --ar 3:2 --stylize 250 --no people --seed 42
```

`dreambot params` (or `/params` in chat) lists every parameter and how it is realised locally:

| | Parameters |
| --- | --- |
| **Native** — same behaviour, done by the local model | `--ar` `--seed` `--sameseed` `--repeat` `--stop` `--raw` `--no` `--edit` |
| **Approximated** — Midjourney's version lives inside its own model | `--stylize` `--chaos` `--weird` `--exp` `--quality` `--hd` `--draft` `--iw` `--sref` `--sw` `--cref` `--cw` `--oref` `--ow` `--niji` |
| **Accepted, no effect locally** | `--v` `--sv` `--fast` `--relax` `--turbo` `--stealth` `--public` `--preview` |
| **Planned** | `--tile` `--p` (personalization) `--video` `--motion` `--loop` `--end` `--bs` |
| **Dreambot extensions** | `--model` `--steps` `--cfg` `--strength` `--lora` `--neg` `--n` |

Also supported: permutations (`a {red, blue} bird --ar {1:1, 16:9}` makes four jobs), multi-prompts with
weights (`space:: ship::2`), image prompts (attach an image or start the prompt with a link), and the em
dash that iPhones substitute for `--`.

**Images as input.** Attach an image and say what it is for:

| | |
| --- | --- |
| `make it night --edit` | change the attached image by instruction |
| `her portrait as an astronaut --oref` | keep this person or object |
| `a cottage on a cliff --sref` | borrow the image's style, not its content |
| `a castle` (image attached, no flag) | use it as an image prompt |

With several attachments, `@1`, `@2` pick which is which: `... --oref @1 --sref @2`.

## Working with results

Under a grid, Discord and Telegram show Midjourney's buttons. Everywhere — and on iMessage, which has no
buttons — you can type the same things:

| Type | What happens |
| --- | --- |
| `U1` … `U4` | pull one image out of the grid at full size |
| `V1` … `V4` | variations of one image (`V2: a new prompt` to remix) |
| `R` | reroll: same prompt, new images |
| `upscale`, `upscale creative` | double the size, faithfully or with re-imagined fine detail |
| `vary subtle`, `vary strong` | variations of a single image |
| `zoom 2x`, `zoom 1.5x` | zoom out |
| `pan left` / `right` / `up` / `down` | extend the picture in one direction |
| `square` | fill a non-square picture out to a square |
| `region B2-D4: a red balloon` | repaint part of the image (send `region` to see the cell grid) |
| `region the sky: an aurora` | …or describe the part instead |
| `edit: make it night` | change the whole image by instruction |
| `fav`, `info`, `cancel` | favourite, show seed and job id, cancel |

Add a number when a job has several images (`upscale 2`). Replying to a specific picture targets that
picture, and replying with plain words edits it.

Commands: `/imagine`, `/describe` (image → four prompts), `/blend` (2–5 images), `/edit`, `/shorten`,
`/settings`, `/prefer suffix`, `/prefer option set|list`, `/prefer remix`, `/prefer variability`, `/show`,
`/info`, `/models`, `/params`, `/cancel`, `/help`.

## Chat frontends

`dreambot setup` walks through all three. A Dreambot is **private by default**: nobody can use it until
its owner allows them.

```sh
dreambot setup             # store tokens in the Keychain, enable frontends
dreambot run               # start the bot in this terminal
dreambot pair              # prints a code; send "/pair <code>" to the bot to become its owner
dreambot access allow telegram:123456789     # let someone else in (or /allow in chat)
dreambot service install   # run in the background and at login
```

### Telegram

1. Message [@BotFather](https://t.me/BotFather), send `/newbot`, copy the token.
2. `dreambot setup` → paste the token (stored in your macOS Keychain, not in a file).
3. `dreambot run`, then `dreambot pair` and send the code to your bot.

In a private chat every message is a prompt; in groups use `/imagine`. It connects by long polling, so
no public address or port forwarding is needed.

### Discord

1. Create an application at <https://discord.com/developers/applications>, open **Bot**, reset and copy the
   token. No privileged intents are required.
2. `dreambot setup` → paste the token; it prints the link that adds the bot to a server.
3. `dreambot run`, then `/pair <code>`.

Slash commands and buttons work in servers; plain-text prompts work in direct messages and when you
@mention the bot.

### iMessage

iMessage has no bot API, so Dreambot reads new messages from the Messages database and replies through the
Messages app. Two modes (`dreambot setup` asks which):

- **self** — Messages on the Mac uses your own Apple Account. From your iPhone, **text your own phone
  number**: that thread is your conversation with Dreambot. You can also summon it inside any conversation
  by starting a message with `/imagine …`. It never answers anyone you have not allowed.
- **dedicated** — Messages on the Mac is signed into an Apple Account that belongs to the bot, and people
  text that account like any contact.

Each image is sent as its own message, so you can reply to a picture or tapback it:
❤️ favourite · 👍 upscale · ‼️ variations · 👎 cancel.

macOS permissions it needs (macOS asks for the second one itself):

1. **Full Disk Access** for the program running Dreambot — your terminal when you use `dreambot run`, and the
   Python binary shown by `dreambot doctor` for the background service. *System Settings → Privacy &
   Security → Full Disk Access.*
2. **Automation → Messages**, on the first reply.

Limits that come from iMessage itself: no buttons, no editing a sent message (so no live progress bar), and
one attachment per message.

## Privacy

- Generation runs with networking disabled: the worker process sets `HF_HUB_OFFLINE=1`, so a model that is
  not on disk produces an error instead of a download.
- The only network access Dreambot starts on its own is `dreambot models pull`, and fetching an image
  *you* reference by link (links to private/local addresses are refused).
- Bot tokens live in the macOS Keychain. Jobs and settings live in a SQLite file under
  `~/Library/Application Support/Dreambot/`. Images live in `~/Pictures/Dreambot/`.
- No telemetry.
- Your prompts and results do pass through whichever chat service you use (Telegram, Discord, Apple), as
  any message does. `dreambot chat` and `dreambot imagine` involve no service at all.

## How it works

```
 iMessage ─┐
 Discord  ─┼─► frontends ─► commands ─► service ─► planner ─► engine client ══► worker process
 Telegram ─┘   (thin)       (text UI)   (queue,     (which model,              (MLX: mflux, mlx-vlm;
 terminal ─┘                             jobs)       which operation)           models stay loaded)
```

- **Worker process.** All MLX inference happens in a subprocess that keeps models loaded between jobs. The
  bot stays responsive, a job can always be cancelled, and stopping the worker (automatically after 15 idle
  minutes) returns all memory to the system.
- **Planner.** Maps each request to an operation — text-to-image, variation, edit, inpaint, upscale — and to
  the best installed model for it, falling back gracefully (for example outpainting through an edit model
  when no inpainting model is installed) and telling you what it did.
- **One codebase for three platforms.** Platform adapters only send, edit and delete messages; everything
  else is shared, so the bots cannot drift apart.

Configuration is optional: `~/Library/Application Support/Dreambot/config.toml`.

```toml
[defaults]
model = "qwen-image"        # instead of the first installed preferred model
count = 4

[prefer]
edit = ["flux2-klein-9b", "flux2-klein-4b"]

[models."flux2-klein-4b"]
weights = "~/models/my-klein-finetune"

[engine]
idle_unload_seconds = 900
```

## Midjourney parity

[docs/PARITY.md](docs/PARITY.md) goes through Midjourney's feature list item by item. In short:

- **Works:** `/imagine` grids, U/V/reroll, variations (subtle/strong, remix), upscales, zoom out, pan, make
  square, vary region, instruction edits, image/style/character references, `/blend`, `/describe`,
  `/shorten`, `/settings`, `/prefer`, permutations, repeat, seeds (exactly reproducible), job history.
- **Approximated:** the aesthetic dials (`--stylize`, `--chaos`, `--weird`, `--exp`) and style codes, which
  steer Midjourney's proprietary model and have no equivalent inside open models.
- **Not yet:** video (Animate), personalization profiles and moodboards, `--tile`, a web gallery, and an
  optional LLM prompt enhancer.

## What is verified

| | How it was checked |
| --- | --- |
| Engine, queue, cancellation, crash recovery, offline loading | automated tests, including real MLX runs |
| Z-Image Turbo, FLUX.2 klein 4B, Qwen-Image 2512, Qwen-Image 2.1, Ming-Image | real weights, in `tests/test_mlx_engine.py` |
| Variations, tiled upscale, edits, references, zoom/pan/square, region edits, describe | real weights, results inspected by eye |
| The other 47 models | class paths, arguments and defaults checked against mflux; **not run** (weights not installed during development) |
| Discord, Telegram | real library objects against a simulated API; **not run against the live service** |
| iMessage | a synthetic Messages database; the AppleScript compiles; **not run against Messages** |

`dreambot models list` marks verified models with ✓.

## Development

```sh
uv sync
uv run pytest                 # ~500 tests, no model weights needed (about a minute)
uv run pytest -m mlx -s       # real inference with whatever models are installed
uv run ruff check . && uv run ruff format .
DREAMBOT_BACKEND=fake uv run dreambot chat    # try everything without any weights
```

`src/dreambot/engine` is the model layer (registry, worker, wire protocol), `core` is platform-independent
behaviour, and `frontends` are the chat adapters. `tests/test_registry.py` pins the model table to the
installed mflux, so an mflux upgrade that adds or renames a model fails a test rather than silently going
unsupported.

## License

MIT for Dreambot itself. Model weights are licensed separately by their authors.

Dreambot is not affiliated with Midjourney, Inc.
