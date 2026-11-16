# Dreambot

A free, open-source alternative to Midjourney that runs **completely locally and privately on your Mac**.
Text it from **iMessage**, **Discord** or **Telegram**, or open its **web app**; it answers with images made
on your own GPU by [MLX](https://github.com/ml-explore/mlx) models.

```
you:       a red fox in the snow, golden hour --ar 16:9 --chaos 20
dreambot:  [grid of four images]   U1 U2 U3 U4 🔄   V1 V2 V3 V4
you:       V3            → four variations of image 3
you:       upscale 2     → image 2 at twice the size
you:       zoom 2x · pan left · region the sky: an aurora · edit: make it night
```

No account, no subscription, no cloud. Prompts and images never leave the Mac except through the chat
app you choose to talk to it with.

> **Status: early (0.1).** The core, the command line and the image pipeline are verified on real models,
> and all three bots (Telegram, Discord and iMessage) have been used on the live services. Not every
> button and command has been pressed on each of them yet. The web app has been driven end to end in
> WebKit. See [what is verified](#what-is-verified).

## Contents

- [Requirements](#requirements) · [Install](#install) · [Quick start](#quick-start)
- [Models](#models) · [Prompts and parameters](#prompts-and-parameters) · [The prompt enhancer](#the-prompt-enhancer) · [Styles and Lookbooks](#styles-and-lookbooks) · [Working with results](#working-with-results)
- [Web app](#web-app) · [Chat frontends](#chat-frontends): [Telegram](#telegram) · [Discord](#discord) · [iMessage](#imessage)
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
dreambot web                              # the web app, in your browser
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
| **Art direction** — done by [the prompt enhancer](#the-prompt-enhancer) | `--stylize` `--chaos` `--weird` `--exp` |
| **Approximated** — Midjourney's version lives inside its own model | `--quality` `--hd` `--draft` `--iw` `--sref` `--sw` `--sv` `--p` `--cref` `--cw` `--oref` `--ow` `--niji` |
| **Accepted, no effect locally** | `--v` `--fast` `--relax` `--turbo` `--stealth` `--public` `--preview` |
| **Planned** | `--tile` `--video` `--motion` `--loop` `--end` `--bs` |
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

With several attachments, `@1`, `@2` pick which is which: `... --oref @1 --sref @2`. Styles blend by
weight — see [Styles and Lookbooks](#styles-and-lookbooks).

## The prompt enhancer

Midjourney's look comes from inside its model: a short prompt comes back composed, lit and
colour-graded. Open models draw what the prompt says and leave the rest to chance, and their four
images of one prompt tend to look alike.

So for a short prompt, a small language model on your Mac acts as art director. For each image it
writes one line — the shot, the light, the palette, the finish — and that line is added after your
prompt:

```
you typed    a red fox in the snow
image 1      a red fox in the snow. Low angle close-up, soft morning light filtering through trees,
             deep red fur contrasting snow white ground, sharp focus on fox's face
image 2      a red fox in the snow. High angle side view, distant trees silhouetted against sky,
             warm sunlight on fox's back, soft focus background
```

Your own words are never rewritten or dropped — it only adds — so nothing you asked for is lost and
quoted text comes through letter for letter. `info` on any image shows the line it was given.

| Dial | What it tells the art director |
| --- | --- |
| `--stylize 0…1000` | How bold to be. 0 uses your prompt as written; 100 (the default) is balanced; 750 is theatrical. |
| `--chaos 0…100` | How different the four images are. At 0 they share a look and differ in the shot; higher values vary light, palette and setting; from 75 each image is a different medium. |
| `--weird 0…3000` | Adds a strange twist to the scene itself. |
| `--exp 0…100` | More intricate detail and more dynamic composition. |
| `--raw` | No art direction for this prompt. |

It leaves alone prompts that are already detailed (45 words or more at default settings), flat
graphics such as logos and icons, and anything with an image attached. It needs the vision model
(`dreambot models pull describe`); it adds about a second per job, and without it the same dials fall
back to simple rule-based phrases. It is on by default on Macs with 24 GB of memory or more — switch
it with `/settings enhance on|off`.

## Styles and Lookbooks

Midjourney takes a picture's *style* from several places at once — reference images, style codes,
moodboards — and mixes them by weight inside its own model. Open models have no such input, so Dreambot
reduces every source of style to one description, a **style fingerprint**, and does the mixing on that:

| Part | What it is | Where it comes from |
| --- | --- | --- |
| Words | medium, rendering, palette, light, texture, mood | the local vision model, asked how a picture is made and never what it shows; or decoded from a style code |
| Numbers | the dominant colours and the tonal balance | measured from the pixels |

A fingerprint never says what a picture is *of*, which is what lets it be applied to a prompt about
something else.

```
a cottage on a cliff --sref @1                    one reference image
a cottage on a cliff --sref @1::3 @2::1           two, three parts to one
a cottage on a cliff --sref 4821 @1 --sw 300      a style code and an image, applied strongly
a cottage on a cliff --sref random --repeat 4     four jobs, each with a style of its own
a cottage on a cliff --sref dusty-noir            a style you saved under a name
```

**Blending.** The colours and the tonal balance are a true weighted mix. Words cannot be averaged, so
the heaviest style sets the medium, the rendering and the light; a style holding at least 30% of the
weight adds its texture and mood; and when no style clearly leads, the medium is named as a cross of
the two heaviest.

**What the model is shown.** The words go into the prompt, so a style works on any model. A model that
takes reference images (FLUX.2 klein, Qwen-Image 2.1) is also shown a *swatch board*: a sheet of
shuffled pieces cut from the style pictures, each picture getting pieces in proportion to its weight.
It is never shown the pictures whole. That is the result of testing, not taste — see
[what is verified](#what-is-verified): shown whole, a reference put its own content into the result
(a hand, a caption) and on its own barely carried its style over.

| | |
| --- | --- |
| `--sw 0…1000` | How much of the fingerprint is applied. Under 50 a hint; from 50 the medium, rendering, palette and light; from 200 the texture and mood too; from 500 it is insisted on. |
| `--sv 1` | The original behaviour: a code is a fixed recipe, and an image is shown whole to an edit model. Codes from before keep their look. |
| `--sv 2` | Words and swatch board. The default. |
| `--sv 3` | Words only, staying on the model you chose. |

A style goes along with the model you chose, typed or as your default, rather than moving the job to
another one: on a model that cannot take reference images it is applied in words.

**Style codes** are Dreambot's own: a number from 1 to 4,294,967,295 that decodes to one of 5.4 million
styles. They are shareable between Dreambot users and mean nothing to Midjourney. The web app's Styles
page lets you browse, search ("anime pastel", "film grain"), like and try them.

### Lookbooks

A Lookbook is a named set of pictures that stands for a look — Midjourney's moodboard. Make one in the
web app (upload, paste a link, or pick from your own images), or from a chat by replying to a picture
with `look: autumn diner`.

```
a harbour at dawn --p autumn-diner                that Lookbook
a harbour at dawn --p autumn-diner::2 noir        two, blended two to one
a harbour at dawn --p                             your default Lookbooks
a harbour at dawn --p --stylize 400               ...applied strongly
```

- Where `--sref` is one look applied as it is, a Lookbook is a range: each image of a grid is led by a
  different picture of the book (60% that picture, 40% the rest), so a varied book gives varied results
  and a tight one consistent results. The same seed gives the same leads.
- `--stylize` sets how strongly it applies; `--sw` is for style references and leaves it alone. With
  both, the style reference decides the medium and the Lookbook colours it.
- Every state of a Lookbook has a **code**. A prompt that names a book is recorded with the code of the
  version used, and that code keeps meaning those pictures after you change or delete the book, so an
  old prompt can always be run again.
- `/looks` lists them. `/looks default autumn-diner` chooses what a bare `--p` means, and
  `/settings personalize on` (the **P** button in the web app) adds it to every prompt.
- Pictures you favourite form a `liked` Lookbook: `--p liked`, and what a bare `--p` uses until you
  choose a default.

`--profile` and `--personalize` are the same parameter, as in Midjourney; `--look` is Dreambot's name
for it. Lookbooks and saved styles belong to the whole Dreambot, so one made in the web app works from
iMessage too.

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
| `look: autumn diner` | add the image to that Lookbook (and start it, if it is new) |
| `fav`, `info`, `cancel` | favourite, show seed and job id, cancel |

Add a number when a job has several images (`upscale 2`). Replying to a specific picture targets that
picture, and replying with plain words edits it.

Commands: `/imagine`, `/describe` (image → four prompts), `/blend` (2–5 images), `/edit`, `/shorten`,
`/settings`, `/looks`, `/prefer suffix`, `/prefer option set|list`, `/prefer remix`, `/prefer variability`,
`/show`, `/info`, `/models`, `/params`, `/cancel`, `/help`.

## Web app

```sh
dreambot web                 # start it and open it in your browser
dreambot setup web           # have `dreambot run` and the background service host it too
```

Midjourney's website, for one Mac: <http://127.0.0.1:7373/>.

| Page | What is there |
| --- | --- |
| **Create** | The prompt bar, with drop zones for an image prompt, a style reference and a subject reference; an uploads library; settings (shape, model, stylize, variety, weirdness, raw, HD); the **P** button. Below it, everything that has been made, newest first, with jobs drawing themselves as they run. |
| **Lightbox** | Click an image: vary (subtle, strong, remix), upscale (subtle, creative), zoom, pan, make square, edit, rerun, describe; copy the prompt, seed or job id; download; like; file it in a folder or a Lookbook; use it in the next prompt. ← → move through the feed. |
| **Organize** | Every image, to search and to filter by liked, model, kind of job, Lookbook, folder and day. Select many (shift-click for a range) to like, trash, file or download as a zip. Trash only hides. |
| **Lookbooks** | Make, fill, rename and delete Lookbooks; choose your defaults; see what Dreambot reads in a book. |
| **Styles** | Explore style codes (shuffle, search, like, the styles one step away, try one on your latest prompt), your saved styles, and the **blend bench**: put codes, pictures and Lookbooks together, set their weights, read what the blend comes to before anything is rendered, then try it or save it under a name. |

Pictures made from iMessage, Discord and Telegram appear in it too: there is one database. When the
background service hosts the web app they appear as they are made; a separate `dreambot web` shows them
when the page is reloaded.

**Who can reach it.** It listens on `127.0.0.1` and has no accounts: whoever opens it is at the Mac. A
local web server can still be reached by any web page open in the same browser, so the app checks that
a request is addressed to it by its own name (which stops another site pointing its own name at your
Mac), requires a session cookie that only its own page is given and that no other site's request
carries, and accepts changes only from its own page. Images are served by id, never by file path.
It is not built to be put on a network: it has no password, and using it from a phone is not supported
yet.

## Chat frontends

`dreambot setup` walks through all three (`dreambot setup telegram` does just one). A Dreambot is
**private by default**: nobody can use it until its owner allows them.

```sh
dreambot setup             # store tokens in the Keychain, enable frontends
dreambot run               # start the bot in this terminal
dreambot pair              # prints a code; send "/pair <code>" to the bot to become its owner
dreambot access allow telegram:123456789     # let someone else in (or /allow in chat)
dreambot service install   # run in the background and at login
```

The background service is a login item (a launchd agent): it starts when you log in, restarts if it
crashes, and gives model memory back after 15 idle minutes. `dreambot service status`, `restart` and
`uninstall` manage it, and `dreambot setup` restarts it for you when you change something. It writes to
`~/Library/Logs/Dreambot/dreambot.log`.

The bots answer only while the Mac is on, awake and logged in to your account: a login item does not run
at the login window, and a sleeping Mac is not reachable. To keep a bot available, stop the Mac from
sleeping (System Settings → Energy, or `caffeinate`).

If your copy of Dreambot is inside Desktop, Documents or Downloads, the service runs from a copy that
`service install` makes under `~/Library/Application Support/Dreambot/runtime`. macOS does not let a
background program read those folders without a confirmation on screen, and a service waiting for a
click nobody sees is a service that is down. **Run `dreambot service install` again after updating
Dreambot** to refresh that copy.

### Telegram

1. Message [@BotFather](https://t.me/BotFather), send `/newbot`, copy the token.
2. `dreambot setup telegram` → paste the token. Dreambot asks Telegram whose token it is, shows you the
   bot's @name, and stores the token in your macOS Keychain (not in a file).
3. `dreambot run` prints a pairing code; send `/pair <code>` to your bot to become its owner.

In a private chat every message is a prompt; in groups use `/imagine`. It connects by long polling, so
no public address or port forwarding is needed.

### Discord

1. Create an application at <https://discord.com/developers/applications>, open **Bot**, reset and copy the
   token. No privileged intents are required. Leave **Requires OAuth2 Code Grant** off, or Discord refuses
   the add-to-server link with "Integration requires code grant".
2. `dreambot setup discord` → paste the token. Dreambot asks Discord whose token it is, stores it in
   your Keychain and prints the link that adds the bot to a server.
3. Open that link and add the bot to a server you manage (a private one is fine).
4. `dreambot run` (or the background service), then `/pair <code>` in that server.

Slash commands and buttons work in servers; plain-text prompts work in direct messages and when you
@mention the bot. A bot cannot be added as a friend: you can message it once you share a server with it.

### iMessage

iMessage has no bot API, so Dreambot reads new messages from the Messages database and replies through the
Messages app. Two modes (`dreambot setup` asks which):

- **self** — Messages on the Mac uses your own Apple Account. From your iPhone, **text your own phone
  number**: that thread is your conversation with Dreambot. You can also summon it inside any conversation
  by starting a message with `/imagine …`. It never answers anyone you have not allowed.
- **dedicated** — Messages on the Mac is signed into an Apple Account that belongs to the bot, and people
  text that account like any contact. Messages has its own sign-in (*Messages → Settings → iMessage*), so
  the rest of the Mac stays on your own account; only your personal iMessages stop arriving on this Mac.

Each image is sent as its own message, so you can reply to a picture or tapback it:
❤️ favourite · 👍 upscale · ‼️ variations · 👎 cancel.

macOS permissions it needs (macOS asks for the second one itself):

1. **Full Disk Access** for the program running Dreambot — your terminal when you use `dreambot run`, and the
   Python binary shown by `dreambot setup imessage` for the background service. *System Settings → Privacy &
   Security → Full Disk Access.*
2. **Automation → Messages.** Dreambot asks for it when it starts, so the prompt appears while you are
   setting up rather than in the middle of its first reply.

When it starts, the log says whether both worked, whether Messages is signed in to iMessage, and then what
arrives and what is sent: the kind of thing only (a message, a picture, a tapback), never the words or
who sent them.

Limits that come from iMessage itself: no buttons, no editing a sent message (so no live progress bar), and
one attachment per message.

## Privacy

- Generation runs with networking disabled: the worker process sets `HF_HUB_OFFLINE=1`, so a model that is
  not on disk produces an error instead of a download.
- The only network access Dreambot starts on its own is `dreambot models pull`, and fetching an image
  *you* reference by link (links to private/local addresses are refused).
- Bot tokens live in the macOS Keychain. Jobs and settings live in a SQLite file under
  `~/Library/Application Support/Dreambot/`, next to the pictures you put in Lookbooks and saved
  styles. Images live in `~/Pictures/Dreambot/`.
- The web app listens on `127.0.0.1` only and has no accounts: it is for the person at the Mac.
  Other web pages open in the same browser are kept out of it (see [Web app](#web-app)).
- No telemetry.
- Your prompts and results do pass through whichever chat service you use (Telegram, Discord, Apple), as
  any message does. `dreambot chat` and `dreambot imagine` involve no service at all.

## How it works

```
 iMessage ─┐
 Discord  ─┼─► frontends ─► commands ─► service ─► planner ─► engine client ══► worker process
 Telegram ─┤   (thin)       (text UI)   (queue,     (which model,              (MLX: mflux, mlx-vlm;
 terminal ─┤                    ▲        jobs)       which operation)           models stay loaded)
 web app  ─┴────────────────────┘ (buttons are the same actions as chat buttons)
```

- **Worker process.** All MLX inference happens in a subprocess that keeps models loaded between jobs. The
  bot stays responsive, a job can always be cancelled, and stopping the worker (automatically after 15 idle
  minutes) returns all memory to the system.
- **Planner.** Maps each request to an operation — text-to-image, variation, edit, inpaint, upscale — and to
  the best installed model for it, falling back gracefully (for example outpainting through an edit model
  when no inpainting model is installed) and telling you what it did.
- **One codebase for every platform.** Platform adapters only send, edit and delete messages, and the web
  app sends the same actions a chat button does; everything else is shared, so they cannot drift apart.

Configuration is optional: `~/Library/Application Support/Dreambot/config.toml`.

```toml
[defaults]
model = "qwen-image"        # instead of the first installed preferred model
count = 4
enhance = true              # the prompt enhancer for everyone (default: on with 24 GB or more)

[prefer]
edit = ["flux2-klein-9b", "flux2-klein-4b"]

[models."flux2-klein-4b"]
weights = "~/models/my-klein-finetune"

[engine]
idle_unload_seconds = 900

[web]
enabled = true              # `dreambot run` and the background service host the web app
port = 7373
```

## Midjourney parity

[docs/PARITY.md](docs/PARITY.md) goes through Midjourney's feature list item by item. In short:

- **Works:** `/imagine` grids, U/V/reroll, variations (subtle/strong, remix), upscales, zoom out, pan, make
  square, vary region, instruction edits, image/style/character references, `/blend`, `/describe`,
  `/shorten`, `/settings`, `/prefer`, permutations, repeat, seeds (exactly reproducible), job history, and
  a web app with the create page, lightbox, gallery, folders and trash.
- **Approximated:** the aesthetic dials (`--stylize`, `--chaos`, `--weird`, `--exp`), which steer
  Midjourney's proprietary model; here a local language model art-directs the prompt instead. Style
  references blend by weight, moodboards exist as Lookbooks and there is a style explorer — all by
  Dreambot's own mechanism, and its style codes are its own, not Midjourney's.
- **Not yet:** video (Animate), personalization by ranking images, the Style Creator, the Editor canvas,
  `--tile`.

## What is verified

| | How it was checked |
| --- | --- |
| Engine, queue, cancellation, crash recovery, offline loading | automated tests, including real MLX runs |
| Z-Image Turbo, FLUX.2 klein 4B, Qwen-Image 2512, Qwen-Image 2.1, Ming-Image, SeedVR2 3B | real weights, in `tests/test_mlx_engine.py` |
| Variations, upscales (SeedVR2 and tiled), `--hd`, edits, references, zoom/pan/square, region edits, describe | real weights, results inspected by eye |
| The prompt enhancer | real language model (Qwen2.5-VL 7B) and real images, compared with and without |
| Style references (`--sref`) | real weights (Z-Image Turbo, FLUX.2 klein 4B; Qwen-Image 2.1 for one comparison) and the real vision model, results inspected by eye. Three prompts across the three style versions with one reference: shown the picture whole (`--sv 1`), none of the three took its style and one had the reference's hand and gun in it; with words and a swatch board (`--sv 2`) and with words alone (`--sv 3`), all three took the style and none took its content. **Two reference pictures in all**, both generated images: photographs and pictures with a lot of text in them have not been tried |
| Blending by weight | real weights, two prompts: the same two references at 3:1, 1:1 and 1:3 give a visible progression from one style to the other; a style code with a picture gives a plausible mix. More than two references: automated tests only |
| What the vision model says about a style | nine real images. It names a medium and five short phrases, drops the labels about half the time (the parser then reads by position), and calls most generated images "digital painting". Asked to fuse two styles into one, it joined their phrases with "and" or copied the worked example, so blending is done by rule |
| Lookbooks | real weights: a four-picture book gave each image of a grid a different lead and a visibly different look; a one-picture book at `--stylize` 0, 20, 100 and 400 gave no look, a touch, the look, and more of it. With a book whose lead was photograph-like the three strengths looked alike, because the look was already close to the model's own. Larger books, several books at once, the `liked` book: automated tests only |
| Web app | every route against the real service on the stand-in backend (automated, including who is let in); the pages driven end to end in WebKit, the engine Safari uses, on the stand-in backend, and one real job with a Lookbook. **Not tried** in Safari itself, Chrome or Firefox, nor on a phone |
| `/looks` and `look: <name>` from the chat apps | automated tests through the shared command layer; **not run on the live services** |
| The other 46 models | class paths, arguments and defaults checked against mflux; **not run** (weights not installed during development) |
| Telegram | **live**: pairing, prompts with the enhancer, grids with buttons, variations, `/describe` with a photo. The other actions: real library objects against a simulated API |
| Discord | **live**: adding the bot to a server, `/pair`, `/imagine` with the enhancer, variation buttons. The other commands and buttons: the real command tree and views, with stand-ins for Discord's interactions and channels |
| iMessage | **live** (macOS 27, a dedicated Apple Account): reading the Messages database, pairing, prompts with the enhancer, typed shortcuts, `/describe` with a photo from an iPhone, sending pictures. Replies to a picture and tapbacks: a synthetic Messages database only (the first live run found and fixed a bug there) |

`dreambot models list` marks verified models with ✓.

## Development

```sh
uv sync
uv run pytest                 # ~650 tests, no model weights needed (about a minute)
uv run pytest -m mlx -s       # real inference with whatever models are installed
uv run ruff check . && uv run ruff format .
DREAMBOT_BACKEND=fake uv run dreambot chat    # try everything without any weights
DREAMBOT_BACKEND=fake uv run dreambot web     # ...or in the browser
```

`src/dreambot/engine` is the model layer (registry, worker, wire protocol), `core` is platform-independent
behaviour, and `frontends` are the chat adapters and the web app. The web app has no build step: its pages
are plain ES modules in `frontends/web/static`, and the one library they use is vendored there. `tests/test_registry.py` pins the model table to the
installed mflux, so an mflux upgrade that adds or renames a model fails a test rather than silently going
unsupported.

## License

MIT for Dreambot itself. Model weights are licensed separately by their authors.

Dreambot is not affiliated with Midjourney, Inc.
