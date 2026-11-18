# Midjourney parity

This walks through Midjourney's feature set as of late 2026 (default model V8.2) and says what Dreambot
does for each item. It is meant to be honest rather than flattering.

| Mark | Meaning |
| --- | --- |
| ✅ | Works, with the same behaviour |
| 🟡 | Works, but by a different mechanism — results are in the same spirit, not identical |
| ⚪ | Accepted for compatibility; there is nothing for it to do locally |
| ⏳ | Not built yet |
| ➖ | Does not apply to a private, local tool |

Whether a feature works on *your* Mac also depends on which models are installed; Dreambot tells you what
to `dreambot models pull` when something is missing.

## Commands

| Midjourney | Dreambot | Notes |
| --- | --- | --- |
| `/imagine` | ✅ | Grid of four (change with `--n` or `/settings count`). On Discord it also takes up to four image attachments. |
| `/blend` (2–5 images, dimensions) | 🟡 | Through FLUX.1 Redux when installed, otherwise an edit model asked to blend its references. |
| `/describe` | ✅ | Four prompts from a local vision model, with 1–4 buttons to imagine them. |
| `/shorten` | 🟡 | Five shorter versions from the local language model (or mechanical trimming without one). No per-token importance display. |
| `/settings` | ✅ | Model, stylize presets, RAW, remix, variation mode, HD, prompt enhancer, images per job. A button panel on Discord. |
| `/prefer suffix`, `/prefer option set`, `/prefer option list`, `/prefer remix`, `/prefer variability` | ✅ | |
| `/show <job id>` | ✅ | Job ids are short (`7QK2MD`). Without an id it lists recent jobs. |
| `/info` | ✅ | Queue, job counts, installed models. |
| `/fast`, `/relax`, `/turbo`, `/stealth`, `/public`, `/subscribe` | ⚪ | Each explains why it is not needed: local jobs run at full speed and everything is private and free. |
| `/ask`, `/help` | ✅ | Both show the help text. |
| `/tune`, `/list_tuners` | ➖ | Retired by Midjourney itself. |
| `/invite`, `/daily_theme`, `/docs`, `/faq`, `/synonyms`, `/userid` | ➖ | Community-server helpers. |
| `/list_personalize_codes` | 🟡 | `/looks` lists every Lookbook with its current code. Older codes are in the prompts that used them. |
| — | `/edit`, `/models`, `/params`, `/looks`, `/pair`, `/allow`, `/revoke`, `/cancel` | Dreambot additions. |

## Buttons and reactions

Buttons exist on Discord and Telegram. Everything is also available as typed text on all three
platforms, which is how iMessage works.

| Midjourney | Dreambot | Typed form | Notes |
| --- | --- | --- | --- |
| U1–U4 | ✅ | `U2` | Returns the full-size image with the single-image buttons. |
| V1–V4 | ✅ | `V2` | Follows the variation mode setting; opens a prompt box when remix is on. |
| Reroll 🔄 | ✅ | `R` | Rerolling one result of a `--repeat` runs it once, as in Midjourney. |
| Vary (Subtle) / (Strong) | ✅ | `vary subtle`, `vary strong` | Controlled by noise level, so it behaves the same across models and sizes. |
| Vary (Region) | 🟡 | `region B2-D4: …`, `region the sky: …` | No drawing canvas: choose cells on a labelled grid, or describe the part. Mask inpainting when such a model is installed, otherwise an edit model plus restoring everything outside the region. |
| Upscale (Subtle) | ✅ | `upscale` | SeedVR2: 2× in about 30 s, faithful to the pixel. Without it, falls back to a low-noise re-render and says so. |
| Upscale (Creative) | 🟡 | `upscale creative` | Re-renders the enlarged image tile by tile with the model that made it. |
| Zoom Out 2x / 1.5x / Custom | ✅ | `zoom 2x`, `zoom 1.3x` | The original pixels are kept exactly; colour is matched across the seam. |
| Make Square | ✅ | `square` | |
| Pan ⬅️➡️⬆️⬇️ | ✅ | `pan left` | Extends by half the picture's size per step. |
| Favorite ❤️ | ✅ | `fav 2` | Stored with the job. Favourites form the `liked` Lookbook: `--p liked`, and what a bare `--p` uses when no default is chosen. |
| Envelope ✉️ (seed and job id) | ✅ | `info 2` | Also the ✉️ / ❓ reaction. |
| Cancel ❌ | ✅ | `cancel` | Stops mid-generation within a step. |
| Animate (High/Low motion), Extend | ⏳ | | Video is not built yet. |
| Web | 🟡 | | There is no button, but every job from every chat app is in the web app (`dreambot web`). |
| — | Edit | `edit: make it night` | Dreambot addition: instruction edit of a finished image. |
| — | Add to a Lookbook | `look 2: autumn diner` | Dreambot addition: starts the Lookbook if there is none by that name. |

Tapbacks and reactions on an image: ❤️ favourite, 👍 upscale, ‼️ / 🔁 variations, ✉️ info, 👎 / ❌ cancel.

## Parameters

`dreambot params` prints this list from the code, with a one-line explanation of each approximation.

| Parameter | Dreambot | How |
| --- | --- | --- |
| `--ar`, `--aspect` | ✅ | Midjourney's exact pixel sizes for 1:1, 4:3, 3:2, 16:9 and their portraits; the same pixel budget for every other ratio. Up to about 10:1. |
| `--seed`, `--sameseed` | ✅ | Exactly reproducible (Midjourney V8 is "99%"). |
| `--no` | ✅ / 🟡 | A real negative prompt on models with classifier-free guidance; otherwise folded into the prompt, best-effort. |
| `--stop` | ✅ | Decodes the partially denoised image at that point. |
| `--repeat`, `--r` | ✅ | Up to 40. |
| `--raw`, `--style raw` | ✅ | The prompt is used exactly as written: no art direction is added. |
| `--edit` (V8 Edit Model) | ✅ | A local instruction-edit model: FLUX.2 klein, Qwen-Image-Edit, Qwen-Image 2.1, FLUX 3 klein, Kontext. |
| `--stylize`, `--s` | 🟡 | How boldly a local language model art-directs the prompt: 0 leaves it as written, 100 is balanced, 750 theatrical. (Without a language model: rule-based emphasis from 250 up.) |
| `--chaos`, `--c` | 🟡 | How different the images of a grid are: the language model writes each its own shot, then light, palette and setting, and from 75 a different medium. (Without one: picked by rule from fixed lists.) |
| `--weird`, `--w` | 🟡 | The language model adds a strange twist to the scene, scaled by the value. |
| `--exp` | 🟡 | Asks for more intricate detail and more dynamic composition, scaled by the value. |
| `--quality`, `--q` | 🟡 | Scales denoising steps; on step-distilled models, trades resolution instead. (Midjourney dropped `--q` in V8.) |
| `--hd` / `--sd` | 🟡 | Generates at standard size, then upscales every image 2× with SeedVR2. |
| `--draft` | 🟡 | Half resolution with the fastest installed model, still a grid of four. |
| `--iw` | 🟡 | Changes how closely image prompts are followed. |
| `--sref` (images) | 🟡 | Each image becomes a *style fingerprint*: its medium, rendering, palette, light, texture and mood in words (from the local vision model) and its measured colours. The words go into the prompt on any model; a model that takes reference images is also shown a board of shuffled swatches cut from the images. |
| `--sref` (several, `::weight`) | 🟡 | Images, codes and saved styles blend in any mix: `--sref a.png::2 4821 dusty-noir`. Colours mix by weight, the heaviest style sets the medium, and each image gets swatches on the board in proportion to its weight. Midjourney only takes per-reference weights in Discord; here they work everywhere. |
| `--sref` (codes, `random`) | 🟡 | Dreambot's own style space of 5.4 million styles; a code decodes to a fingerprint. **Midjourney's codes do not carry over.** With `--repeat` or permutations each job draws its own `random` style, as in Midjourney. |
| `--sref <name>` | — | Dreambot addition: a style or blend saved under a name. (Midjourney cannot make a style code from your own image.) |
| `--sw` | 🟡 | How much of the fingerprint is applied: a hint below 50; medium, rendering, palette and light from 50; texture and mood too from 200; insisted on from 500. |
| `--sv` | 🟡 | The style version: `1` the original behaviour (a code is a fixed recipe, an image is shown whole to an edit model), `2` words and swatch board (the default), `3` words only, on your chosen model. Midjourney's other version numbers are accepted and use 2. |
| `--cref`, `--cw`, `--oref`, `--ow` | 🟡 | An edit model told to keep the character or subject. |
| `--niji` | 🟡 | Switches to the Anima anime model. |
| `--v`, `--version`, `--preview` | ⚪ | Midjourney model versions mean nothing here; choose a local model with `--model`. |
| `--fast`, `--relax`, `--turbo`, `--stealth`, `--public` | ⚪ | |
| `--p`, `--profile`, `--personalize` | 🟡 | Lookbooks (see below): `--p` for your defaults, `--p <name or code>` for particular ones, several with `::weight`. `--stylize` sets how strongly they apply. Also `--look`. |
| `--tile` | ⏳ | |
| `--video`, `--motion`, `--loop`, `--end`, `--bs` | ⏳ | |
| Multi-prompt weights `a::2 b::1`, negative weights | 🟡 | Parts are ordered by weight; negative parts become things to avoid. (Midjourney removed `::` in V7; Dreambot keeps it.) |
| Permutations `{a, b}` | ✅ | In text and in parameters, nested, with `\,`; at most 40 jobs. |
| Image prompts (links or attachments first) | 🟡 | Edit model, FLUX Redux, or plain image-to-image, whichever is installed. |
| Option sets (`--name`) | ✅ | |
| — | `--model`, `--steps`, `--cfg`, `--strength`, `--lora`, `--neg`, `--n` | Dreambot additions. |

## References and personalization

| Midjourney | Dreambot | Notes |
| --- | --- | --- |
| Image prompts | 🟡 | See above. |
| Style references, weights | 🟡 | Images, codes and saved styles, blended by weight. Checked on Z-Image Turbo, FLUX.2 klein 4B and Qwen-Image 2.1: the style carries over and, with the swatch board, the reference's content does not. Shown whole (`--sv 1`), a reference leaked its content on klein and barely carried its style. |
| Style codes, `--sref random` | 🟡 | Within Dreambot's own style space. |
| Style Explorer | 🟡 | The Styles page of the web app: shuffle, search by words ("anime pastel", "film grain"), like, look at the styles one step away, try a style on your latest prompt. There is no "popular": there is no community. A card shows a style's palette and description, not sample images. |
| Style Creator | ⏳ | Not built. The web app's blend bench is a different way to the same end: put styles, pictures and Lookbooks together by weight, see what the blend comes to, and save it under a name. |
| Character reference, Omni reference | 🟡 | Quality depends on the edit model; larger ones hold identity better. |
| Edit Model references (up to 4) | ✅ | The limit is the model's: 1 for Kontext, 4 for FLUX.2 klein and Qwen 2.1, 10 for FLUX 3 klein. |
| Moodboards | 🟡 | **Lookbooks.** A named set of pictures added by upload, link or from your own images. Every state of one has a code; a code keeps working after the book is edited or deleted. Each image of a grid is led by a different picture of the book, so a varied book gives varied results. Several blend, and here their `::weight`s count (Midjourney ignores weights on moodboards). `--sw` does not apply to them, as in Midjourney; `--sv 3` does (words only). |
| Default profiles, the Personalize button | ✅ | Choose default Lookbooks in the web app or with `/looks default`; `/settings personalize on` (the **P** button on the web) adds them to every prompt. |
| Personalization profiles from ranking images | ⏳ | Not built. Favourites form a `liked` Lookbook, which is the nearest thing. |

## Modes

| Midjourney | Dreambot | Notes |
| --- | --- | --- |
| Remix mode | ✅ | A modal on Discord, a reply prompt on Telegram, `V2: new prompt` anywhere. |
| Variation mode (strong / subtle) | ✅ | |
| RAW mode | ✅ | |
| HD / SD | 🟡 | Via upscaling. |
| Draft mode | 🟡 | |
| Fast / Relax / Turbo, Stealth / Public | ⚪ | |
| Conversational and voice mode | ⏳ | |
| Thinking mode | ⏳ | |

## Editor

| Midjourney | Dreambot | Notes |
| --- | --- | --- |
| Erase and repaint a region | 🟡 | `region …` |
| Move / resize / outpaint | ✅ | Zoom, pan and square cover it. |
| Retexture | ✅ | `retexture 2: the same scene as a watercolour`, `retexture 2 --lora pixel-art`, and **Retexture…** in the web app's lightbox. A ControlNet follows the picture's depth while the layout is settled; the look comes from the prompt and from LoRAs. Verified on Z-Image Turbo's ControlNet; the layout is kept, colours are not. |
| Editing your own uploaded images | ✅ | Attach an image with `--edit`, or `/edit`. |
| Layers, smart select, drawing tools | ⏳ | These need a canvas. The web app has none yet: its lightbox has the same edit, zoom, pan and square actions as the chat buttons, and no way to paint a region. |

## Video

⏳ Not built. Midjourney's video is image-to-video; MLX has working ports (LTX-2, Wan 2.2) that could back an
`animate` action.

## Web app

`dreambot web` opens it. It answers this Mac only.

| Midjourney's website | Dreambot | Notes |
| --- | --- | --- |
| Create: the Imagine bar and the creation feed | ✅ | Enter sends, ⌘Enter sends and keeps the prompt. The feed is grouped by day and draws jobs as they run. One view; no grid/list/thumbnail switch. |
| Images menu: uploads library, drop zones, pin | ✅ | Zones for image prompt, style reference and subject reference; upload, paste a link, or drag a picture from the feed. `--edit` is typed, or is the lightbox's Edit action. |
| Settings menu | ✅ | Aspect ratio, model, raw, stylize, variety, weirdness; also images per prompt, HD, art direction, variation mode and remix. |
| Personalize button with default profiles | ✅ | The **P** button and its Lookbook picker. |
| Hover actions on an image | ✅ | Vary subtle, vary strong, like, trash. |
| Prompt panel: rerun, use, copy | ✅ | No Style Creator button. |
| Lightbox and its creation actions | ✅ | Vary subtle/strong, remix, upscale subtle/creative, zoom, pan, square, edit, rerun, describe; copy prompt, seed or job id; download; like; folders; Lookbooks; what it was made from and what was made from it. |
| Organize | ✅ / ⏳ | See [Organisation and social](#organisation-and-social). |
| Moodboards page | ✅ | The Lookbooks page, which also shows what Dreambot reads in a book. |
| Explore → Styles | 🟡 | The Styles page: Dreambot's own codes. |
| Style Creator | ⏳ | |
| Personalize page (ranking images) | ⏳ | |
| Editor (canvas, layers, smart select) | ⏳ | |
| Explore feed, profiles, Tasks, shared folders | ➖ | Community features. |
| Phone layout | 🟡 | The layout adapts to narrow windows, but it has not been tried on a phone, and the app only answers the Mac it runs on. |
| — | Blend bench, saved styles | Dreambot additions. |

## Organisation and social

| Midjourney | Dreambot |
| --- | --- |
| Job ids, `/show`, seeds | ✅ |
| Results by DM / in the channel | ✅ |
| Image files | ✅ `~/Pictures/Dreambot/<date>/`, each PNG carrying its prompt, seed and model in metadata |
| Web gallery | ✅ The web app's Organize page: every image from every frontend |
| Search, filters, timeline | ✅ Prompt search, and filters for liked, model, kind of job, Lookbook, folder and day |
| Folders, bulk actions, trash | ✅ Folders; select many to like, trash, file or download as a zip. Trash only hides |
| Folder groups, saved searches, creating inside a folder | ⏳ |
| Explore feed, public profiles, ranking, shared folders | ➖ |
| Subscription, GPU hours, queues per plan | ➖ Free; one queue, one job at a time, a per-person cap on waiting jobs |

## Platforms

| | Discord | Telegram | iMessage |
| --- | --- | --- | --- |
| Slash commands | ✅ native | ✅ | ✅ typed |
| Plain-text prompts | DMs and @mentions | private chats | always (with a trigger word in other people's conversations) |
| Buttons | ✅ | ✅ | — typed shortcuts instead |
| Text input for buttons | modal | reply prompt | — |
| Live progress | ✅ message edits | ✅ message edits | — one acknowledgement |
| Reply to a picture | ✅ | ✅ | ✅ |
| Reactions / tapbacks | ✅ | ✅ | ✅ |
| Grid | one image | one image | four separate images |
| Full-quality originals | attached PNG (JPEG above 9.5 MB) | "Original" button sends the file | the PNG itself |

All three have been used on the live services, though not every action on each. See the README's
verification table.
