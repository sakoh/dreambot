// The Create page: the prompt bar, its reference tray and settings, and the feed of everything
// that has been made, pictures and videos, newest first, with jobs in progress drawing themselves
// as they run.

import { html, useEffect, useRef, useState } from "./vendor/preact-htm.js";
import { api, thumb } from "./api.js";
import {
  EXTEND_SECONDS,
  act,
  addRef,
  animateManually,
  attempt,
  compose,
  extendManually,
  imagine,
  loadFeed,
  openLightbox,
  refreshLookbooks,
  refreshUploads,
  removeRef,
  saveSettings,
  setDraft,
  setVideo,
  store,
  toast,
  updateRef,
  useStore,
  videoFrom,
} from "./store.js";
import {
  ClipBadge,
  ClipPreview,
  ContextMenu,
  Dialog,
  Empty,
  Icon,
  IconButton,
  Menu,
  Swatches,
  clipLength,
  copy,
  dayKey,
  dayLabel,
  pictures,
  splitPrompt,
} from "./ui.js";

const RATIOS = ["1:1", "4:3", "3:2", "16:9", "2:1", "3:4", "2:3", "9:16"];
const ROLES = {
  image: { title: "Image prompt", hint: "what the picture is of", icon: "image" },
  style: { title: "Style reference", hint: "how it should look", icon: "styles" },
  subject: { title: "Subject reference", hint: "a person or object to keep", icon: "person" },
};
// What a picture can be to a video. The starting frame is the image prompt under another name.
const VIDEO_ROLES = {
  image: { title: "Starting frame", hint: "the picture the video starts on", icon: "image" },
  end: { title: "Ending frame", hint: "the picture it ends on", icon: "image" },
};
// A video that is being continued already has its start: the picture it can be given is where it ends.
const EXTEND_ROLES = { end: VIDEO_ROLES.end };
/**
 * The roles a picture can be given in the prompt bar as it is now, and the one that is chosen
 * (none, when a picture has no part to play). `ends` says whether the video model can lead a
 * video up to a picture: without that, a video has no ending frame.
 */
const rolesNow = (draft, role, ends) => {
  const filming = ends ? VIDEO_ROLES : { image: VIDEO_ROLES.image };
  const roles = draft.extend ? (ends ? EXTEND_ROLES : {}) : draft.video.on ? filming : ROLES;
  return { roles, role: roles[role] ? role : Object.keys(roles)[0] };
};
const SECONDS = [2, 3, 4, 5];
const MORE_SECONDS = [2, 3, 4];

// --------------------------------------------------------------------------- the prompt bar

function RefChip({ item, filming }) {
  const role = (filming && VIDEO_ROLES[item.role]) || ROLES[item.role] || VIDEO_ROLES[item.role];
  const label = item.role === "look" ? "Lookbook" : item.picture ? role.title : item.saved ? "Saved style" : "Style code";
  return html`<div class=${`ref-chip role-${item.role}`} title=${`${label}: ${item.name || item.code || ""}`}>
    ${item.url
      ? html`<img src=${thumb(item.url, 256)} alt="" />`
      : html`<div class="ref-swatch"><${Swatches} swatches=${item.swatches} height=${34} /></div>`}
    <div class="ref-text">
      <span class="ref-role">${label}</span>
      <span class="ref-name">${item.name || item.code || item.look || "picture"}</span>
    </div>
    ${(item.role === "style" || item.role === "look") &&
    html`<label class="ref-weight" title="Weight: how much this counts next to the others">
      ×<input
        type="number"
        min="0"
        max="10"
        step="0.5"
        value=${item.weight}
        onInput=${(event) => updateRef(item.id, { weight: Math.max(0, Number(event.target.value) || 0) })}
      />
    </label>`}
    <${IconButton} icon="x" label="Remove" size=${14} onClick=${() => removeRef(item.id)} />
  </div>`;
}

function DropZone({ role, info, onFiles }) {
  const [over, setOver] = useState(false);
  const drop = (event) => {
    event.preventDefault();
    setOver(false);
    const dragged = event.dataTransfer.getData("application/x-dreambot-picture");
    if (dragged) {
      const picture = JSON.parse(dragged);
      addRef({ role, picture: picture.picture, url: picture.url, name: picture.name });
    } else {
      onFiles(pictures(event.dataTransfer.files), role);
    }
  };
  return html`<div
    class=${`drop-zone ${over ? "is-over" : ""}`}
    onDragOver=${(event) => {
      event.preventDefault();
      setOver(true);
    }}
    onDragLeave=${() => setOver(false)}
    onDrop=${drop}
  >
    <${Icon} name=${info.icon} />
    <strong>${info.title}</strong>
    <span>${info.hint}</span>
  </div>`;
}

function ReferenceTray({ onFiles, role, roles, setRole }) {
  const { uploads, draft } = useStore();
  const input = useRef(null);
  const [link, setLink] = useState("");
  useEffect(() => {
    refreshUploads();
  }, []);
  const addLink = async (event) => {
    event.preventDefault();
    const answer = await attempt(() => api.post("/api/uploads", { url: link.trim() }));
    if (!answer) return;
    setLink("");
    await refreshUploads();
    for (const upload of answer.uploads) addRef({ role, picture: { upload: upload.id }, url: upload.url, name: upload.name });
  };
  const remove = async (upload) => {
    if (await attempt(() => api.del(`/api/uploads/${upload.id}`))) refreshUploads();
  };
  return html`<div class="tray">
    <div class="tray-zones">
      ${Object.keys(roles).map(
        (name) => html`<div key=${name} class=${`tray-zone ${role === name ? "is-chosen" : ""}`} onClick=${() => setRole(name)}>
          <${DropZone} role=${name} info=${roles[name]} onFiles=${onFiles} />
        </div>`,
      )}
    </div>
    <div class="tray-library">
      <div class="row">
        <button type="button" class="button" onClick=${() => input.current.click()}>
          <${Icon} name="upload" size=${16} /><span>Upload</span>
        </button>
        <input
          ref=${input}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange=${(event) => {
            onFiles(pictures(event.target.files), role);
            event.target.value = "";
          }}
        />
        <form class="link-form" onSubmit=${addLink}>
          <${Icon} name="link" size=${16} />
          <input placeholder="…or paste a link to an image" value=${link} onInput=${(event) => setLink(event.target.value)} />
          <button type="submit" class="button" disabled=${!link.trim()}>Add</button>
        </form>
        <span class="grow"></span>
        <button
          type="button"
          class=${`button ${draft.pinned ? "is-on" : ""}`}
          title="Keep these references in the bar after a prompt is sent"
          onClick=${() => setDraft({ pinned: !draft.pinned })}
        >
          <${Icon} name="pin" size=${16} /><span>${draft.pinned ? "Pinned" : "Pin"}</span>
        </button>
      </div>
      <p class="muted small">
        Click a picture to add it as ${roles[role].title.toLowerCase()} (pick a box above to change that), or drag it
        into a box. Pictures from your feed can be dragged in too.
      </p>
      <div class="library">
        ${uploads.length === 0 && html`<span class="muted small">Nothing uploaded yet.</span>`}
        ${uploads.map(
          (upload) =>
            html`<div
              key=${upload.id}
              class="library-item"
              draggable="true"
              onDragStart=${(event) =>
                event.dataTransfer.setData(
                  "application/x-dreambot-picture",
                  JSON.stringify({ picture: { upload: upload.id }, url: upload.url, name: upload.name }),
                )}
              onClick=${() => addRef({ role, picture: { upload: upload.id }, url: upload.url, name: upload.name })}
            >
              <img src=${thumb(upload.url, 256)} alt=${upload.name} loading="lazy" />
              <${Menu}
                label="More"
                items=${[
                  {
                    label: "Describe",
                    icon: "text",
                    onClick: async () => {
                      const answer = await attempt(() => api.post("/api/describe", { picture: { upload: upload.id } }));
                      if (answer) toast("Writing four prompts for it — they will appear in the feed.");
                    },
                  },
                  { label: "Remove from library", icon: "trash", danger: true, onClick: () => remove(upload) },
                ]}
              />
            </div>`,
        )}
      </div>
    </div>
  </div>`;
}

function SettingsPanel() {
  const { app, draft } = useStore();
  if (!app) return null;
  const settings = app.settings;
  const setParam = (name, value) => setDraft({ params: { ...draft.params, [name]: value } });
  const toggle = (name, label, hint) => html`<label class="toggle" title=${hint}>
    <input type="checkbox" checked=${settings[name]} onChange=${(event) => saveSettings({ [name]: event.target.checked })} />
    <span>${label}</span>
  </label>`;
  return html`<div class="settings">
    <section>
      <h4>Shape</h4>
      <div class="ratios">
        <button type="button" class=${`chip ${!draft.params.ar ? "is-on" : ""}`} onClick=${() => setParam("ar", "")}>auto</button>
        ${RATIOS.map(
          (ratio) =>
            html`<button type="button" class=${`chip ${draft.params.ar === ratio ? "is-on" : ""}`} onClick=${() => setParam("ar", ratio)}>
              ${ratio}
            </button>`,
        )}
      </div>
    </section>
    <section>
      <h4>Model</h4>
      <select value=${settings.model || ""} onChange=${(event) => saveSettings({ model: event.target.value || null })}>
        <option value="">Automatic (${app.model || "none installed"})</option>
        ${app.models.filter((model) => model.generates).map((model) => html`<option value=${model.id}>${model.title}</option>`)}
      </select>
      <label class="field">
        <span>Images per prompt</span>
        <input
          type="number"
          min="1"
          max="16"
          value=${settings.count}
          onChange=${(event) => saveSettings({ count: Number(event.target.value) || 4 })}
        />
      </label>
      <div class="toggles">
        ${toggle("raw", "Raw", "Use prompts exactly as written: no art direction is added")}
        ${toggle("hd", "HD", "Upscale every image to twice the size")}
        ${toggle("enhance", "Art direction", "A language model on this Mac directs each image of a short prompt")}
      </div>
    </section>
    <section>
      <h4>Aesthetics</h4>
      <label class="slider">
        <span>Stylize <b>${settings.stylize}</b></span>
        <input
          type="range"
          min="0"
          max="1000"
          step="10"
          value=${settings.stylize}
          onChange=${(event) => saveSettings({ stylize: Number(event.target.value) })}
        />
      </label>
      <label class="slider">
        <span>Variety <b>${draft.params.chaos}</b></span>
        <input type="range" min="0" max="100" step="5" value=${draft.params.chaos} onInput=${(event) => setParam("chaos", Number(event.target.value))} />
      </label>
      <label class="slider">
        <span>Weirdness <b>${draft.params.weird}</b></span>
        <input type="range" min="0" max="3000" step="50" value=${draft.params.weird} onInput=${(event) => setParam("weird", Number(event.target.value))} />
      </label>
    </section>
    <section>
      <h4>Variations</h4>
      <div class="ratios">
        ${["strong", "subtle"].map(
          (mode) =>
            html`<button type="button" class=${`chip ${settings.variation === mode ? "is-on" : ""}`} onClick=${() => saveSettings({ variation: mode })}>
              ${mode}
            </button>`,
        )}
      </div>
      <div class="toggles">${toggle("remix", "Remix", "Ask for a new prompt before making variations")}</div>
    </section>
    ${app.video.model &&
    html`<section>
      <h4>Video</h4>
      <div class="field">
        <span title="How many videos each prompt makes. Each is a different take, and each takes about as long as a grid of pictures.">
          Videos per prompt
        </span>
        <div class="ratios">
          ${[1, 2, 4].map(
            (count) =>
              html`<button
                type="button"
                class=${`chip ${settings.videos === count ? "is-on" : ""}`}
                aria-pressed=${settings.videos === count}
                onClick=${() => saveSettings({ videos: count })}
              >
                ${count}
              </button>`,
          )}
        </div>
      </div>
      <div class="field">
        <span title="SD is 896×512 for a 16:9 video; HD is 1280×704, in about twice the time.">Resolution</span>
        <div class="ratios">
          ${[
            [false, "SD"],
            [true, "HD"],
          ].map(
            ([hd, label]) =>
              html`<button
                type="button"
                class=${`chip ${settings.video_hd === hd ? "is-on" : ""}`}
                aria-pressed=${settings.video_hd === hd}
                onClick=${() => saveSettings({ video_hd: hd })}
              >
                ${label}
              </button>`,
          )}
        </div>
      </div>
      <div class="field">
        <span title="What a video starts on when you give it words and no picture. Picture first: your words are drawn as a picture, and the video starts on it. Words alone: the video model films the words itself.">
          From words
        </span>
        <div class="ratios">
          ${[
            [true, "Picture first"],
            [false, "Words alone"],
          ].map(
            ([first, label]) =>
              html`<button
                type="button"
                class=${`chip ${settings.picture_first === first ? "is-on" : ""}`}
                aria-pressed=${settings.picture_first === first}
                onClick=${() => saveSettings({ picture_first: first })}
              >
                ${label}
              </button>`,
          )}
        </div>
      </div>
    </section>`}
  </div>`;
}

function Personalize() {
  const { app, lookbooks } = useStore();
  if (!app) return null;
  const on = app.settings.personalize;
  const chosen = new Set(app.settings.looks);
  const books = lookbooks || [];
  const flip = async (id) => {
    const next = chosen.has(id) ? [...chosen].filter((other) => other !== id) : [...chosen, id];
    if (await attempt(() => api.put("/api/lookbooks/defaults", { lookbooks: next }))) {
      app.settings.looks = next;
      refreshLookbooks();
    }
  };
  return html`<div class="personalize">
    <button
      type="button"
      class=${`icon-button p-button ${on ? "is-on" : ""}`}
      title=${on ? "Personalize is on: every prompt uses your default Lookbooks" : "Personalize: use your default Lookbooks on every prompt"}
      aria-pressed=${on}
      onClick=${() => saveSettings({ personalize: !on })}
    >
      P
    </button>
    <${Menu}
      icon="down"
      label="Choose your default Lookbooks"
      cls="p-menu"
      items=${() => [
        ...books.map((book) => ({
          label: `${book.name} · ${book.count}`,
          icon: "lookbook",
          checked: chosen.has(book.id),
          onClick: () => flip(book.id),
        })),
        ...(books.length ? ["divider"] : []),
        { label: "Manage Lookbooks…", icon: "lookbook", onClick: () => (location.hash = "#/lookbooks") },
      ]}
    />
  </div>`;
}

// The LoRAs every prompt is given are kept as `--lora` would name them: "pixel-art", "grain:0.5".
const readLora = (token) => {
  const [name, scale] = token.split(":");
  return { name, scale: scale === undefined ? 1 : Number(scale) };
};
const writeLoras = (chosen) => chosen.map((lora) => (lora.scale === 1 ? lora.name : `${lora.name}:${lora.scale}`));

/** The prompt bar's LoRA button: pick the fine-tuned looks that every prompt is made with. */
function Loras() {
  const { app } = useStore();
  const [help, setHelp] = useState(false);
  if (!app) return null;
  const chosen = app.settings.loras.map(readLora);
  const names = new Set(chosen.map((lora) => lora.name));
  const flip = (entry) => {
    if (names.has(entry.name)) return saveSettings({ loras: writeLoras(chosen.filter((lora) => lora.name !== entry.name)) });
    // LoRAs trained on different models cannot share a picture: choosing one takes the place of the others.
    const family = (name) => (app.loras.find((other) => other.name === name) || {}).family || "";
    const together = chosen.filter((lora) => !entry.family || !family(lora.name) || family(lora.name) === entry.family);
    return saveSettings({ loras: writeLoras([...together, { name: entry.name, scale: 1 }]) });
  };
  const label = chosen.length
    ? `LoRA: ${chosen.map((lora) => lora.name).join(", ")} — on every prompt until you change it`
    : "LoRA: give every prompt one fine-tuned look (pixel art, anime, …)";
  return html`<${Menu}
      icon="down"
      text="LoRA"
      label=${label}
      cls=${`lora-menu ${chosen.length ? "is-on" : ""}`}
      items=${() => [
        ...app.loras.map((entry) => ({
          label: entry.base ? `${entry.title} · ${entry.base}` : entry.title,
          icon: "blend",
          checked: names.has(entry.name),
          onClick: () => flip(entry),
        })),
        ...(chosen.length ? ["divider", { label: "No LoRA", icon: "x", onClick: () => saveSettings({ loras: [] }) }] : []),
        ...(app.loras.length ? ["divider"] : []),
        { label: "Add a LoRA…", icon: "info", onClick: () => setHelp(true) },
      ]}
    />
    ${help &&
    html`<${Dialog} title="Add a LoRA" onClose=${() => setHelp(false)}>
      <p>
        A LoRA is a small fine-tune that gives a model one look: pixel art, anime, one illustrator's line. Thousands
        are published on Hugging Face. Add one in Terminal on this Mac, and it appears in this menu:
      </p>
      <p><code>dreambot loras add owner/name --name pixel-art</code></p>
      <p class="muted small">
        <code>owner/name</code> is its Hugging Face page, or the path of a .safetensors file. In a prompt, a LoRA can
        also be named by hand: <code>--lora pixel-art:0.6</code>.
      </p>
    <//>`}`;
}

/** A chosen LoRA under the prompt: what it is, how strongly it is applied, and a way to drop it. */
function LoraChip({ lora, chosen }) {
  const { app } = useStore();
  const entry = app.loras.find((other) => other.name === lora.name) || { title: lora.name, base: "", trigger: "" };
  const save = (next) => saveSettings({ loras: writeLoras(next) });
  const about = [
    entry.base ? `Runs on ${entry.base}, the model it was trained on.` : "Goes on whichever model the prompt runs on.",
    entry.trigger ? `Every prompt starts with “${entry.trigger}”` : "",
  ];
  return html`<div class="ref-chip role-lora" title=${about.join(" ").trim()}>
    <div class="ref-text">
      <span class="ref-role">LoRA</span>
      <span class="ref-name">${entry.title}</span>
    </div>
    <label class="ref-weight" title="Strength: 1 is as it was trained, less is subtler">
      ×<input
        type="number"
        min="0.1"
        max="2"
        step="0.1"
        value=${lora.scale}
        aria-label=${`Strength of ${entry.title}`}
        onChange=${(event) => {
          const scale = Math.min(2, Math.max(0.1, Math.round((Number(event.target.value) || 1) * 100) / 100));
          save(chosen.map((other) => (other.name === lora.name ? { ...other, scale } : other)));
        }}
      />
    </label>
    <${IconButton} icon="x" label="Stop using this LoRA" size=${14} onClick=${() => save(chosen.filter((other) => other.name !== lora.name))} />
  </div>`;
}

/** The prompt bar's Video button: the prompt makes a video instead of pictures while it is lit. */
function VideoButton() {
  const { draft, app } = useStore();
  const on = draft.video.on;
  const ready = Boolean(app && app.video.model);
  const label = on
    ? "Video is on: this prompt makes a video. Click to go back to pictures."
    : ready
      ? "Video: make a video from the words, or one that starts on an image"
      : "Video: needs a video model (in Terminal: dreambot models pull wan-2.2)";
  return html`<button
    type="button"
    class=${`button video-button ${on ? "is-on" : ""}`}
    title=${label}
    aria-label=${label}
    aria-pressed=${on}
    onClick=${() => setVideo({ on: !on })}
  >
    <${Icon} name="video" size=${15} /><span>Video</span>
  </button>`;
}

/** What kind of video the prompt makes: how much moves, whether it loops, and its length. */
function VideoOptions() {
  const { draft, app } = useStore();
  if (!app) return null;
  const video = draft.video;
  const usual = app.video;
  if (!usual.model) {
    return html`<div class="video-row">
      <span class="note small">
        No video model is installed yet. In Terminal on this Mac: <code>dreambot models pull wan-2.2</code> (about 17 GB).
      </span>
    </div>`;
  }
  const extending = Boolean(draft.extend);
  const starts = draft.refs.some((ref) => ref.role === "image");
  const ends = draft.refs.some((ref) => ref.role === "end");
  const motion = video.motion || usual.motion;
  // With no picture given, one is drawn from the words first, as Midjourney's videos all start
  // on a picture; or the video model is given the words alone.
  const wordsOnly = !extending && !starts && !ends;
  const from = videoFrom(draft);
  // A new video loops back to the picture it starts on; one that is continued, to its first
  // frame, which takes a model that can lead a video up to a picture.
  const canLoop = (starts || (extending && usual.ends) || (wordsOnly && from === "picture")) && !ends;
  const what = extending ? "Continues your video" : starts ? "Starts on your picture" : "Ends on your picture";
  const lengths = extending ? MORE_SECONDS : SECONDS;
  const usually = extending ? EXTEND_SECONDS : usual.seconds;
  return html`<div class="video-row">
    ${wordsOnly
      ? html`<div class="segmented video-from" role="group" aria-label="What the video starts on">
          ${[
            ["picture", "Picture first", "Your words are drawn as a picture first, and the video starts on it"],
            ["words", "Words alone", "The video model films your words itself, with no picture to start on"],
          ].map(
            ([name, text, hint]) => html`<button
              type="button"
              class=${from === name ? "is-on" : ""}
              aria-pressed=${from === name}
              title=${hint}
              onClick=${() => setVideo({ from: name })}
            >
              ${text}
            </button>`,
          )}
        </div>`
      : html`<span class="video-what">${what}</span>`}
    <div class="segmented" role="group" aria-label="How much moves">
      ${[
        ["low", "Low motion", "The camera holds still and things move gently"],
        ["high", "High motion", "The camera and everything in front of it move"],
      ].map(
        ([name, text, hint]) => html`<button
          type="button"
          class=${motion === name ? "is-on" : ""}
          aria-pressed=${motion === name}
          title=${hint}
          onClick=${() => setVideo({ motion: name })}
        >
          ${text}
        </button>`,
      )}
    </div>
    <button
      type="button"
      class=${`chip ${video.loop && canLoop ? "is-on" : ""}`}
      aria-pressed=${video.loop && canLoop}
      disabled=${!canLoop}
      title=${canLoop
        ? extending
          ? "Lead it back to the frame the video started on, so that it plays round and round"
          : "End on the frame it starts on, so that it plays round and round"
        : ends
          ? "A loop ends where it starts, so it has no ending frame of its own"
          : extending
            ? "This video model can't lead a video back to the frame it started on"
            : "A loop needs a picture to start on and come back to: give it one, or have one drawn first"}
      onClick=${() => setVideo({ loop: !video.loop })}
    >
      <${Icon} name="loop" size=${13} /> Loop
    </button>
    <label class="video-length" title=${extending ? "How many seconds are added" : "How long the video is"}>
      <span>${extending ? "Add" : "Length"}</span>
      <select value=${String(video.seconds || "")} onChange=${(event) => setVideo({ seconds: event.target.value })}>
        <option value="">${usually} seconds</option>
        ${lengths.filter((each) => each !== usually).map((each) => html`<option value=${String(each)}>${each} seconds</option>`)}
      </select>
    </label>
  </div>`;
}

/** The video the prompt bar is set to continue, and a way to let go of it. */
function ExtendChip({ extend }) {
  return html`<div class="ref-chip role-end" title="What you send continues this video">
    <img src=${thumb(extend.poster, 256)} alt="" />
    <div class="ref-text">
      <span class="ref-role">Extending</span>
      <span class="ref-name">#${extend.job} · ${clipLength(extend.seconds)}</span>
    </div>
    <${IconButton} icon="x" label="Stop extending this video" size=${14} onClick=${() => setDraft({ extend: null })} />
  </div>`;
}

export function PromptBar() {
  const { draft, app } = useStore();
  const [tray, setTray] = useState(false);
  const [panel, setPanel] = useState(false);
  const [picked, setRole] = useState("image");
  const [busy, setBusy] = useState(false);
  const area = useRef(null);
  const filming = draft.video.on;
  const { roles, role } = rolesNow(draft, picked, Boolean(app && app.video.ends));

  useEffect(() => {
    refreshLookbooks();
  }, []);
  useEffect(() => {
    // Grow with the text, up to a point.
    const node = area.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(220, node.scrollHeight)}px`;
  }, [draft.text]);

  const addFiles = async (files, as = role) => {
    // (With no role to give it, a picture has no part to play: a video that is being continued.)
    if (!files.length || !as) return;
    const answer = await attempt(() => api.upload(files));
    if (!answer) return;
    await refreshUploads();
    for (const upload of answer.uploads) addRef({ role: as, picture: { upload: upload.id }, url: upload.url, name: upload.name });
  };

  const submit = async (keep = false) => {
    const { prompt, attachments, action } = compose();
    if (action) {
      // Animating one of Dreambot's pictures, or continuing a video: what its button would
      // send, with what the bar holds.
      setBusy(true);
      const more = { index: action.index, value: action.motion, prompt: prompt || undefined, attachments };
      const answer = await act(action.job, action.kind, more);
      setBusy(false);
      if (answer && answer.jobs.length && !keep) {
        setDraft({ text: "", refs: draft.pinned && !draft.extend ? draft.refs : [], extend: null });
      }
      return;
    }
    if (!prompt && !attachments.length) return;
    setBusy(true);
    const jobs = await imagine(prompt, attachments);
    setBusy(false);
    if (jobs && !keep) setDraft({ text: "", refs: draft.pinned ? draft.refs : [] });
  };

  const onKey = (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    submit(event.metaKey || event.ctrlKey);
  };
  const onPaste = (event) => {
    const files = pictures(event.clipboardData && event.clipboardData.files);
    if (files.length) {
      event.preventDefault();
      addFiles(files);
    }
  };
  const loras = app ? app.settings.loras.map(readLora) : [];
  const shown = draft.refs;
  // What is added to the prompt without being typed. A video takes the shape of the picture it
  // starts on, and the picture-making parameters do nothing for one, so they are left out.
  const framed = Boolean(draft.extend) || shown.some((ref) => ref.role === "image" || ref.role === "end");
  const sticky = [
    draft.params.ar && !(filming && framed) && `--ar ${draft.params.ar}`,
    !filming && Number(draft.params.chaos) > 0 && `--chaos ${draft.params.chaos}`,
    !filming && Number(draft.params.weird) > 0 && `--weird ${draft.params.weird}`,
  ].filter(Boolean);

  return html`<div
    class="prompt-bar"
    onDragOver=${(event) => {
      if (role && [...event.dataTransfer.types].some((type) => type === "Files" || type === "application/x-dreambot-picture")) {
        event.preventDefault();
        setTray(true);
      }
    }}
  >
    <div class="prompt-row">
      <${IconButton}
        icon="image"
        label=${role ? "Reference images" : "A video that is being continued takes no picture"}
        active=${tray && Boolean(role)}
        disabled=${!role}
        onClick=${() => setTray(!tray)}
      />
      <textarea
        ref=${area}
        rows="1"
        placeholder=${!filming
          ? "What will you imagine?"
          : draft.extend
            ? "What happens next? Leave it empty to let it go on as it was going."
            : draft.refs.some((ref) => ref.role === "image")
              ? "What should happen? Leave it empty to let the picture move as it will."
              : "What will you film?"}
        value=${draft.text}
        onInput=${(event) => setDraft({ text: event.target.value })}
        onKeyDown=${onKey}
        onPaste=${onPaste}
        aria-label="Prompt"
      ></textarea>
      ${draft.text && html`<${IconButton} icon="x" label="Clear the prompt" onClick=${() => setDraft({ text: "" })} />`}
      <${VideoButton} />
      ${!filming && html`<${Personalize} />`}
      ${!filming && html`<${Loras} />`}
      <${IconButton} icon="sliders" label="Settings" active=${panel} onClick=${() => setPanel(!panel)} />
      <button type="button" class="button is-primary send" disabled=${busy} onClick=${() => submit(false)} title="Enter sends; ⌘Enter sends and keeps the prompt">
        ${!filming ? "Imagine" : draft.extend ? "Extend" : "Make video"}
      </button>
    </div>
    ${filming && html`<${VideoOptions} />`}
    ${(shown.length > 0 || sticky.length > 0 || draft.extend || (loras.length > 0 && !filming)) &&
    html`<div class="ref-row">
      ${draft.extend && html`<${ExtendChip} extend=${draft.extend} />`}
      ${shown.map((item) => html`<${RefChip} key=${item.id} item=${item} filming=${filming} />`)}
      ${!filming && loras.map((lora) => html`<${LoraChip} key=${`lora-${lora.name}`} lora=${lora} chosen=${loras} />`)}
      ${sticky.map(
        (text) => html`<span class="chip is-param" title="Added to every prompt; change it in Settings">${text}</span>`,
      )}
    </div>`}
    ${tray && role && html`<${ReferenceTray} onFiles=${addFiles} role=${role} roles=${roles} setRole=${setRole} />`}
    ${panel && html`<${SettingsPanel} />`}
  </div>`;
}

// -------------------------------------------------------------------------------- the feed

const WAITING = {
  queued: (live) => (live.position > 0 ? `Waiting — ${live.position} ahead of it` : "Starting…"),
  started: () => "Starting…",
  loading: () => "Loading the model…",
  thinking: () => "Reading the style and directing the shots…",
  progress: (live) => `${Math.round(live.fraction * 100)}%`,
  preview: (live) => `${Math.round(live.fraction * 100)}%`,
};

function dragPicture(event, job, image) {
  event.dataTransfer.setData(
    "application/x-dreambot-picture",
    JSON.stringify({ picture: { job: job.id, index: image.index }, url: image.url, name: `#${job.id}` }),
  );
}

/** Bring the prompt bar into view after something was put in it. */
const toTheBar = () => window.scrollTo({ top: 0, behavior: "smooth" });

/**
 * One result in the feed. A video is shown by the frame it ends on, plays, without sound, while
 * the pointer is over it, and is held at `scrub` (0 to 1 of its length) while that is a number.
 */
function ImageTile({ job, image, scrub = null }) {
  const many = job.images.length > 1;
  const [over, setOver] = useState(false);
  const [menu, setMenu] = useState(null);
  const clip = Boolean(image.video);
  return html`<div
    class=${`tile ${clip ? "is-video" : ""}`}
    style=${{ aspectRatio: `${image.width} / ${image.height}` }}
    draggable="true"
    onDragStart=${(event) => dragPicture(event, job, image)}
    onClick=${() => openLightbox(job.id, image.index)}
    onMouseEnter=${() => clip && setOver(true)}
    onMouseLeave=${() => clip && setOver(false)}
    onContextMenu=${(event) => {
      if (!clip) return;
      event.preventDefault();
      setMenu({ x: event.clientX, y: event.clientY });
    }}
  >
    <img src=${thumb(clip ? image.poster : image.url, many ? 512 : 1024)} alt=${job.text} loading="lazy" />
    ${clip && (over || scrub != null) && html`<${ClipPreview} src=${image.video} at=${scrub} />`}
    <${ClipBadge} image=${image} />
    <div class="tile-actions">
      ${clip
        ? html`<${IconButton}
              icon="forward"
              label="Extend Auto: make it four seconds longer"
              onClick=${() => act(job.id, "extend", { index: image.index })}
            />
            <${IconButton}
              icon="edit"
              label="Extend Manual: say what happens next, in the prompt bar"
              onClick=${() => {
                extendManually(job, image);
                toTheBar();
              }}
            />`
        : html`<${IconButton}
              icon="wand"
              label="Vary (subtle)"
              onClick=${() => act(job.id, "vary_subtle", { index: image.index })}
            />
            <${IconButton}
              icon="refresh"
              label="Vary (strong)"
              onClick=${() => act(job.id, "vary_strong", { index: image.index })}
            />
            <${IconButton}
              icon="video"
              label="Animate: turn it into a video"
              onClick=${() => act(job.id, "animate", { index: image.index })}
            />`}
      <span class="grow"></span>
      <${IconButton}
        icon="heart"
        label=${image.favorite ? "Unlike" : "Like"}
        active=${image.favorite}
        fill=${image.favorite}
        onClick=${() => act(job.id, "favorite", { index: image.index })}
      />
      <${IconButton}
        icon="trash"
        label="Move to trash"
        onClick=${async () => {
          const body = { action: "trash", items: [{ job: job.id, index: image.index }] };
          if (await attempt(() => api.post("/api/images/bulk", body))) {
            image.trashed = true;
            store.jobs.set(job.id, { ...job });
            toast("Moved to trash. It is under Organize → Trash.");
          }
        }}
      />
    </div>
    ${menu &&
    html`<${ContextMenu}
      at=${menu}
      onClose=${() => setMenu(null)}
      items=${[
        { label: "Download video", icon: "download", href: `${image.video}?download=1` },
        { label: "Download GIF", icon: "download", href: `${image.video}?as=gif` },
      ]}
    />`}
  </div>`;
}

/** A result that is still to come. `preview` is the picture so far, when there is one. */
function PendingTile({ job, preview = null }) {
  const ratio = job.width && job.height ? `${job.width} / ${job.height}` : "1 / 1";
  return html`<div class="tile is-pending" style=${{ aspectRatio: ratio }}>
    ${preview && html`<img src=${preview} alt="" />`}
  </div>`;
}

/**
 * The results of a job side by side: the ones that are made, then a place for each one still
 * to come. Videos are made one after another, so the first can be watched while the last is
 * being made. With Command or Control held, moving over one video holds every video of the job
 * at that point of its length, to compare them moment by moment.
 */
// `shown` are the tiles to draw and `pending` how many more are to come. `made` is how many of
// the job's pictures exist, shown or not: one that was put in the trash still has its place.
function Tiles({ job, shown, live = null, pending = 0, made = shown.length }) {
  const [scrub, setScrub] = useState(null);
  const filmed = shown.some((image) => image.video);
  const move = (event) => {
    if (!(event.metaKey || event.ctrlKey)) return scrub != null && setScrub(null);
    const tile = event.target.closest && event.target.closest(".tile.is-video");
    if (!tile) return undefined;
    const box = tile.getBoundingClientRect();
    return setScrub(Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)));
  };
  const count = Math.min(shown.length + pending, 16);
  return html`<div
    class=${`tiles count-${Math.min(count, 4)}`}
    onMouseMove=${filmed ? move : undefined}
    onMouseLeave=${filmed ? () => setScrub(null) : undefined}
  >
    ${shown.map((image) => html`<${ImageTile} key=${image.index} job=${job} image=${image} scrub=${image.video ? scrub : null} />`)}
    ${Array.from({ length: Math.max(0, count - shown.length) }, (_, slot) => {
      // A video that is still to come is shown by the picture it is going to start on, once
      // that is drawn. Videos are made in order, so this place is the next one's but `slot`.
      const start = live && live.starts ? live.starts[made + slot + 1] : null;
      const preview = start || (slot === 0 && live ? live.preview : null);
      return html`<${PendingTile} key=${`pending-${slot}`} job=${job} preview=${preview} />`;
    })}
  </div>`;
}

function JobSide({ job }) {
  const { text, params } = splitPrompt(job.prompt);
  const reuse = () => {
    setDraft({ text: job.prompt });
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const style = job.style;
  return html`<div class="job-side">
    <p class="job-prompt" title="Click to copy" onClick=${() => copy(job.prompt, "Prompt copied")}>
      ${job.instruction && html`<span class="tag">edit</span>`} ${text || html`<span class="muted">(no text)</span>`}
    </p>
    ${params.length > 0 && html`<div class="chips">${params.map((param) => html`<span class="chip is-param">${param}</span>`)}</div>`}
    ${style &&
    html`<div class="job-style" title=${style.summary}>
      <${Swatches} swatches=${style.swatches} height=${8} />
      <span class="small"
        >${style.sources.map((part) => `${part.label}${style.sources.length > 1 ? ` ${Math.round(part.share * 100)}%` : ""}`).join(" + ")}
        ${" — "}${style.summary.split(" · ").slice(0, 2).join(", ")}</span
      >
    </div>`}
    ${job.notes.map((note) => html`<p class="note small">${note.replace(/`/g, "")}</p>`)}
    <div class="job-meta small muted">
      <span>${job.kind}</span>
      ${job.model && html`<span>${job.model}</span>`}
      ${(job.loras || []).map((lora) => html`<span title="LoRA">${lora.name}${lora.scale === 1 ? "" : ` ×${lora.scale}`}</span>`)}
      ${job.video && job.video.seconds && html`<span title="How long the video is">${clipLength(job.video.seconds)} long</span>`}
      ${job.seconds != null &&
      html`<span title="How long it took to make">${job.video ? "made in " : ""}${Math.round(job.seconds)}s</span>`}
      ${!job.user.startsWith("web:") && html`<span title="Made from a chat app">${job.user.split(":")[0]}</span>`}
      <span>#${job.id}</span>
    </div>
    <div class="job-actions">
      <${IconButton} icon="refresh" label="Run it again" onClick=${() => act(job.id, "reroll")} />
      <${IconButton} icon="edit" label="Put this prompt in the bar" onClick=${reuse} />
      <${IconButton} icon="copy" label="Copy the prompt" onClick=${() => copy(job.prompt, "Prompt copied")} />
    </div>
  </div>`;
}

function JobCard({ job }) {
  const { live } = useStore();
  const state = live.get(job.id);
  const running = job.status === "queued" || job.status === "running";
  const shown = job.images.filter((image) => !image.trashed);
  let body;
  if (running) {
    const say = WAITING[(state && state.type) || job.status] || (() => "Working…");
    const wanted = Math.max(1, job.count || 1);
    // A job's videos are shown one by one, as each is made. Its pictures are shown when it is
    // done: until then they are not kept where they are served from (so a tile for one would be
    // a broken picture), and a finishing pass may still draw them again.
    const made = job.images.filter((image) => image.video);
    const watchable = made.filter((image) => !image.trashed);
    body = html`<div class="job-media">
      <${Tiles} job=${job} shown=${watchable} live=${state} made=${made.length} pending=${Math.max(0, wanted - made.length)} />
      <div class="progress">
        <div class="progress-bar"><span style=${{ width: `${Math.round(((state && state.fraction) || 0) * 100)}%` }}></span></div>
        <span class="small">${say(state || { position: 0, fraction: 0 })}</span>
        <${IconButton} icon="stop" label="Cancel" size=${16} onClick=${() => act(job.id, "cancel")} />
      </div>
    </div>`;
  } else if (job.status === "failed" || job.status === "cancelled") {
    body = html`<div class="job-media">
      <div class=${`failure is-${job.status}`}>
        <strong>${job.status === "cancelled" ? "Cancelled" : "This one did not work"}</strong>
        ${job.error && html`<p>${job.error.replace(/`/g, "")}</p>`}
      </div>
    </div>`;
  } else if (job.texts.length) {
    body = html`<div class="job-media">
      <ol class="texts">
        ${job.texts.map(
          (line) =>
            html`<li>
              <span>${line}</span>
              <button type="button" class="button" onClick=${() => { setDraft({ text: line }); window.scrollTo({ top: 0, behavior: "smooth" }); }}>Use</button>
            </li>`,
        )}
      </ol>
    </div>`;
  } else if (!shown.length) {
    return null; // every image of it is in the trash
  } else {
    body = html`<div class="job-media"><${Tiles} job=${job} shown=${shown} /></div>`;
  }
  return html`<article class=${`job is-${job.status}`}>${body}<${JobSide} job=${job} /></article>`;
}

export function CreatePage() {
  const { order, jobs, next, loading, search } = useStore();
  const sentinel = useRef(null);
  const [query, setQuery] = useState(search);

  useEffect(() => {
    if (!sentinel.current) return undefined;
    const watcher = new IntersectionObserver((entries) => entries[0].isIntersecting && loadFeed({ more: true }), {
      rootMargin: "600px",
    });
    watcher.observe(sentinel.current);
    return () => watcher.disconnect();
  }, []);

  const runSearch = (event) => {
    event.preventDefault();
    store.search = query.trim();
    loadFeed();
  };

  const days = [];
  for (const id of order) {
    const job = jobs.get(id);
    if (!job) continue;
    const key = dayKey(job.created);
    if (!days.length || days[days.length - 1].key !== key) days.push({ key, label: dayLabel(job.created), jobs: [] });
    days[days.length - 1].jobs.push(job);
  }

  return html`<div class="page create">
    <div class="create-top">
      <${PromptBar} />
      <form class="search" onSubmit=${runSearch}>
        <${Icon} name="search" size=${16} />
        <input placeholder="Search your prompts" value=${query} onInput=${(event) => setQuery(event.target.value)} />
        ${search && html`<${IconButton} icon="x" label="Clear the search" size=${14} onClick=${() => { setQuery(""); store.search = ""; loadFeed(); }} />`}
      </form>
    </div>
    ${days.length === 0 &&
    !loading &&
    (search
      ? html`<${Empty} title="Nothing matches">No prompt has all of those words in it.<//>`
      : html`<${Empty} title="Nothing here yet">
          Type what you want to see and press Enter. Add Midjourney-style parameters at the end if you like:
          <code>a red fox in the snow --ar 16:9 --stylize 250</code>. The Video button makes it move.
        <//>`)}
    ${days.map(
      (day) => html`<section key=${day.key} class="day">
        <h3 class="day-title">${day.label}</h3>
        ${day.jobs.map((job) => html`<${JobCard} key=${job.id} job=${job} />`)}
      </section>`,
    )}
    <div ref=${sentinel} class="sentinel">${loading ? "Loading…" : next ? "" : days.length ? "That is everything." : ""}</div>
  </div>`;
}
