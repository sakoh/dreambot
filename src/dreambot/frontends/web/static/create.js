// The Create page: the prompt bar, its reference tray and settings, and the feed of everything
// that has been made, newest first, with jobs in progress drawing themselves as they run.

import { html, useEffect, useRef, useState } from "./vendor/preact-htm.js";
import { api, thumb } from "./api.js";
import {
  act,
  addRef,
  attempt,
  compose,
  imagine,
  loadFeed,
  openLightbox,
  refreshLookbooks,
  refreshUploads,
  removeRef,
  saveSettings,
  setDraft,
  store,
  toast,
  updateRef,
  useStore,
} from "./store.js";
import { Dialog, Empty, Icon, IconButton, Menu, Swatches, copy, dayKey, dayLabel, pictures, splitPrompt } from "./ui.js";

const RATIOS = ["1:1", "4:3", "3:2", "16:9", "2:1", "3:4", "2:3", "9:16"];
const ROLES = {
  image: { title: "Image prompt", hint: "what the picture is of", icon: "image" },
  style: { title: "Style reference", hint: "how it should look", icon: "styles" },
  subject: { title: "Subject reference", hint: "a person or object to keep", icon: "person" },
};

// --------------------------------------------------------------------------- the prompt bar

function RefChip({ item }) {
  const label =
    item.role === "look" ? "Lookbook" : item.picture ? ROLES[item.role].title : item.saved ? "Saved style" : "Style code";
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

function DropZone({ role, onFiles }) {
  const [over, setOver] = useState(false);
  const info = ROLES[role];
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

function ReferenceTray({ onFiles, role, setRole }) {
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
      ${Object.keys(ROLES).map(
        (name) => html`<div class=${`tray-zone ${role === name ? "is-chosen" : ""}`} onClick=${() => setRole(name)}>
          <${DropZone} role=${name} onFiles=${onFiles} />
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
        Click a picture to add it as ${ROLES[role].title.toLowerCase()} (pick a box above to change that), or drag it
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

export function PromptBar() {
  const { draft, app } = useStore();
  const [tray, setTray] = useState(false);
  const [panel, setPanel] = useState(false);
  const [role, setRole] = useState("image");
  const [busy, setBusy] = useState(false);
  const area = useRef(null);

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
    if (!files.length) return;
    const answer = await attempt(() => api.upload(files));
    if (!answer) return;
    await refreshUploads();
    for (const upload of answer.uploads) addRef({ role: as, picture: { upload: upload.id }, url: upload.url, name: upload.name });
  };

  const submit = async (keep = false) => {
    const { prompt, attachments } = compose();
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
  const sticky = [
    draft.params.ar && `--ar ${draft.params.ar}`,
    Number(draft.params.chaos) > 0 && `--chaos ${draft.params.chaos}`,
    Number(draft.params.weird) > 0 && `--weird ${draft.params.weird}`,
  ].filter(Boolean);

  return html`<div
    class="prompt-bar"
    onDragOver=${(event) => {
      if ([...event.dataTransfer.types].some((type) => type === "Files" || type === "application/x-dreambot-picture")) {
        event.preventDefault();
        setTray(true);
      }
    }}
  >
    <div class="prompt-row">
      <${IconButton} icon="image" label="Reference images" active=${tray} onClick=${() => setTray(!tray)} />
      <textarea
        ref=${area}
        rows="1"
        placeholder="What will you imagine?"
        value=${draft.text}
        onInput=${(event) => setDraft({ text: event.target.value })}
        onKeyDown=${onKey}
        onPaste=${onPaste}
        aria-label="Prompt"
      ></textarea>
      ${draft.text && html`<${IconButton} icon="x" label="Clear the prompt" onClick=${() => setDraft({ text: "" })} />`}
      <${Personalize} />
      <${Loras} />
      <${IconButton} icon="sliders" label="Settings" active=${panel} onClick=${() => setPanel(!panel)} />
      <button type="button" class="button is-primary send" disabled=${busy} onClick=${() => submit(false)} title="Enter sends; ⌘Enter sends and keeps the prompt">
        Imagine
      </button>
    </div>
    ${(draft.refs.length > 0 || sticky.length > 0 || loras.length > 0) &&
    html`<div class="ref-row">
      ${draft.refs.map((item) => html`<${RefChip} key=${item.id} item=${item} />`)}
      ${loras.map((lora) => html`<${LoraChip} key=${`lora-${lora.name}`} lora=${lora} chosen=${loras} />`)}
      ${sticky.map(
        (text) => html`<span class="chip is-param" title="Added to every prompt; change it in Settings">${text}</span>`,
      )}
    </div>`}
    ${tray && html`<${ReferenceTray} onFiles=${addFiles} role=${role} setRole=${setRole} />`}
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

function ImageTile({ job, image }) {
  const many = job.images.length > 1;
  return html`<div
    class="tile"
    style=${{ aspectRatio: `${image.width} / ${image.height}` }}
    draggable="true"
    onDragStart=${(event) => dragPicture(event, job, image)}
    onClick=${() => openLightbox(job.id, image.index)}
  >
    <img src=${thumb(image.url, many ? 512 : 1024)} alt=${job.text} loading="lazy" />
    <div class="tile-actions">
      <${IconButton}
        icon="wand"
        label="Vary (subtle)"
        onClick=${() => act(job.id, "vary_subtle", { index: image.index })}
      />
      <${IconButton}
        icon="refresh"
        label="Vary (strong)"
        onClick=${() => act(job.id, "vary_strong", { index: image.index })}
      />
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
  </div>`;
}

function Placeholders({ job, live }) {
  const count = Math.max(1, job.count || 1);
  const ratio = job.width && job.height ? `${job.width} / ${job.height}` : "1 / 1";
  return html`<div class=${`tiles count-${Math.min(count, 4)}`}>
    ${Array.from({ length: Math.min(count, 16) }, (_, slot) =>
      html`<div class="tile is-pending" style=${{ aspectRatio: ratio }}>
        ${live && live.preview && slot === 0 && html`<img src=${live.preview} alt="" />`}
      </div>`,
    )}
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
      ${job.seconds != null && html`<span>${Math.round(job.seconds)}s</span>`}
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
    body = html`<div class="job-media">
      <${Placeholders} job=${job} live=${state} />
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
    body = html`<div class="job-media">
      <div class=${`tiles count-${Math.min(shown.length, 4)}`}>
        ${shown.map((image) => html`<${ImageTile} key=${image.index} job=${job} image=${image} />`)}
      </div>
    </div>`;
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
          <code>a red fox in the snow --ar 16:9 --stylize 250</code>
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
