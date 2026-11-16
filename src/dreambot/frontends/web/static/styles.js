// The Styles page: style codes to explore, like and try, the styles you have saved, and a bench
// for blending styles — codes, pictures, Lookbooks — by weight, and seeing what the blend comes
// to before anything is rendered.

import { html, useEffect, useRef, useState } from "./vendor/preact-htm.js";
import { api, thumb } from "./api.js";
import { ImagePicker } from "./picker.js";
import { addRef, attempt, emit, go, imagine, refreshLookbooks, store, toast, useStore } from "./store.js";
import { Ask, Dialog, Empty, Icon, IconButton, Menu, StyleWords, Swatches, copy, pictures } from "./ui.js";

/** The prompt to try a style on: what is in the bar, or else the last thing imagined. */
function latestPrompt() {
  if (store.draft.text.trim()) return store.draft.text.trim().replace(/\s--sref\b.*?(?=\s--|$)/, "");
  for (const id of store.order) {
    const job = store.jobs.get(id);
    if (job && job.kind === "imagine" && job.text && !job.instruction) return job.text;
  }
  return "";
}

async function tryStyle(sref, attachments = []) {
  const prompt = latestPrompt();
  if (!prompt) return toast("Type a prompt in Create first: that is what the style is tried on.", "error");
  const jobs = await imagine(`${prompt} --sref ${sref}`, attachments);
  if (jobs) {
    toast(`Trying it on “${prompt.length > 48 ? `${prompt.slice(0, 48)}…` : prompt}”.`);
    go("create");
  }
  return jobs;
}

function putStyleInPrompt(style) {
  addRef(
    style.kind === "saved"
      ? { role: "style", saved: style.use, name: style.name, swatches: style.swatches }
      : { role: "style", code: style.code, name: style.name, swatches: style.swatches },
  );
  toast(`${style.name} is in the prompt bar.`);
}

let benchId = 0;

export function toBench(part) {
  const same = (other) => ["code", "upload", "job", "index", "lookbook"].every((field) => other[field] === part[field]);
  if (!store.bench.some(same)) store.bench = [...store.bench, { weight: 1, ...part, id: ++benchId }];
  emit();
}

function benchStyle(style) {
  toBench({ code: style.code, label: style.name, swatches: style.swatches });
  toast(`${style.name} is on the blend bench.`);
}

// ------------------------------------------------------------------------------- cards

function StyleCard({ style, onLike, onSimilar, onDelete }) {
  const words = style.words || {};
  return html`<article class="style-card">
    <div class="style-face" onClick=${onSimilar} title="Look closer">
      <${Swatches} swatches=${style.swatches} height=${64} />
      ${style.pictures && style.pictures.length > 0 && html`<div class="style-pictures">${style.pictures.slice(0, 3).map((url) => html`<img src=${thumb(url, 256)} alt="" />`)}</div>`}
    </div>
    <div class="style-body">
      <div class="style-name">
        <strong>${style.name}</strong>
        <button type="button" class="linkish small" title="Copy" onClick=${() => copy(`--sref ${style.use}`, "Copied")}>
          <code>${style.kind === "saved" ? style.use : style.code}</code>
        </button>
      </div>
      <p class="small muted style-line">${[words.medium, words.rendering, words.light].filter(Boolean).join(" · ") || "colours only so far"}</p>
      <div class="row style-actions">
        <button type="button" class="button" title="Run your latest prompt in this style" onClick=${() => tryStyle(style.use)}>Try</button>
        <button type="button" class="button" title="Add it to the prompt bar" onClick=${() => putStyleInPrompt(style)}>Use</button>
        <${IconButton} icon="blend" label="Put it on the blend bench" onClick=${() => benchStyle(style)} />
        <span class="grow"></span>
        ${onLike && html`<${IconButton} icon="heart" label=${style.liked ? "Unlike" : "Like"} active=${style.liked} fill=${style.liked} onClick=${onLike} />`}
        ${onDelete && html`<${IconButton} icon="trash" label="Delete this style" onClick=${onDelete} />`}
      </div>
    </div>
  </article>`;
}

function StyleDialog({ code, onClose, onLike }) {
  const [detail, setDetail] = useState(null);
  const [current, setCurrent] = useState(code);
  useEffect(() => {
    setDetail(null);
    attempt(() => api.get(`/api/styles/${current}`)).then((answer) => answer && setDetail(answer));
  }, [current]);
  if (!detail) return html`<${Dialog} title="Style" onClose=${onClose} wide><p class="muted">Loading…</p><//>`;
  const style = detail.style;
  const like = async () => {
    await onLike(style);
    setDetail({ ...detail, style: { ...style, liked: !style.liked } });
  };
  return html`<${Dialog} title=${style.name} onClose=${onClose} wide>
    <div class="style-detail">
      <div>
        <${Swatches} swatches=${style.swatches} height=${90} />
        <div class="row style-detail-actions">
          <button type="button" class="button is-primary" onClick=${() => { tryStyle(style.use); onClose(); }}>Try on your latest prompt</button>
          <button type="button" class="button" onClick=${() => { putStyleInPrompt(style); onClose(); go("create"); }}>Use in a prompt</button>
          <button type="button" class="button" onClick=${() => benchStyle(style)}><${Icon} name="blend" size=${16} /><span>Blend</span></button>
          <${IconButton} icon="heart" label=${style.liked ? "Unlike" : "Like"} active=${style.liked} fill=${style.liked} onClick=${like} />
        </div>
        <button type="button" class="linkish" onClick=${() => copy(`--sref ${style.use}`, "Copied")}><code>--sref ${style.use}</code></button>
      </div>
      <${StyleWords} style=${style} />
    </div>
    ${detail.similar.length > 0 &&
    html`<h4 class="similar-title">One step away</h4>
      <div class="style-grid is-small">
        ${detail.similar.map(
          (other) => html`<article class="style-card" key=${other.code} onClick=${() => setCurrent(other.code)}>
            <div class="style-face"><${Swatches} swatches=${other.swatches} height=${40} /></div>
            <div class="style-body">
              <strong class="small">${other.name}</strong>
              <p class="small muted style-line">${[other.words.medium, other.words.light, other.words.mood].join(" · ")}</p>
            </div>
          </article>`,
        )}
      </div>`}
  <//>`;
}

// ----------------------------------------------------------------------------- explore

function Explore({ onOpen, like }) {
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [seed, setSeed] = useState(() => String(Date.now() % 100000));
  const [pages, setPages] = useState(1);
  const [styles, setStyles] = useState(null);

  useEffect(() => {
    let stale = false;
    (async () => {
      const found = [];
      for (let page = 0; page < pages; page += 1) {
        const answer = await attempt(() => api.get("/api/styles", { q: search, seed: `${seed}-${page}`, count: 24 }));
        if (!answer) return;
        found.push(...answer.styles);
      }
      const seen = new Set();
      if (!stale) setStyles(found.filter((style) => !seen.has(style.code) && seen.add(style.code)));
    })();
    return () => (stale = true);
  }, [search, seed, pages]);

  const flip = async (style) => {
    await like(style);
    setStyles(styles.map((other) => (other.code === style.code ? { ...other, liked: !other.liked } : other)));
  };
  return html`<div>
    <div class="row explore-head">
      <form
        class="search inline-search"
        onSubmit=${(event) => {
          event.preventDefault();
          setPages(1);
          setSearch(query.trim());
        }}
      >
        <${Icon} name="search" size=${16} />
        <input placeholder="Search styles: anime, film, linocut, pastel, moody…" value=${query} onInput=${(event) => setQuery(event.target.value)} />
        ${search && html`<${IconButton} icon="x" label="Clear the search" size=${14} onClick=${() => { setQuery(""); setSearch(""); setPages(1); }} />`}
      </form>
      <button type="button" class="button" onClick=${() => { setPages(1); setSeed(String(Math.floor(Math.random() * 1e6))); }}>
        <${Icon} name="refresh" size=${16} /><span>Shuffle</span>
      </button>
      <span class="muted small">Each code is a style of Dreambot's own: its medium, rendering, palette, light, texture and mood.</span>
    </div>
    ${styles && styles.length === 0 &&
    html`<${Empty} title="No style matches">
      Styles are made of a medium, a way of rendering, a palette, a light, a texture and a mood. Search for words
      from those: <code>watercolor</code>, <code>neon</code>, <code>film grain</code>, <code>cozy</code>.
    <//>`}
    <div class="style-grid">
      ${(styles || []).map(
        (style) => html`<${StyleCard} key=${style.code} style=${style} onLike=${() => flip(style)} onSimilar=${() => onOpen(style.code)} />`,
      )}
    </div>
    ${styles && styles.length >= 24 && html`<div class="more"><button type="button" class="button" onClick=${() => setPages(pages + 1)}>More</button></div>`}
  </div>`;
}

function Mine({ tab, onOpen, like }) {
  const [mine, setMine] = useState(null);
  const load = () => attempt(() => api.get("/api/styles/mine")).then((answer) => answer && setMine(answer));
  useEffect(() => {
    load();
  }, [tab]);
  if (!mine) return null;
  const list = tab === "liked" ? mine.liked : mine.saved;
  const remove = async (style) => {
    if (!window.confirm(`Delete the style “${style.name}”? Prompts that name it will stop working.`)) return;
    if (await attempt(() => api.del(`/api/styles/${style.code}`))) load();
  };
  if (!list.length) {
    return tab === "liked"
      ? html`<${Empty} title="No liked styles yet">Press the heart on a style in Explore to keep it here.<//>`
      : html`<${Empty} title="No saved styles yet">
          Blend styles on the bench — codes, pictures of your own, Lookbooks — and save the result under a name. It is
          then used like any style: <code>--sref your-name</code>.
        <//>`;
  }
  return html`<div class="style-grid">
    ${list.map(
      (style) => html`<${StyleCard}
        key=${style.code}
        style=${style}
        onLike=${tab === "liked" ? async () => { await like(style); load(); } : null}
        onDelete=${tab === "mine" ? () => remove(style) : null}
        onSimilar=${() => onOpen(style.code)}
      />`,
    )}
  </div>`;
}

// ------------------------------------------------------------------------------- bench

const partOf = (item) =>
  item.code
    ? { code: item.code, weight: item.weight }
    : item.lookbook
      ? { lookbook: item.lookbook, weight: item.weight }
      : item.upload
        ? { upload: item.upload, weight: item.weight }
        : { job: item.job, index: item.index, weight: item.weight };

function Bench() {
  const { bench, lookbooks } = useStore();
  const [result, setResult] = useState(null);
  const [code, setCode] = useState("");
  const [picking, setPicking] = useState(false);
  const [saving, setSaving] = useState(false);
  const input = useRef(null);
  const parts = bench.map(partOf);
  const signature = JSON.stringify(parts);

  const refresh = async () => {
    if (!bench.length) return setResult(null);
    const answer = await attempt(() => api.post("/api/styles/blend", { parts }));
    if (answer) setResult(answer);
  };
  useEffect(() => {
    const timer = setTimeout(refresh, 180);
    return () => clearTimeout(timer);
  }, [signature]);
  // A picture's medium, light and mood arrive once the vision model has read it.
  const waiting = result && result.parts.some((part) => part.waiting || (part.style && !part.style.read));
  useEffect(() => {
    if (!waiting) return undefined;
    const timer = setInterval(refresh, 2500);
    return () => clearInterval(timer);
  }, [waiting, signature]);
  useEffect(() => {
    if (!lookbooks) refreshLookbooks();
  }, []);

  const change = (id, changes) => {
    store.bench = store.bench.map((item) => (item.id === id ? { ...item, ...changes } : item));
    emit();
  };
  const remove = (id) => {
    store.bench = store.bench.filter((item) => item.id !== id);
    emit();
  };
  const addCode = async (event) => {
    event.preventDefault();
    const wanted = code.trim();
    const answer = await attempt(() => api.get(`/api/styles/${encodeURIComponent(wanted)}`));
    if (!answer) return;
    toBench({ code: answer.style.code, label: answer.style.name, swatches: answer.style.swatches });
    setCode("");
  };
  const addFiles = async (files) => {
    if (!files.length) return;
    const answer = await attempt(() => api.upload(files));
    if (answer) for (const upload of answer.uploads) toBench({ upload: upload.id, url: upload.url, label: upload.name || "picture" });
  };
  const total = bench.reduce((sum, item) => sum + Number(item.weight), 0) || 1;

  return html`<div
    class="bench"
    onDragOver=${(event) => event.preventDefault()}
    onDrop=${(event) => {
      event.preventDefault();
      const dragged = event.dataTransfer.getData("application/x-dreambot-picture");
      if (dragged) {
        const picture = JSON.parse(dragged);
        toBench({ ...picture.picture, url: picture.url, label: picture.name || "picture" });
      } else addFiles(pictures(event.dataTransfer.files));
    }}
  >
    <section class="bench-parts">
      <h4>Ingredients</h4>
      ${bench.length === 0 &&
      html`<p class="muted">
        Put two or more styles here and set how much each one counts. A style can be a code, a picture of yours, a saved
        style or a whole Lookbook. Nothing is rendered until you try the blend.
      </p>`}
      ${bench.map((item, position) => {
        const part = result && result.parts[position];
        return html`<div class="bench-part" key=${item.id}>
          ${item.url
            ? html`<img src=${thumb(item.url, 256)} alt="" />`
            : html`<div class="bench-swatch"><${Swatches} swatches=${(part && part.style && part.style.swatches) || item.swatches || []} height=${52} /></div>`}
          <div class="bench-text">
            <strong>${item.label}</strong>
            <span class="small muted">
              ${part && part.style
                ? part.style.read
                  ? [part.style.words.medium, part.style.words.rendering].filter(Boolean).join(" · ")
                  : "reading it… colours only so far"
                : ""}
            </span>
            <label class="slider">
              <input type="range" min="0" max="5" step="0.25" value=${item.weight} onInput=${(event) => change(item.id, { weight: Number(event.target.value) })} />
            </label>
          </div>
          <div class="bench-share">
            <b>${Math.round((Number(item.weight) / total) * 100)}%</b>
            <span class="small muted">×${item.weight}</span>
          </div>
          <${IconButton} icon="x" label="Take it off the bench" onClick=${() => remove(item.id)} />
        </div>`;
      })}
      <div class="bench-add">
        <form class="row" onSubmit=${addCode}>
          <input placeholder="A style code or saved style name" value=${code} onInput=${(event) => setCode(event.target.value)} />
          <button type="submit" class="button" disabled=${!code.trim()}>Add</button>
        </form>
        <div class="row">
          <button type="button" class="button" onClick=${() => input.current.click()}><${Icon} name="upload" size=${16} /><span>Upload a picture</span></button>
          <input ref=${input} type="file" accept="image/*" multiple hidden onChange=${(event) => { addFiles(pictures(event.target.files)); event.target.value = ""; }} />
          <button type="button" class="button" onClick=${() => setPicking(true)}><${Icon} name="organize" size=${16} /><span>From your images</span></button>
          <${Menu}
            icon="lookbook"
            text="A Lookbook"
            label="Add a Lookbook"
            align="left"
            items=${() =>
              (lookbooks || [])
                .filter((book) => book.count)
                .map((book) => ({
                  label: `${book.name} · ${book.count}`,
                  icon: "lookbook",
                  onClick: () => toBench({ lookbook: book.id, label: book.name, swatches: book.style ? book.style.swatches : [] }),
                }))}
          />
        </div>
        <p class="small muted with-icon">Or press <${Icon} name="blend" size=${13} /> on any style card, or drop pictures here.</p>
      </div>
    </section>

    <section class="bench-result">
      <h4>The blend</h4>
      ${!result || !result.style
        ? html`<p class="muted">It appears here as soon as there is something to blend.</p>`
        : html`<${Swatches} swatches=${result.style.swatches} height=${72} />
            <strong class="blend-name">${result.style.name}</strong>
            <${StyleWords} style=${result.style} />
            ${result.phrases.length > 0 &&
            html`<div class="blend-phrases">
              <h4>What it adds to a prompt</h4>
              <p class="small">${result.phrases.join(", ")}</p>
              <p class="small muted">
                A model that takes reference images is also shown a board of swatches cut from the pictures, shared out
                by the same weights.
              </p>
            </div>`}
            ${waiting && html`<p class="small note">Still reading a picture: the words will sharpen in a moment.</p>`}
            <div class="row">
              ${result.direct
                ? html`<button type="button" class="button is-primary" onClick=${() => tryStyle(result.sref, result.attachments)}>Try on your latest prompt</button>`
                : html`<span class="small muted">A blend with a Lookbook in it is used by saving it.</span>`}
              <button type="button" class="button" onClick=${() => setSaving(true)}>Save as a style…</button>
            </div>
            ${result.direct &&
            result.sref &&
            !result.attachments.length &&
            html`<button type="button" class="linkish" onClick=${() => copy(`--sref ${result.sref}`, "Copied")}><code>--sref ${result.sref}</code></button>`}`}
    </section>
    ${picking &&
    html`<${ImagePicker}
      action="Add"
      onClose=${() => setPicking(false)}
      onPick=${(images) => images.forEach((image) => toBench({ job: image.job, index: image.index, url: image.url, label: `#${image.job}` }))}
    />`}
    ${saving &&
    html`<${Ask}
      title="Save this blend as a style"
      label="Name (starts with a letter)"
      placeholder="e.g. dusty noir"
      action="Save"
      onClose=${() => setSaving(false)}
      onSubmit=${async (name) => {
        const answer = await attempt(() => api.post("/api/styles", { name, parts }));
        if (!answer) return false;
        toast(`Saved. Use it with --sref ${answer.style.use}`);
        return true;
      }}
    />`}
  </div>`;
}

// -------------------------------------------------------------------------------- page

const TABS = { explore: "Explore", liked: "Liked", mine: "Mine", blend: "Blend" };

export function StylesPage() {
  const { bench } = useStore();
  const [tab, setTab] = useState(() => localStorage.getItem("dreambot-styles-tab") || "explore");
  const [open, setOpen] = useState(null);
  const choose = (name) => {
    setTab(name);
    localStorage.setItem("dreambot-styles-tab", name);
  };
  const like = (style) => attempt(() => api.post("/api/styles/like", { code: style.code, liked: !style.liked }));
  return html`<div class="page styles">
    <div class="page-head">
      <h2>Styles</h2>
      <div class="segmented" role="tablist">
        ${Object.entries(TABS).map(
          ([name, title]) => html`<button type="button" role="tab" class=${tab === name ? "is-on" : ""} aria-selected=${tab === name} onClick=${() => choose(name)}>
            ${title}${name === "blend" && bench.length ? ` · ${bench.length}` : ""}
          </button>`,
        )}
      </div>
    </div>
    ${tab === "explore" && html`<${Explore} onOpen=${setOpen} like=${like} />`}
    ${(tab === "liked" || tab === "mine") && html`<${Mine} tab=${tab} onOpen=${setOpen} like=${like} />`}
    ${tab === "blend" && html`<${Bench} />`}
    ${open && html`<${StyleDialog} code=${open} onClose=${() => setOpen(null)} onLike=${like} />`}
  </div>`;
}
