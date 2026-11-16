// The Lookbooks page. A Lookbook is a set of pictures that stands for a look — Midjourney's
// moodboard. Make one, fill it by upload, link or from your own images, and use it with --p.

import { html, useEffect, useRef, useState } from "./vendor/preact-htm.js";
import { api, thumb } from "./api.js";
import { ImagePicker } from "./picker.js";
import { addRef, attempt, go, refreshLookbooks, store, toast, useStore } from "./store.js";
import { Ask, Dialog, Empty, Icon, IconButton, Menu, StyleWords, Swatches, copy, pictures } from "./ui.js";

function putInPrompt(book) {
  addRef({ role: "look", look: book.slug, name: book.name, swatches: book.style ? book.style.swatches : [] });
  toast(`${book.name} is in the prompt bar.`);
  go("create");
}

function BookCard({ book, onOpen, onRename, onDelete, onDefault }) {
  const covers = book.images.slice(0, 4);
  return html`<article class="book-card" onClick=${onOpen}>
    <div class=${`book-cover count-${covers.length}`}>
      ${covers.length === 0 && html`<div class="book-empty"><${Icon} name="lookbook" size=${28} /><span>No pictures yet</span></div>`}
      ${covers.map((image) => html`<img src=${thumb(image.url, 512)} alt="" loading="lazy" />`)}
    </div>
    ${book.style && html`<${Swatches} swatches=${book.style.swatches} height=${6} />`}
    <div class="book-info">
      <div class="book-title">
        <strong>${book.name}</strong>
        <span class="muted small">${book.count} ${book.count === 1 ? "picture" : "pictures"}</span>
      </div>
      <div class="row book-actions">
        <${IconButton}
          icon="star"
          label=${book.default ? "One of your defaults: a bare --p uses it" : "Make it a default for a bare --p"}
          active=${book.default}
          fill=${book.default}
          onClick=${onDefault}
        />
        <button
          type="button"
          class="button"
          disabled=${!book.count}
          onClick=${(event) => {
            event.stopPropagation();
            putInPrompt(book);
          }}
        >
          Use
        </button>
        <${Menu}
          label=${`More for ${book.name}`}
          items=${[
            { label: "Rename…", icon: "edit", onClick: onRename },
            ...(book.code ? [{ label: `Copy --p ${book.code}`, icon: "copy", onClick: () => copy(`--p ${book.code}`, "Copied") }] : []),
            "divider",
            { label: "Delete Lookbook", icon: "trash", danger: true, onClick: onDelete },
          ]}
        />
      </div>
    </div>
  </article>`;
}

function BookDialog({ id, onClose, onChanged }) {
  const { app } = useStore();
  const [book, setBook] = useState(null);
  const [link, setLink] = useState("");
  const [picking, setPicking] = useState(false);
  const [busy, setBusy] = useState(false);
  const input = useRef(null);

  const show = (answer) => {
    if (!answer) return;
    setBook(answer.lookbook);
    onChanged();
  };
  const load = () => attempt(() => api.get(`/api/lookbooks/${id}`)).then((answer) => answer && setBook(answer.lookbook));
  useEffect(() => {
    load();
  }, [id]);
  // The vision model reads the pictures when the GPU is free: look again until it has.
  const unread = book ? book.unread : 0;
  useEffect(() => {
    if (!unread || !app || !app.vision) return undefined;
    const timer = setInterval(load, 2500);
    return () => clearInterval(timer);
  }, [unread, app && app.vision]);

  const add = async (body) => {
    setBusy(true);
    show(await attempt(() => api.post(`/api/lookbooks/${id}/images`, body)));
    setBusy(false);
  };
  const addFiles = async (files) => {
    if (!files.length) return;
    setBusy(true);
    const answer = await attempt(() => api.upload(files));
    setBusy(false);
    if (answer) await add({ pictures: answer.uploads.map((upload) => ({ upload: upload.id })) });
  };
  const remove = async (image) => show(await attempt(() => api.del(`/api/lookbooks/${id}/images/${image.id}`)));

  if (!book) return html`<${Dialog} title="Lookbook" onClose=${onClose} wide><p class="muted">Loading…</p><//>`;
  return html`<${Dialog} title=${book.name} onClose=${onClose} wide>
    <div
      class="book-detail"
      onDragOver=${(event) => event.preventDefault()}
      onDrop=${(event) => {
        event.preventDefault();
        const dragged = event.dataTransfer.getData("application/x-dreambot-picture");
        if (dragged) add({ pictures: [JSON.parse(dragged).picture] });
        else addFiles(pictures(event.dataTransfer.files));
      }}
    >
      <div class="book-pictures">
        <div class="row">
          <button type="button" class="button" disabled=${busy} onClick=${() => input.current.click()}>
            <${Icon} name="upload" size=${16} /><span>Upload</span>
          </button>
          <input
            ref=${input}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange=${(event) => {
              addFiles(pictures(event.target.files));
              event.target.value = "";
            }}
          />
          <button type="button" class="button" disabled=${busy} onClick=${() => setPicking(true)}>
            <${Icon} name="organize" size=${16} /><span>From your images</span>
          </button>
          <form
            class="link-form"
            onSubmit=${async (event) => {
              event.preventDefault();
              await add({ url: link.trim() });
              setLink("");
            }}
          >
            <${Icon} name="link" size=${16} />
            <input placeholder="…or paste a link to an image" value=${link} onInput=${(event) => setLink(event.target.value)} />
            <button type="submit" class="button" disabled=${!link.trim() || busy}>Add</button>
          </form>
        </div>
        ${book.images.length === 0 &&
        html`<p class="muted book-hint">
          Add the pictures that show the look you are after — a handful is enough to start. Drop files anywhere here.
        </p>`}
        <div class="gallery-grid" style=${{ "--tile": "140px" }}>
          ${book.images.map(
            (image) => html`<div key=${image.id} class="gallery-tile book-picture" title=${image.style ? image.style.summary : ""}>
              <img src=${thumb(image.url, 512)} alt="" loading="lazy" />
              <${IconButton} icon="x" label="Take it out of the Lookbook" size=${14} onClick=${() => remove(image)} />
              ${image.style && html`<div class="picture-swatch"><${Swatches} swatches=${image.style.swatches} height=${5} /></div>`}
            </div>`,
          )}
        </div>
      </div>
      <aside class="book-side">
        <h4>What Dreambot sees in it</h4>
        ${book.style
          ? html`<${Swatches} swatches=${book.style.swatches} height=${22} />
              <strong>${book.style.name}</strong>
              <${StyleWords} style=${book.style} />`
          : html`<p class="muted small">Nothing yet: it is empty.</p>`}
        ${unread > 0 &&
        html`<p class="small note">
          ${app && app.vision
            ? `Reading ${unread} ${unread === 1 ? "picture" : "pictures"}… this happens when the GPU is free.`
            : "Only colours are read without a vision model. On the Mac, run: dreambot models pull describe"}
        </p>`}
        <p class="small muted">
          Each image of a grid leans on a different picture of the book, so a varied book gives varied results and a
          tight one consistent results. <code>--stylize</code> sets how strongly it is applied.
        </p>
        <h4>Use it</h4>
        <div class="row">
          <button type="button" class="button is-primary" disabled=${!book.count} onClick=${() => putInPrompt(book)}>Use in a prompt</button>
          <button type="button" class="button" disabled=${!book.count} onClick=${() => copy(`--p ${book.slug}`, "Copied")}>
            <code>--p ${book.slug}</code>
          </button>
        </div>
        ${book.code &&
        html`<p class="small muted">
          This version is <code>${book.code}</code>. A code keeps meaning these pictures even after you change or delete
          the Lookbook, so an old prompt can always be run again.${" "}
          <button
            type="button"
            class="linkish"
            onClick=${() => {
              store.organize = { look: book.code };
              onClose();
              go("organize");
            }}
          >
            See what was made with it
          </button>
        </p>`}
      </aside>
    </div>
    ${picking &&
    html`<${ImagePicker}
      action="Add"
      onClose=${() => setPicking(false)}
      onPick=${(images) => images.length && add({ pictures: images.map((image) => ({ job: image.job, index: image.index })) })}
    />`}
  <//>`;
}

export function LookbooksPage() {
  const { lookbooks, app } = useStore();
  const [open, setOpen] = useState(null);
  const [asking, setAsking] = useState(null); // { kind: "new" } | { kind: "rename", book }
  useEffect(() => {
    refreshLookbooks();
  }, []);

  const save = async (name) => {
    if (asking.kind === "new") {
      const answer = await attempt(() => api.post("/api/lookbooks", { name }));
      if (!answer) return false;
      await refreshLookbooks();
      setOpen(answer.lookbook.id);
      return true;
    }
    if (!(await attempt(() => api.patch(`/api/lookbooks/${asking.book.id}`, { name })))) return false;
    await refreshLookbooks();
    return true;
  };
  const remove = async (book) => {
    const note = book.code ? ` Prompts that used it keep working through its code (${book.code}).` : "";
    if (!window.confirm(`Delete the Lookbook “${book.name}”?${note}`)) return;
    if (await attempt(() => api.del(`/api/lookbooks/${book.id}`))) refreshLookbooks();
  };
  const flipDefault = async (book) => {
    const chosen = new Set(app ? app.settings.looks : []);
    chosen.has(book.id) ? chosen.delete(book.id) : chosen.add(book.id);
    if (await attempt(() => api.put("/api/lookbooks/defaults", { lookbooks: [...chosen] }))) {
      if (app) app.settings.looks = [...chosen];
      refreshLookbooks();
    }
  };

  return html`<div class="page lookbooks">
    <div class="page-head">
      <h2>Lookbooks</h2>
      <span class="muted">Sets of pictures that stand for a look. Use one with <code>--p name</code>.</span>
      <span class="grow"></span>
      <button type="button" class="button is-primary" onClick=${() => setAsking({ kind: "new" })}>
        <${Icon} name="plus" size=${16} /><span>New Lookbook</span>
      </button>
    </div>
    ${lookbooks && lookbooks.length === 0 &&
    html`<${Empty} title="No Lookbooks yet">
      Collect a few pictures that share the look you want — your own results, uploads or links — and every prompt you
      add <code>--p</code> to takes after them. Several Lookbooks can be blended, each with a weight.
    <//>`}
    <div class="book-grid">
      ${(lookbooks || []).map(
        (book) => html`<${BookCard}
          key=${book.id}
          book=${book}
          onOpen=${() => setOpen(book.id)}
          onRename=${() => setAsking({ kind: "rename", book })}
          onDelete=${() => remove(book)}
          onDefault=${() => flipDefault(book)}
        />`,
      )}
    </div>
    ${open && html`<${BookDialog} id=${open} onClose=${() => setOpen(null)} onChanged=${refreshLookbooks} />`}
    ${asking &&
    html`<${Ask}
      title=${asking.kind === "new" ? "New Lookbook" : "Rename Lookbook"}
      label="Name"
      placeholder="e.g. Autumn diner"
      value=${asking.kind === "rename" ? asking.book.name : ""}
      action=${asking.kind === "new" ? "Create" : "Rename"}
      onClose=${() => setAsking(null)}
      onSubmit=${save}
    />`}
  </div>`;
}
