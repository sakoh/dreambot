// The Organize page: every image ever made, to search, filter, sort into folders, like, trash
// and download — one at a time or many at once.

import { html, useEffect, useRef, useState } from "./vendor/preact-htm.js";
import { api, thumb } from "./api.js";
import { attempt, openLightbox, refreshFolders, store, toast, useStore } from "./store.js";
import { Ask, Empty, Icon, IconButton, Menu, dayKey, dayLabel } from "./ui.js";

const SIZES = { small: 132, medium: 196, large: 290 };
const keyOf = (image) => `${image.job}:${image.index}`;
const itemOf = (key) => ({ job: key.split(":")[0], index: Number(key.split(":")[1]) });

function Rail({ title, children }) {
  return html`<section class="rail-group">
    <h4>${title}</h4>
    ${children}
  </section>`;
}

function Pick({ active, onClick, children, count }) {
  return html`<button type="button" class=${`rail-item ${active ? "is-on" : ""}`} aria-pressed=${active} onClick=${onClick}>
    <span>${children}</span>${count != null && html`<span class="muted">${count}</span>`}
  </button>`;
}

export function OrganizePage() {
  const { folders } = useStore();
  const [filters, setFilters] = useState(() => {
    // Another page may have sent the person here to look at something in particular.
    const start = { q: "", liked: false, trashed: false, model: "", kind: "", folder: "", look: "", day: "", ...(store.organize || {}) };
    store.organize = null;
    return start;
  });
  const [query, setQuery] = useState("");
  const [images, setImages] = useState([]);
  const [next, setNext] = useState(null);
  const [loading, setLoading] = useState(false);
  const [facets, setFacets] = useState(null);
  const [selected, setSelected] = useState(new Set());
  const [size, setSize] = useState(localStorage.getItem("dreambot-grid") || "medium");
  const [asking, setAsking] = useState(null); // { kind: "new" } | { kind: "rename", folder }
  const anchor = useRef(null);
  const sentinel = useRef(null);
  const state = useRef({});
  state.current = { filters, next, loading, images };

  const load = async (more = false) => {
    const now = state.current;
    if (now.loading || (more && !now.next)) return;
    setLoading(true);
    const params = { ...now.filters, liked: now.filters.liked ? 1 : "", trashed: now.filters.trashed ? 1 : "", limit: 90 };
    if (more) params.before = now.next;
    const answer = await attempt(() => api.get("/api/images", params));
    setLoading(false);
    if (!answer) return;
    setImages(more ? [...now.images, ...answer.images] : answer.images);
    setNext(answer.next);
  };
  const loadFacets = async () => {
    const answer = await attempt(() => api.get("/api/facets"));
    if (answer) setFacets(answer);
    refreshFolders();
  };

  useEffect(() => {
    setSelected(new Set());
    setImages([]);
    setNext(null);
    state.current = { ...state.current, images: [], next: null, loading: false };
    load();
  }, [JSON.stringify(filters)]);
  useEffect(() => {
    loadFacets();
  }, []);
  useEffect(() => {
    if (!sentinel.current) return undefined;
    const watcher = new IntersectionObserver((entries) => entries[0].isIntersecting && load(true), { rootMargin: "800px" });
    watcher.observe(sentinel.current);
    return () => watcher.disconnect();
  }, []);

  const set = (changes) => setFilters({ ...filters, ...changes });
  const flip = (name, value) => set({ [name]: filters[name] === value ? "" : value });

  // ---------------------------------------------------------------------- selection
  const order = images.map(keyOf);
  const choose = (image, event) => {
    const key = keyOf(image);
    const picked = new Set(selected);
    if (event.shiftKey && anchor.current && order.includes(anchor.current)) {
      const [from, to] = [order.indexOf(anchor.current), order.indexOf(key)].sort((a, b) => a - b);
      for (const other of order.slice(from, to + 1)) picked.add(other);
    } else if (picked.has(key)) picked.delete(key);
    else picked.add(key);
    anchor.current = key;
    setSelected(picked);
  };
  const click = (image, event) => {
    if (selected.size || event.shiftKey || event.metaKey || event.ctrlKey) return choose(image, event);
    openLightbox(image.job, image.index, images.map((other) => ({ job: other.job, index: other.index })));
  };
  const chosen = [...selected].map(itemOf);

  const bulk = async (action, more = {}, done = null) => {
    const answer = await attempt(() => api.post("/api/images/bulk", { action, items: chosen, ...more }));
    if (!answer) return;
    if (done) toast(done);
    setSelected(new Set());
    state.current = { ...state.current, loading: false };
    await load();
    loadFacets();
  };
  const download = async () => {
    const blob = await attempt(() => api.blob("/api/images/download", { items: chosen }));
    if (!blob) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "dreambot-images.zip";
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 4000);
  };

  // ------------------------------------------------------------------------ folders
  const saveFolder = async (name) => {
    const answer =
      asking.kind === "new"
        ? await attempt(() => api.post("/api/folders", { name }))
        : await attempt(() => api.patch(`/api/folders/${asking.folder.id}`, { name }));
    if (!answer) return false;
    refreshFolders();
    return true;
  };
  const removeFolder = async (folder) => {
    if (!window.confirm(`Delete the folder “${folder.name}”? The images in it stay in your gallery.`)) return;
    if (await attempt(() => api.del(`/api/folders/${folder.id}`))) {
      if (filters.folder === folder.id) set({ folder: "" });
      refreshFolders();
    }
  };

  const days = [];
  for (const image of images) {
    const key = dayKey(image.created);
    if (!days.length || days[days.length - 1].key !== key) days.push({ key, label: dayLabel(image.created), images: [] });
    days[days.length - 1].images.push(image);
  }
  const filtered = Object.entries(filters).some(([, value]) => value);

  return html`<div class="page organize">
    <div class="page-head">
      <h2>Organize</h2>
      <form
        class="search inline-search"
        onSubmit=${(event) => {
          event.preventDefault();
          set({ q: query.trim() });
        }}
      >
        <${Icon} name="search" size=${16} />
        <input placeholder="Search your prompts" value=${query} onInput=${(event) => setQuery(event.target.value)} />
        ${filters.q && html`<${IconButton} icon="x" label="Clear the search" size=${14} onClick=${() => { setQuery(""); set({ q: "" }); }} />`}
      </form>
      <span class="grow"></span>
      ${filtered && html`<button type="button" class="button" onClick=${() => { setQuery(""); setFilters({ q: "", liked: false, trashed: false, model: "", kind: "", folder: "", look: "", day: "" }); }}>Clear filters</button>`}
      <div class="segmented" role="group" aria-label="Thumbnail size">
        ${Object.keys(SIZES).map(
          (name) => html`<button
            type="button"
            class=${size === name ? "is-on" : ""}
            aria-pressed=${size === name}
            onClick=${() => {
              setSize(name);
              localStorage.setItem("dreambot-grid", name);
            }}
          >
            ${name[0].toUpperCase()}
          </button>`,
        )}
      </div>
    </div>

    <div class="organize-body">
      <aside class="rail">
        <${Rail} title="Show">
          <${Pick} active=${filters.liked} onClick=${() => set({ liked: !filters.liked })}><${Icon} name="heart" size=${15} /> Liked<//>
          <${Pick} active=${filters.trashed} count=${facets ? facets.trash : null} onClick=${() => set({ trashed: !filters.trashed })}>
            <${Icon} name="trash" size=${15} /> Trash
          <//>
        <//>
        <${Rail} title="Folders">
          ${folders.map(
            (folder) => html`<div class="rail-folder" key=${folder.id}>
              <${Pick} active=${filters.folder === folder.id} count=${folder.count} onClick=${() => flip("folder", folder.id)}>
                <${Icon} name="folder" size=${15} /> ${folder.name}
              <//>
              <${Menu}
                label=${`Folder ${folder.name}`}
                items=${[
                  { label: "Rename…", icon: "edit", onClick: () => setAsking({ kind: "rename", folder }) },
                  { label: "Delete folder", icon: "trash", danger: true, onClick: () => removeFolder(folder) },
                ]}
              />
            </div>`,
          )}
          <button type="button" class="rail-item is-add" onClick=${() => setAsking({ kind: "new" })}>
            <span><${Icon} name="plus" size=${15} /> New folder</span>
          </button>
        <//>
        ${facets &&
        facets.looks.length > 0 &&
        html`<${Rail} title="Lookbooks">
          ${facets.looks.map(
            ([code, label]) => html`<${Pick} active=${filters.look === code} onClick=${() => flip("look", code)}>
              <${Icon} name="lookbook" size=${15} /> ${label.replace(/^Lookbook /, "")} <code>${code}</code>
            <//>`,
          )}
        <//>`}
        ${facets &&
        facets.models.length > 1 &&
        html`<${Rail} title="Model">
          ${facets.models.map(([model, count]) => html`<${Pick} active=${filters.model === model} count=${count} onClick=${() => flip("model", model)}>${model}<//>`)}
        <//>`}
        ${facets &&
        facets.kinds.length > 1 &&
        html`<${Rail} title="Made by">
          ${facets.kinds.map(([kind, count]) => html`<${Pick} active=${filters.kind === kind} count=${count} onClick=${() => flip("kind", kind)}>${kind}<//>`)}
        <//>`}
        ${facets &&
        facets.days.length > 0 &&
        html`<${Rail} title="Timeline">
          <div class="timeline">
            ${facets.days.map(
              ([day, count]) => html`<${Pick} active=${filters.day === day} count=${count} onClick=${() => flip("day", day)}>
                ${new Date(`${day}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}
              <//>`,
            )}
          </div>
        <//>`}
      </aside>

      <div class="gallery" style=${{ "--tile": `${SIZES[size]}px` }}>
        ${images.length === 0 &&
        !loading &&
        html`<${Empty} title=${filters.trashed ? "The trash is empty" : filtered ? "Nothing matches" : "No images yet"}>
          ${filters.trashed
            ? "Images you move to the trash are kept here until you restore them. Nothing is ever deleted from disk."
            : filtered
              ? "No image fits all of those filters."
              : "Everything you make — here, or from iMessage, Discord and Telegram — is collected on this page."}
        <//>`}
        ${days.map(
          (day) => html`<section key=${day.key}>
            <div class="gallery-day">
              <h3 class="day-title">${day.label}</h3>
              <button
                type="button"
                class="icon-button"
                title="Select everything from this day"
                aria-label="Select everything from this day"
                onClick=${() => setSelected(new Set([...selected, ...day.images.map(keyOf)]))}
              >
                <${Icon} name="plus" size=${15} />
              </button>
            </div>
            <div class="gallery-grid">
              ${day.images.map((image) => {
                const key = keyOf(image);
                const picked = selected.has(key);
                return html`<div
                  key=${key}
                  class=${`gallery-tile ${picked ? "is-picked" : ""} ${selected.size ? "is-selecting" : ""}`}
                  title=${image.prompt}
                  onClick=${(event) => click(image, event)}
                >
                  <img src=${thumb(image.url, size === "large" ? 512 : 256)} alt=${image.prompt} loading="lazy" />
                  <button
                    type="button"
                    class="tick"
                    aria-label=${picked ? "Deselect" : "Select"}
                    aria-pressed=${picked}
                    onClick=${(event) => {
                      event.stopPropagation();
                      choose(image, event);
                    }}
                  >
                    ${picked && html`<${Icon} name="check" size=${14} />`}
                  </button>
                  ${image.favorite && html`<span class="liked"><${Icon} name="heart" size=${13} fill=${true} /></span>`}
                </div>`;
              })}
            </div>
          </section>`,
        )}
        <div ref=${sentinel} class="sentinel">${loading ? "Loading…" : ""}</div>
      </div>
    </div>

    ${selected.size > 0 &&
    html`<div class="bulk" role="toolbar" aria-label="Selected images">
      <strong>${selected.size} selected</strong>
      <button type="button" class="button" onClick=${() => bulk("like", {}, "Liked.")}><${Icon} name="heart" size=${16} /><span>Like</span></button>
      <button type="button" class="button" onClick=${() => bulk("unlike", {}, "Unliked.")}>Unlike</button>
      <${Menu}
        icon="folder"
        text="Folder"
        label="Add to or remove from a folder"
        align="left"
        cls="opens-up"
        items=${() => [
          ...folders.map((folder) => ({ label: `Add to ${folder.name}`, icon: "folder", onClick: () => bulk("folder-add", { folder: folder.id }, `Added to ${folder.name}.`) })),
          ...(filters.folder
            ? ["divider", { label: "Remove from this folder", icon: "x", onClick: () => bulk("folder-remove", { folder: filters.folder }, "Removed from the folder.") }]
            : []),
          ...(folders.length ? [] : [{ label: "New folder…", icon: "plus", onClick: () => setAsking({ kind: "new" }) }]),
        ]}
      />
      <button type="button" class="button" onClick=${download}><${Icon} name="download" size=${16} /><span>Download</span></button>
      ${filters.trashed
        ? html`<button type="button" class="button" onClick=${() => bulk("restore", {}, "Restored.")}><${Icon} name="restore" size=${16} /><span>Restore</span></button>`
        : html`<button type="button" class="button is-danger" onClick=${() => bulk("trash", {}, "Moved to trash.")}><${Icon} name="trash" size=${16} /><span>Trash</span></button>`}
      <span class="grow"></span>
      <button type="button" class="button" onClick=${() => setSelected(new Set())}>Clear</button>
    </div>`}
    ${asking &&
    html`<${Ask}
      title=${asking.kind === "new" ? "New folder" : "Rename folder"}
      label="Name"
      value=${asking.kind === "rename" ? asking.folder.name : ""}
      action=${asking.kind === "new" ? "Create" : "Rename"}
      onClose=${() => setAsking(null)}
      onSubmit=${saveFolder}
    />`}
  </div>`;
}
