// Small pieces every page uses: icons, a palette bar, a style card, menus, dialogs, toasts.

import { html, useEffect, useRef, useState } from "./vendor/preact-htm.js";
import { toast, useStore } from "./store.js";

// ------------------------------------------------------------------------------- icons

const PATHS = {
  create: "M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.4z M19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z",
  organize: "M4 4h7v7H4z M13 4h7v7h-7z M4 13h7v7H4z M13 13h7v7h-7z",
  lookbook: "M5 4h11a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3z M5 17a3 3 0 0 1 3-3h11 M9 8h6",
  styles: "M12 3a9 9 0 1 0 0 18c1.1 0 1.7-.9 1.3-1.9-.5-1.3.3-2.6 1.7-2.6H17a4 4 0 0 0 4-4c0-5-4-9.5-9-9.5z M7.5 12a1 1 0 1 0 0-.01 M10 7.5a1 1 0 1 0 0-.01 M15 7.5a1 1 0 1 0 0-.01",
  heart: "M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z",
  trash: "M5 7h14 M10 7V4h4v3 M7 7l1 13h8l1-13 M10 11v6 M14 11v6",
  restore: "M4 12a8 8 0 1 0 2.5-5.8 M4 4v5h5",
  download: "M12 4v11 M7 11l5 5 5-5 M5 20h14",
  copy: "M9 9h10v11H9z M5 15V4h10",
  plus: "M12 5v14 M5 12h14",
  x: "M6 6l12 12 M18 6L6 18",
  image: "M4 5h16v14H4z M4 16l5-5 4 4 3-3 4 4 M9 9.5a1 1 0 1 0 0-.01",
  sliders: "M4 7h10 M18 7h2 M4 17h2 M10 17h10 M16 5v4 M8 15v4",
  search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14z M16.5 16.5L21 21",
  folder: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
  star: "M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z",
  wand: "M5 19L16 8 M14 6l4 4 M18 3v3 M21 6h-3 M6 4v2 M7 5H5 M19 15v2 M20 16h-2",
  left: "M15 5l-7 7 7 7",
  right: "M9 5l7 7-7 7",
  up: "M5 15l7-7 7 7",
  down: "M5 9l7 7 7-7",
  refresh: "M20 12a8 8 0 1 1-2.5-5.8 M20 4v5h-5",
  expand: "M4 10V4h6 M20 14v6h-6 M4 4l7 7 M20 20l-7-7",
  check: "M5 12.5l4.5 4.5L19 7",
  dots: "M5 12a1 1 0 1 0 0-.01 M12 12a1 1 0 1 0 0-.01 M19 12a1 1 0 1 0 0-.01",
  upload: "M12 20V9 M7 13l5-5 5 5 M5 4h14",
  link: "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1 M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
  pin: "M9 4h6l-1 6 3 3H7l3-3z M12 13v7",
  zoom: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14z M16.5 16.5L21 21 M8 11h6",
  square: "M5 5h14v14H5z",
  edit: "M4 20h4L19 9l-4-4L4 16z M13 7l4 4",
  blend: "M9 5a6 6 0 1 0 0 12A6 6 0 0 0 9 5z M15 7a6 6 0 1 0 0 12 6 6 0 0 0 0-12z",
  person: "M12 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8z M4 21a8 8 0 0 1 16 0",
  info: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z M12 11v6 M12 7.5v.5",
  text: "M5 6h14 M5 12h14 M5 18h9",
  stop: "M7 7h10v10H7z",
  play: "M8 5l11 7-11 7z",
  video: "M4 7h11v10H4z M15 10.5l5-3v9l-5-3",
  sound: "M4 10v4h3l5 4V6l-5 4z M16 9a4 4 0 0 1 0 6 M18.5 6.5a7.5 7.5 0 0 1 0 11",
  loop: "M17 3l3 3-3 3 M4 11V9a3 3 0 0 1 3-3h13 M7 21l-3-3 3-3 M20 13v2a3 3 0 0 1-3 3H4",
  forward: "M5 6l7 6-7 6z M13 6l7 6-7 6z",
};

export function Icon({ name, size = 18, fill = false }) {
  return html`<svg
    class="icon"
    width=${size}
    height=${size}
    viewBox="0 0 24 24"
    fill=${fill ? "currentColor" : "none"}
    stroke="currentColor"
    stroke-width="1.7"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path d=${PATHS[name] || ""} />
  </svg>`;
}

export function IconButton({ icon, label, onClick, active = false, fill = false, size = 18, disabled = false, cls = "" }) {
  return html`<button
    type="button"
    class=${`icon-button ${active ? "is-active" : ""} ${cls}`}
    title=${label}
    aria-label=${label}
    aria-pressed=${active}
    disabled=${disabled}
    onClick=${(event) => {
      event.stopPropagation();
      onClick && onClick(event);
    }}
  >
    <${Icon} name=${icon} size=${size} fill=${fill} />
  </button>`;
}

// ------------------------------------------------------------------------------ styles

/** A style's colours as one bar, each colour as wide as its share. */
export function Swatches({ swatches, height = 10 }) {
  const list = (swatches || []).map((swatch) => (typeof swatch === "string" ? { hex: swatch, share: 1 } : swatch));
  if (!list.length) return null;
  return html`<div class="swatches" style=${{ height: `${height}px` }}>
    ${list.map(
      (swatch) =>
        html`<span title=${swatch.hex} style=${{ background: swatch.hex, flexGrow: Math.max(0.05, swatch.share) }}></span>`,
    )}
  </div>`;
}

const FIELDS = ["medium", "rendering", "palette", "light", "texture", "mood"];

/** The six things a style is described by, as a small table. */
export function StyleWords({ style }) {
  if (!style) return null;
  if (!style.read) {
    return html`<p class="muted small">Only its colours are known so far. Its medium, light and mood appear once the vision model has read it.</p>`;
  }
  return html`<dl class="style-words">
    ${FIELDS.filter((field) => style.words[field]).map(
      (field) => html`<dt>${field}</dt><dd>${style.words[field]}</dd>`,
    )}
  </dl>`;
}

// ------------------------------------------------------------------------------- menus

/** A button that opens a small menu. `items`: [{ label, icon, onClick, danger, checked }] or "divider". */
export function Menu({ icon = "dots", label = "More", items, cls = "", text = null, align = "right" }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const close = (event) => {
      if (!root.current || !root.current.contains(event.target)) setOpen(false);
    };
    const key = (event) => event.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", key);
    };
  }, [open]);
  const list = typeof items === "function" ? (open ? items() : []) : items;
  return html`<div class=${`menu ${cls}`} ref=${root}>
    <button
      type="button"
      class=${text ? "button" : "icon-button"}
      title=${label}
      aria-label=${label}
      aria-expanded=${open}
      onClick=${(event) => {
        event.stopPropagation();
        setOpen(!open);
      }}
    >
      <${Icon} name=${icon} />${text && html`<span>${text}</span>`}
    </button>
    ${open &&
    html`<div class=${`menu-list align-${align}`} role="menu">
      ${list.length === 0 && html`<div class="menu-empty">Nothing here yet</div>`}
      ${list.map((item) =>
        item === "divider"
          ? html`<hr />`
          : html`<button
              type="button"
              role="menuitem"
              class=${item.danger ? "is-danger" : ""}
              onClick=${(event) => {
                event.stopPropagation();
                setOpen(false);
                item.onClick();
              }}
            >
              ${item.icon && html`<${Icon} name=${item.checked ? "check" : item.icon} size=${16} />`}
              <span>${item.label}</span>
            </button>`,
      )}
    </div>`}
  </div>`;
}

/** A dialog over the page. Escape and a click outside close it. */
export function Dialog({ title, onClose, children, wide = false }) {
  useEffect(() => {
    const key = (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
      }
    };
    document.addEventListener("keydown", key, true);
    return () => document.removeEventListener("keydown", key, true);
  }, [onClose]);
  return html`<div class="dialog-veil" onMouseDown=${(event) => event.target === event.currentTarget && onClose()}>
    <div class=${`dialog ${wide ? "is-wide" : ""}`} role="dialog" aria-label=${title}>
      <header>
        <h2>${title}</h2>
        <${IconButton} icon="x" label="Close" onClick=${onClose} />
      </header>
      <div class="dialog-body">${children}</div>
    </div>
  </div>`;
}

/** Ask for one line of text. `onSubmit` may return false (or a rejected promise) to keep it open. */
export function Ask({ title, label, value = "", action = "Save", placeholder = "", onSubmit, onClose }) {
  const [text, setText] = useState(value);
  const input = useRef(null);
  useEffect(() => {
    input.current && input.current.select();
  }, []);
  const submit = async (event) => {
    event.preventDefault();
    if (!text.trim()) return;
    if ((await onSubmit(text.trim())) !== false) onClose();
  };
  return html`<${Dialog} title=${title} onClose=${onClose}>
    <form class="ask" onSubmit=${submit}>
      <label>
        <span>${label}</span>
        <input ref=${input} value=${text} placeholder=${placeholder} onInput=${(event) => setText(event.target.value)} />
      </label>
      <div class="row end">
        <button type="button" class="button" onClick=${onClose}>Cancel</button>
        <button type="submit" class="button is-primary" disabled=${!text.trim()}>${action}</button>
      </div>
    </form>
  <//>`;
}

export function Toasts() {
  const { toasts } = useStore();
  return html`<div class="toasts" aria-live="polite">
    ${toasts.map((item) => html`<div key=${item.id} class=${`toast is-${item.kind}`}>${item.message}</div>`)}
  </div>`;
}

export function Empty({ title, children }) {
  return html`<div class="empty">
    <h3>${title}</h3>
    <p>${children}</p>
  </div>`;
}

// ----------------------------------------------------------------------------- helpers

export async function copy(text, what = "Copied") {
  try {
    await navigator.clipboard.writeText(text);
    toast(what);
  } catch {
    window.prompt("Copy this:", text);
  }
}

const DAY = 86400000;

export function dayLabel(seconds) {
  const date = new Date(seconds * 1000);
  const start = (value) => new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
  const days = Math.round((start(new Date()) - start(date)) / DAY);
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  return date.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: days > 300 ? "numeric" : undefined });
}

export const dayKey = (seconds) => {
  const date = new Date(seconds * 1000);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
};

/** Split a prompt as it is stored ("a fox --ar 16:9 --seed 4") into its text and its parameters. */
export function splitPrompt(prompt) {
  const at = (prompt || "").search(/(^|\s)--[a-zA-Z]/);
  if (at === -1) return { text: prompt || "", params: [] };
  const params = prompt
    .slice(at)
    .trim()
    .split(/\s+(?=--[a-zA-Z])/)
    .map((piece) => piece.replace(/^--/, ""));
  return { text: prompt.slice(0, at).trim(), params };
}

/** How long a video is, the way it is written on it: "5s". */
export const clipLength = (seconds) => `${Math.max(1, Math.round(seconds))}s`;

/**
 * What marks a tile as a video: a play sign, how long it is, and whether it has sound.
 * `image` is a picture as the server describes it; nothing is drawn for one that is not a video.
 */
export function ClipBadge({ image }) {
  if (!image.video) return null;
  return html`<span class="clip-badge" title=${image.sound ? "A video, with sound" : "A video"}>
    <${Icon} name="play" size=${10} fill=${true} /><span>${clipLength(image.seconds)}</span>
    ${image.sound && html`<${Icon} name="sound" size=${12} />`}
  </span>`;
}

/**
 * A video over the picture that stands for it, without sound, for as long as it is drawn. It
 * is drawn while the pointer is over a tile, so that a page of videos loads none of them.
 * `at` (0 to 1) holds it at that point of its length instead of playing it: that is scrubbing.
 */
export function ClipPreview({ src, at = null }) {
  const node = useRef(null);
  useEffect(() => {
    const video = node.current;
    if (!video) return undefined;
    video.muted = true; // set by hand: a browser only starts a video on its own when it is silent
    if (at == null) {
      const playing = video.play();
      if (playing) playing.catch(() => {});
      return undefined;
    }
    video.pause();
    const seek = () => {
      if (video.duration) video.currentTime = Math.min(Math.max(0, video.duration - 0.04), at * video.duration);
    };
    if (video.readyState >= 1) seek();
    else video.addEventListener("loadedmetadata", seek, { once: true });
    return () => video.removeEventListener("loadedmetadata", seek);
  }, [at, src]);
  return html`<video class="clip-preview" ref=${node} src=${src} muted loop playsinline preload="auto"></video>`;
}

/**
 * A small menu where the pointer is, as a right-click gives one. `at` is { x, y } in the
 * window; `items` are { label, icon, href, download, onClick }.
 */
export function ContextMenu({ at, items, onClose }) {
  useEffect(() => {
    const close = () => onClose();
    const key = (event) => event.key === "Escape" && onClose();
    document.addEventListener("mousedown", close);
    document.addEventListener("scroll", close, true);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("scroll", close, true);
      document.removeEventListener("keydown", key);
    };
  }, []);
  const left = Math.min(at.x, window.innerWidth - 220);
  const top = Math.min(at.y, window.innerHeight - 44 * items.length - 12);
  return html`<div
    class="menu-list is-floating"
    role="menu"
    style=${{ left: `${left}px`, top: `${top}px` }}
    onMouseDown=${(event) => event.stopPropagation()}
    onClick=${(event) => event.stopPropagation()}
  >
    ${items.map((item) =>
      item.href
        ? html`<a role="menuitem" href=${item.href} download=${item.download || true} onClick=${() => onClose()}>
            ${item.icon && html`<${Icon} name=${item.icon} size=${16} />`}<span>${item.label}</span>
          </a>`
        : html`<button
            type="button"
            role="menuitem"
            onClick=${() => {
              onClose();
              item.onClick();
            }}
          >
            ${item.icon && html`<${Icon} name=${item.icon} size=${16} />`}<span>${item.label}</span>
          </button>`,
    )}
  </div>`;
}

/** Whether a file being dragged or pasted is a picture. */
export const pictures = (list) => [...(list || [])].filter((file) => file.type && file.type.startsWith("image/"));
