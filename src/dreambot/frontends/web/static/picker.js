// A dialog for choosing pictures from what has already been made.

import { html, useEffect, useState } from "./vendor/preact-htm.js";
import { api, thumb } from "./api.js";
import { attempt } from "./store.js";
import { Dialog, Icon } from "./ui.js";

export function ImagePicker({ title = "Choose from your images", action = "Add", onPick, onClose }) {
  const [images, setImages] = useState(null);
  const [chosen, setChosen] = useState(new Set());
  const [liked, setLiked] = useState(false);
  useEffect(() => {
    setImages(null);
    attempt(() => api.get("/api/images", { limit: 120, liked: liked ? 1 : "" })).then((answer) => setImages(answer ? answer.images : []));
  }, [liked]);
  const keyOf = (image) => `${image.job}:${image.index}`;
  const flip = (image) => {
    const next = new Set(chosen);
    next.has(keyOf(image)) ? next.delete(keyOf(image)) : next.add(keyOf(image));
    setChosen(next);
  };
  const submit = () => {
    onPick((images || []).filter((image) => chosen.has(keyOf(image))));
    onClose();
  };
  return html`<${Dialog} title=${title} onClose=${onClose} wide>
    <div class="row picker-head">
      <label class="toggle"><input type="checkbox" checked=${liked} onChange=${(event) => setLiked(event.target.checked)} /><span>Liked only</span></label>
      <span class="grow"></span>
      <button type="button" class="button is-primary" disabled=${!chosen.size} onClick=${submit}>${action}${chosen.size ? ` ${chosen.size}` : ""}</button>
    </div>
    ${images === null && html`<p class="muted">Loading…</p>`}
    ${images && images.length === 0 && html`<p class="muted">${liked ? "You have not liked any images yet." : "There are no images yet."}</p>`}
    <div class="gallery-grid picker-grid" style=${{ "--tile": "120px" }}>
      ${(images || []).map(
        (image) => html`<div
          key=${keyOf(image)}
          class=${`gallery-tile is-selecting ${chosen.has(keyOf(image)) ? "is-picked" : ""}`}
          title=${image.prompt}
          onClick=${() => flip(image)}
        >
          <img src=${thumb(image.url, 256)} alt=${image.prompt} loading="lazy" />
          <span class="tick">${chosen.has(keyOf(image)) && html`<${Icon} name="check" size=${14} />`}</span>
        </div>`,
      )}
    </div>
  <//>`;
}
