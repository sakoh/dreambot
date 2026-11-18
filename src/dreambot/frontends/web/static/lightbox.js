// The lightbox: one image large, everything known about it, and every action Midjourney puts
// under an image — vary, upscale, rerun, remix, retexture, zoom, pan, square, edit — plus where
// to keep it.

import { html, useEffect, useRef, useState } from "./vendor/preact-htm.js";
import { api, thumb } from "./api.js";
import {
  act,
  addRef,
  attempt,
  closeLightbox,
  go,
  keepJob,
  openLightbox,
  refreshFolders,
  refreshJob,
  refreshLookbooks,
  setDraft,
  toast,
  useStore,
} from "./store.js";
import { Ask, Icon, IconButton, Menu, Swatches, copy, splitPrompt } from "./ui.js";

function Group({ title, children }) {
  return html`<div class="action-group">
    <h4>${title}</h4>
    <div class="action-row">${children}</div>
  </div>`;
}

function Action({ label, onClick, icon, title }) {
  return html`<button type="button" class="button" title=${title || label} onClick=${onClick}>
    ${icon && html`<${Icon} name=${icon} size=${16} />`}<span>${label}</span>
  </button>`;
}

/** A line of text to send with an action: an edit instruction, a remix prompt, a custom zoom. */
function Inline({ placeholder, action, value = "", onSubmit, onCancel, multiline = false }) {
  const [text, setText] = useState(value);
  const submit = (event) => {
    event.preventDefault();
    if (text.trim()) onSubmit(text.trim());
  };
  return html`<form class="inline" onSubmit=${submit}>
    ${multiline
      ? html`<textarea
          rows="3"
          autofocus
          placeholder=${placeholder}
          value=${text}
          onInput=${(event) => setText(event.target.value)}
          onKeyDown=${(event) => event.key === "Enter" && !event.shiftKey && submit(event)}
        ></textarea>`
      : html`<input autofocus placeholder=${placeholder} value=${text} onInput=${(event) => setText(event.target.value)} />`}
    <div class="row end">
      <button type="button" class="button" onClick=${onCancel}>Cancel</button>
      <button type="submit" class="button is-primary" disabled=${!text.trim()}>${action}</button>
    </div>
  </form>`;
}

/**
 * What to draw a picture as when it is retextured: a LoRA, words of its own, or both.
 * `loras` are the LoRAs that can be used, `usual` the name of the person's own one (or "").
 */
function Retexture({ loras, usual, onSubmit, onCancel }) {
  const [lora, setLora] = useState(loras.some((each) => each.name === usual) ? usual : loras.length ? loras[0].name : "");
  const [text, setText] = useState("");
  const [hold, setHold] = useState("");
  const submit = (event) => {
    event.preventDefault();
    const typed = text.trim();
    // The choices are sent the way they are typed, "No LoRA" included; what is typed by hand wins.
    const named = /(^|\s)--loras?(\s|$)/i.test(typed) ? "" : `--lora ${lora || "none"}`;
    const held = hold && !/(^|\s)--strength(\s|$)/i.test(typed) ? `--strength ${hold}` : "";
    onSubmit([typed, named, held].filter(Boolean).join(" "));
  };
  return html`<form class="inline" onSubmit=${submit}>
    <label class="inline-lora" title="The look it is drawn in. A LoRA is one fine-tuned look; add more with dreambot loras add.">
      <span>Draw it as</span>
      <select autofocus value=${lora} onChange=${(event) => setLora(event.target.value)}>
        ${loras.map((each) => html`<option value=${each.name}>${each.title}</option>`)}
        <option value="">No LoRA</option>
      </select>
    </label>
    <label class="inline-lora" title="How much of the picture is kept. The more of its layout is held, the less room the new look has.">
      <span>Keep</span>
      <select value=${hold} onChange=${(event) => setHold(event.target.value)}>
        <option value="0.35">Less of the picture, more of the look</option>
        <option value="">The layout (balanced)</option>
        <option value="0.7">More of the picture, less of the look</option>
      </select>
    </label>
    <textarea
      rows="2"
      placeholder="Optional: a prompt of its own, e.g. the same scene as a watercolour. Empty keeps its own."
      value=${text}
      onInput=${(event) => setText(event.target.value)}
      onKeyDown=${(event) => event.key === "Enter" && !event.shiftKey && submit(event)}
    ></textarea>
    <div class="row end">
      <button type="button" class="button" onClick=${onCancel}>Cancel</button>
      <button type="submit" class="button is-primary">Retexture</button>
    </div>
  </form>`;
}

export function Lightbox() {
  const { lightbox, jobs, order, lookbooks, folders, app } = useStore();
  const [detail, setDetail] = useState(null);
  const [asking, setAsking] = useState(null); // "edit" | "remix" | "retexture" | "zoom" | "folder" | "lookbook"
  const root = useRef(null);
  const jobId = lightbox && lightbox.job;
  const job = jobId ? jobs.get(jobId) : null;

  useEffect(() => {
    setAsking(null);
    setDetail(null);
    if (!jobId) return;
    let stale = false;
    refreshJob(jobId).then((answer) => !stale && answer && setDetail(answer));
    refreshFolders();
    if (!lookbooks) refreshLookbooks();
    return () => (stale = true);
  }, [jobId]);

  // Take the keyboard when it opens: the arrow keys and Escape belong to the lightbox now, not
  // to the prompt box the person may have been typing in.
  const open = Boolean(lightbox);
  useEffect(() => {
    if (open && root.current) root.current.focus();
  }, [open]);

  // What the arrow keys walk through: the gallery the image was opened from, or else every
  // image the feed knows about, in order.
  let sequence = lightbox && lightbox.sequence;
  if (!sequence) {
    sequence = [];
    for (const id of order) {
      const other = jobs.get(id);
      if (other) for (const picture of other.images) if (!picture.trashed) sequence.push({ job: id, index: picture.index });
    }
  }
  const here = lightbox ? sequence.findIndex((item) => item.job === lightbox.job && item.index === lightbox.index) : -1;
  const step = (by) => {
    const target = sequence[here + by];
    if (here !== -1 && target) openLightbox(target.job, target.index);
  };

  useEffect(() => {
    if (!lightbox) return undefined;
    const key = (event) => {
      if (event.target.matches("input, textarea, select")) return;
      if (event.key === "Escape") closeLightbox();
      else if (event.key === "ArrowLeft") step(-1);
      else if (event.key === "ArrowRight") step(1);
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  });

  if (!lightbox || !job) return null;
  const image = job.images.find((picture) => picture.index === lightbox.index) || job.images[0];
  if (!image) return null;
  const index = image.index;
  const { text, params } = splitPrompt(job.prompt);
  const run = async (kind, more = {}) => {
    const answer = await act(job.id, kind, { index, ...more });
    if (answer && answer.jobs.length) {
      closeLightbox();
      go("create");
      window.scrollTo({ top: 0 });
    }
    return answer;
  };
  const picture = { job: job.id, index };
  const use = (role) => {
    addRef({ role, picture, url: image.url, name: `#${job.id}` });
    toast(`Added to the prompt bar as ${role === "image" ? "an image prompt" : `a ${role} reference`}.`);
  };
  const bulk = (action, more = {}) => attempt(() => api.post("/api/images/bulk", { action, items: [picture], ...more }));
  const inFolders = new Set((detail && detail.folders[String(index)]) || []);
  const toggleFolder = async (folder) => {
    const adding = !inFolders.has(folder.id);
    if (await bulk(adding ? "folder-add" : "folder-remove", { folder: folder.id })) {
      toast(adding ? `Added to ${folder.name}.` : `Removed from ${folder.name}.`);
      refreshJob(job.id).then((answer) => answer && setDetail(answer));
      refreshFolders();
    }
  };
  const addToBook = async (name) => {
    const answer = await act(job.id, "look", { index, prompt: name });
    if (answer) refreshLookbooks();
    return Boolean(answer);
  };
  const trash = async () => {
    if (await bulk(image.trashed ? "restore" : "trash")) {
      toast(image.trashed ? "Restored." : "Moved to trash. It is under Organize → Trash.");
      await refreshJob(job.id);
      if (!image.trashed) (sequence[here + 1] || sequence[here - 1]) ? step(sequence[here + 1] ? 1 : -1) : closeLightbox();
    }
  };
  const square = image.width === image.height;
  const remixDefault = app && app.settings.remix;
  const vary = (kind) => (remixDefault ? setAsking(kind) : run(kind));

  return html`<div
    class="lightbox"
    ref=${root}
    tabindex="-1"
    role="dialog"
    aria-label="Image"
    onMouseDown=${(event) => event.target === event.currentTarget && closeLightbox()}
  >
    <div class="lightbox-stage" onMouseDown=${(event) => event.target === event.currentTarget && closeLightbox()}>
      ${here > 0 && html`<${IconButton} icon="left" label="Previous (←)" cls="nav prev" size=${26} onClick=${() => step(-1)} />`}
      <img
        key=${image.url}
        src=${image.url}
        alt=${job.text}
        style=${{ aspectRatio: `${image.width} / ${image.height}` }}
        draggable="true"
        onDragStart=${(event) =>
          event.dataTransfer.setData(
            "application/x-dreambot-picture",
            JSON.stringify({ picture, url: image.url, name: `#${job.id}` }),
          )}
      />
      ${here !== -1 &&
      here < sequence.length - 1 &&
      html`<${IconButton} icon="right" label="Next (→)" cls="nav next" size=${26} onClick=${() => step(1)} />`}
    </div>
    <aside class="lightbox-panel">
      <header>
        <div class="row">
          <${IconButton}
            icon="heart"
            label=${image.favorite ? "Unlike" : "Like"}
            active=${image.favorite}
            fill=${image.favorite}
            onClick=${() => act(job.id, "favorite", { index })}
          />
          <a class="icon-button" href=${`${image.url}?download=1`} title="Download the original" aria-label="Download the original">
            <${Icon} name="download" />
          </a>
          <${Menu}
            icon="folder"
            label="Folders"
            items=${() => [
              ...folders.map((folder) => ({
                label: folder.name,
                icon: "folder",
                checked: inFolders.has(folder.id),
                onClick: () => toggleFolder(folder),
              })),
              ...(folders.length ? ["divider"] : []),
              { label: "New folder…", icon: "plus", onClick: () => setAsking("folder") },
            ]}
          />
          <${Menu}
            icon="lookbook"
            label="Add to a Lookbook"
            items=${() => [
              ...(lookbooks || []).map((book) => ({ label: book.name, icon: "lookbook", onClick: () => addToBook(book.name) })),
              ...((lookbooks || []).length ? ["divider"] : []),
              { label: "New Lookbook…", icon: "plus", onClick: () => setAsking("lookbook") },
            ]}
          />
          <${IconButton} icon=${image.trashed ? "restore" : "trash"} label=${image.trashed ? "Restore from trash" : "Move to trash"} onClick=${trash} />
          <${Menu}
            label="Copy"
            icon="copy"
            items=${[
              { label: "Copy prompt", icon: "copy", onClick: () => copy(job.prompt, "Prompt copied") },
              { label: "Copy seed", icon: "copy", onClick: () => copy(String(image.seed), "Seed copied") },
              { label: "Copy job id", icon: "copy", onClick: () => copy(job.id, "Job id copied") },
              ...(image.prompt
                ? [{ label: "Copy what it was rendered from", icon: "copy", onClick: () => copy(image.prompt, "Copied") }]
                : []),
            ]}
          />
        </div>
        <${IconButton} icon="x" label="Close (Esc)" onClick=${closeLightbox} />
      </header>

      <div class="lightbox-scroll">
        <p class="job-prompt">${job.instruction && html`<span class="tag">edit</span>`} ${text || html`<span class="muted">(no text)</span>`}</p>
        ${params.length > 0 && html`<div class="chips">${params.map((param) => html`<span class="chip is-param">${param}</span>`)}</div>`}
        ${image.prompt &&
        html`<details class="small">
          <summary>What this image was rendered from</summary>
          <p>${image.prompt}</p>
        </details>`}
        ${job.style &&
        html`<div class="panel-style">
          <${Swatches} swatches=${job.style.swatches} height=${12} />
          <strong>${job.style.name || "Style"}</strong>
          <span class="small">${job.style.summary}</span>
          <span class="small muted">
            ${job.style.sources.map((part) => `${part.label} ${Math.round(part.share * 100)}%`).join(" · ")}
            ${job.style.board ? " · shown to the model as a swatch board" : " · applied in words"}
          </span>
        </div>`}
        <div class="job-meta small muted">
          <span>${job.kind}</span><span>${job.model}</span>
          ${(job.loras || []).map((lora) => html`<span title="LoRA">${lora.name}${lora.scale === 1 ? "" : ` ×${lora.scale}`}</span>`)}
          <span>${image.width}×${image.height}</span>
          <span>seed ${image.seed}</span><span>#${job.id}</span>
          <span>${new Date(job.created * 1000).toLocaleString()}</span>
        </div>

        ${asking === "edit" &&
        html`<${Inline}
          placeholder="What should change? e.g. make it night"
          action="Edit"
          onCancel=${() => setAsking(null)}
          onSubmit=${(instruction) => run("edit", { prompt: instruction })}
        />`}
        ${asking === "zoom" &&
        html`<${Inline}
          placeholder="How far to zoom out, between 1 and 4 — e.g. 1.3"
          action="Zoom out"
          onCancel=${() => setAsking(null)}
          onSubmit=${(factor) => run("zoom", { value: factor.replace(/x$/i, "") })}
        />`}
        ${(asking === "vary_subtle" || asking === "vary_strong" || asking === "remix") &&
        html`<${Inline}
          multiline
          placeholder="The prompt the variations should follow"
          action=${asking === "remix" ? "Remix" : "Vary"}
          value=${job.prompt.replace(/\s--(seed|sameseed)\s+\d+/g, "")}
          onCancel=${() => setAsking(null)}
          onSubmit=${(prompt) => run(asking === "remix" ? "vary_strong" : asking, { prompt })}
        />`}
        ${asking === "retexture" &&
        html`<${Retexture}
          loras=${((app && app.loras) || []).filter((lora) => lora.retexture)}
          usual=${((app && app.settings.loras[0]) || "").split(":")[0]}
          onCancel=${() => setAsking(null)}
          onSubmit=${(prompt) => run("retexture", { prompt })}
        />`}

        <${Group} title="Vary">
          <${Action} label="Subtle" icon="wand" onClick=${() => vary("vary_subtle")} />
          <${Action} label="Strong" icon="refresh" onClick=${() => vary("vary_strong")} />
          <${Action} label="Remix…" icon="edit" title="Variations that follow a new prompt" onClick=${() => setAsking("remix")} />
          <${Action} label="Retexture…" icon="styles" title="Draw it again in another look, keeping its layout" onClick=${() => setAsking("retexture")} />
        <//>
        <${Group} title="Upscale">
          <${Action} label="Subtle" icon="expand" title="Twice the size, faithful to the pixel" onClick=${() => run("upscale_subtle")} />
          <${Action} label="Creative" icon="create" title="Twice the size, with fine detail re-imagined" onClick=${() => run("upscale_creative")} />
        <//>
        <${Group} title="Reframe">
          <${Action} label="Zoom 1.5×" icon="zoom" onClick=${() => run("zoom", { value: "1.5" })} />
          <${Action} label="Zoom 2×" icon="zoom" onClick=${() => run("zoom", { value: "2" })} />
          <${Action} label="Custom…" onClick=${() => setAsking("zoom")} />
          ${!square && html`<${Action} label="Make square" icon="square" onClick=${() => run("square")} />`}
        <//>
        <${Group} title="Pan">
          ${["left", "right", "up", "down"].map(
            (direction) => html`<${IconButton} icon=${direction} label=${`Extend ${direction}`} cls="button" onClick=${() => run("pan", { value: direction })} />`,
          )}
        <//>
        <${Group} title="More">
          <${Action} label="Edit…" icon="edit" title="Change the image by instruction" onClick=${() => setAsking("edit")} />
          <${Action} label="Rerun" icon="refresh" title="The same prompt again, with new seeds" onClick=${() => run("reroll")} />
          <${Action}
            label="Describe"
            icon="text"
            title="Write four prompts for this image"
            onClick=${async () => {
              if (await attempt(() => api.post("/api/describe", { picture }))) {
                toast("Writing four prompts for it — they will appear in the feed.");
                closeLightbox();
                go("create");
              }
            }}
          />
        <//>
        <${Group} title="Use in a prompt">
          <${Action} label="Image prompt" icon="image" onClick=${() => use("image")} />
          <${Action} label="Style" icon="styles" onClick=${() => use("style")} />
          <${Action} label="Subject" icon="person" onClick=${() => use("subject")} />
          <${Action}
            label="Prompt"
            icon="text"
            title="Put this prompt in the bar"
            onClick=${() => {
              setDraft({ text: job.prompt });
              closeLightbox();
              go("create");
            }}
          />
        <//>

        ${detail &&
        (detail.parent || detail.children.length > 0) &&
        html`<div class="lineage">
          ${detail.parent &&
          html`<div>
            <h4>Made from</h4>
            <div class="lineage-row">
              ${detail.parent.images
                .filter((picture) => picture.index === job.parentIndex)
                .map(
                  (picture) =>
                    html`<img
                      src=${thumb(picture.url, 256)}
                      alt=""
                      onClick=${() => {
                        keepJob(detail.parent);
                        openLightbox(detail.parent.id, picture.index);
                      }}
                    />`,
                )}
            </div>
          </div>`}
          ${detail.children.length > 0 &&
          html`<div>
            <h4>Made from this job</h4>
            <div class="lineage-row">
              ${detail.children.flatMap((child) =>
                child.images.slice(0, 4).map(
                  (picture) =>
                    html`<img
                      src=${thumb(picture.url, 256)}
                      alt=""
                      title=${child.kind}
                      onClick=${() => {
                        keepJob(child);
                        openLightbox(child.id, picture.index);
                      }}
                    />`,
                ),
              )}
            </div>
          </div>`}
        </div>`}
      </div>
    </aside>
    ${asking === "folder" &&
    html`<${Ask}
      title="New folder"
      label="Name"
      action="Create and add"
      onClose=${() => setAsking(null)}
      onSubmit=${async (name) => {
        const answer = await attempt(() => api.post("/api/folders", { name }));
        if (!answer) return false;
        await toggleFolder(answer.folder);
        return true;
      }}
    />`}
    ${asking === "lookbook" &&
    html`<${Ask} title="New Lookbook" label="Name" action="Create and add" placeholder="e.g. Autumn diner" onClose=${() => setAsking(null)} onSubmit=${addToBook} />`}
  </div>`;
}
