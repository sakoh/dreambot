// The page's state, kept in one place: what the server says, the feed, the prompt being written.
// Components read it through useStore() and are redrawn whenever emit() is called.

import { useEffect, useReducer } from "./vendor/preact-htm.js";
import { api } from "./api.js";

const DRAFT_KEY = "dreambot-draft";
const listeners = new Set();
let scheduled = false;

/** What the prompt bar's Video button and its options start as: off, and everything as Dreambot is set up. */
const NO_VIDEO = { on: false, motion: "", loop: false, seconds: "", from: "" };

/**
 * What a video made from words alone starts on: "picture" (one drawn from the words first) or
 * "words" (nothing: the video model films the words itself). The bar's own choice, or else the
 * person's setting.
 */
export function videoFrom(draft = store.draft) {
  const usual = !store.app || store.app.settings.picture_first ? "picture" : "words";
  return (draft.video && draft.video.from) || usual;
}
/** How many seconds an extension adds unless the bar says otherwise (Midjourney's four). */
export const EXTEND_SECONDS = 4;

function loadDraft() {
  const empty = {
    text: "",
    refs: [],
    params: { ar: "", chaos: 0, weird: 0 },
    pinned: false,
    video: NO_VIDEO,
    /** The video the bar is set to continue ("Extend Manual"): { job, index, poster, seconds } or null. */
    extend: null,
  };
  try {
    const saved = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
    if (!saved) return empty;
    return {
      ...empty,
      ...saved,
      params: { ...empty.params, ...(saved.params || {}) },
      video: { ...NO_VIDEO, ...(saved.video || {}) },
      refs: saved.refs || [],
      extend: null, // what was being continued does not outlive a reload
    };
  } catch {
    return empty;
  }
}

export const store = {
  /** /api/state: settings, installed models, queue length. */
  app: null,
  /** Jobs by id, and the ids in the order the feed shows them (newest first). */
  jobs: new Map(),
  order: [],
  next: null,
  loading: false,
  search: "",
  /** Progress of jobs that are queued or running: id -> { type, fraction, position, preview }. */
  live: new Map(),
  /**
   * The prompt being written: its text, its reference pictures and styles, its sticky parameters,
   * and whether it asks for a video (and for what kind of one).
   */
  draft: loadDraft(),
  /** The image open in the lightbox: { job, index, sequence } with ids, or null. */
  lightbox: null,
  toasts: [],
  uploads: [],
  lookbooks: null,
  folders: [],
  /** The ingredients on the style blend bench. */
  bench: [],
  /** Filters for the Organize page to start with, set by a link from another page. */
  organize: null,
  route: "create",
};

export function emit() {
  if (scheduled) return;
  scheduled = true;
  queueMicrotask(() => {
    scheduled = false;
    for (const listener of listeners) listener();
  });
}

export function useStore() {
  const [, redraw] = useReducer((count) => count + 1, 0);
  useEffect(() => {
    listeners.add(redraw);
    return () => listeners.delete(redraw);
  }, []);
  return store;
}

// ------------------------------------------------------------------------------ toasts

let toastId = 0;

export function toast(message, kind = "info") {
  const id = ++toastId;
  store.toasts = [...store.toasts, { id, message, kind }];
  emit();
  setTimeout(() => {
    store.toasts = store.toasts.filter((item) => item.id !== id);
    emit();
  }, kind === "error" ? 7000 : 3500);
}

/** Run something that talks to the server, showing its error to the person if it fails. */
export async function attempt(work, done) {
  try {
    const result = await work();
    if (done) toast(done);
    return result;
  } catch (error) {
    toast(error.message || String(error), "error");
    return undefined;
  }
}

// ------------------------------------------------------------------------------- draft

export function saveDraft() {
  const { text, params, pinned, refs, video } = store.draft;
  // References are only kept across a reload when the person pinned them.
  localStorage.setItem(DRAFT_KEY, JSON.stringify({ text, params, pinned, video, refs: pinned ? refs : [] }));
}

export function setDraft(changes) {
  store.draft = { ...store.draft, ...changes };
  saveDraft();
  emit();
}

let refId = 0;

/**
 * Add a reference to the prompt being written.
 * role: "image" (image prompt; for a video, the frame it starts on), "style", "subject", "look",
 * or "end" (the frame a video ends on).
 * A picture reference has `picture` ({upload} or {job, index}) and `url`; a style code has `code`;
 * a saved style has `saved` (its one-word name); a Lookbook has `look` (name or code).
 */
export function addRef(ref) {
  const same = (other) =>
    other.role === ref.role &&
    JSON.stringify([other.picture, other.code, other.saved, other.look]) ===
      JSON.stringify([ref.picture, ref.code, ref.saved, ref.look]);
  if (store.draft.refs.some(same)) return;
  // A video starts on one picture and ends on one: a new one takes the place of the last.
  const single = ref.role === "end" || (ref.role === "image" && store.draft.video.on);
  const kept = single ? store.draft.refs.filter((other) => other.role !== ref.role) : store.draft.refs;
  setDraft({ refs: [...kept, { weight: 1, ...ref, id: `r${++refId}-${Date.now()}` }] });
}

/** Switch the prompt bar between pictures and video. */
export function setVideo(changes) {
  const video = { ...store.draft.video, ...changes };
  // An ending frame belongs to a video: it goes when the bar goes back to pictures. So does
  // the video that was going to be continued.
  const refs = video.on ? store.draft.refs : store.draft.refs.filter((ref) => ref.role !== "end");
  setDraft({ video, refs, extend: video.on ? store.draft.extend : null });
}

/** The words of a prompt as a job keeps it, without the parameters after them. */
const wordsOf = (prompt) => {
  const at = (prompt || "").search(/(^|\s)--[a-zA-Z]/);
  return (at === -1 ? prompt || "" : prompt.slice(0, at)).trim();
};

/**
 * "Animate Manually": put a picture in the prompt bar as the frame a video starts on, with the
 * words it was made from, to be changed into what should happen before it is sent.
 */
export function animateManually(job, image) {
  const start = { role: "image", picture: { job: job.id, index: image.index }, url: image.url, name: `#${job.id}` };
  setDraft({
    text: wordsOf(job.prompt),
    video: { ...store.draft.video, on: true, loop: false },
    refs: [{ weight: 1, ...start, id: `r${++refId}-${Date.now()}` }],
    extend: null,
  });
}

/**
 * "Extend Manual": set the prompt bar to continue a video, with the words it was made from,
 * to be changed into what happens next. How much is added can be chosen there.
 */
export function extendManually(job, image) {
  setDraft({
    text: wordsOf(job.prompt),
    video: { ...store.draft.video, on: true, seconds: "", loop: false },
    refs: [],
    extend: { job: job.id, index: image.index, poster: image.poster || image.url, seconds: image.seconds },
  });
}

export function updateRef(id, changes) {
  setDraft({ refs: store.draft.refs.map((ref) => (ref.id === id ? { ...ref, ...changes } : ref)) });
}

export function removeRef(id) {
  setDraft({ refs: store.draft.refs.filter((ref) => ref.id !== id) });
}

const weighted = (token, weight) => (Number(weight) === 1 ? token : `${token}::${Number(weight)}`);
const typed = (text, name) => new RegExp(`(^|\\s)--${name}(\\s|$)`).test(text);

/**
 * Turn the draft into what the server is sent: the prompt text with its parameters, and the
 * pictures it refers to as @1, @2, ... Pictures nobody claims are image prompts, as in Midjourney.
 */
export function compose(draft = store.draft) {
  const attachments = [];
  const attach = (ref) => {
    attachments.push(ref.picture);
    return `@${attachments.length}`;
  };
  const filming = Boolean(draft.video && draft.video.on);
  // One of Dreambot's own pictures as the starting frame, and nothing else that shapes a picture.
  const first = draft.refs.find((ref) => ref.role === "image");
  const own = first && first.picture && first.picture.job ? first.picture : null;
  const plain = draft.refs.every((ref) => ref === first || ref.role === "end");
  if (filming && (draft.extend || (own && plain))) {
    // A video is continued, and a picture of Dreambot's animated, by the action their buttons
    // send, with these words: so that what is made is kept with what it was made from.
    const ends = draft.refs.filter((ref) => ref.role === "end").map((ref) => attach(ref));
    const { motion, loop, seconds } = draft.video;
    let words = draft.text.trim();
    const say = (parameter) => (words = `${words} ${parameter}`.trim());
    if (ends.length && !typed(words, "end")) say(`--end ${ends[0]}`);
    // A loop ends on the frame it starts on; a video continued as one is led back to its first.
    if (loop && !ends.length && !typed(words, "loop")) say("--loop");
    if (seconds && !typed(words, "seconds") && !typed(words, "sec")) say(`--seconds ${seconds}`);
    const action = draft.extend
      ? { kind: "extend", job: draft.extend.job, index: draft.extend.index }
      : { kind: "animate", job: own.job, index: own.index };
    return { prompt: words, attachments, action: { ...action, motion: motion || undefined } };
  }
  const subjects = draft.refs.filter((ref) => ref.role === "subject").map((ref) => attach(ref));
  const styles = draft.refs
    .filter((ref) => ref.role === "style")
    .map((ref) => weighted(ref.code || ref.saved || attach(ref), ref.weight));
  // For a video, the image prompt is the frame it starts on.
  const starts = draft.refs.filter((ref) => ref.role === "image").map((ref) => attach(ref));
  const ends = filming ? draft.refs.filter((ref) => ref.role === "end").map((ref) => attach(ref)) : [];
  const looks = draft.refs.filter((ref) => ref.role === "look").map((ref) => weighted(ref.look, ref.weight));

  let prompt = draft.text.trim();
  const add = (parameter) => (prompt = `${prompt} ${parameter}`.trim());
  if (subjects.length) add(`--oref ${subjects.join(" ")}`);
  if (styles.length) add(`--sref ${styles.join(" ")}`);
  if (looks.length) add(`--p ${looks.join(" ")}`);
  // Sticky parameters apply unless the prompt says otherwise itself.
  const { ar, chaos, weird } = draft.params;
  const shaped = typed(prompt, "ar") || typed(prompt, "aspect");
  if (filming) {
    const { motion, loop, seconds, from } = draft.video;
    // With no picture given, one is drawn from the words first, unless the words are to be filmed.
    const wordsOnly = !starts.length && !ends.length;
    const drawn = wordsOnly && videoFrom(draft) === "picture";
    if (!typed(prompt, "video")) add("--video");
    if (wordsOnly && from && !typed(prompt, "from")) add(`--from ${from}`);
    if (ends.length && !typed(prompt, "end")) add(`--end ${ends[0]}`);
    if (motion && !typed(prompt, "motion")) add(`--motion ${motion}`);
    // A loop comes back to the frame it starts on, so it has one, and no other frame to end on.
    if (loop && (starts.length || drawn) && !ends.length && !typed(prompt, "loop")) add("--loop");
    if (seconds && !typed(prompt, "seconds") && !typed(prompt, "sec")) add(`--seconds ${seconds}`);
    // A video has the shape of the picture it starts on; the sticky shape is for one made from words.
    if (ar && !shaped && wordsOnly) add(`--ar ${ar}`);
    if (drawn) {
      // What shapes a picture shapes the one the video starts on.
      if (Number(chaos) > 0 && !typed(prompt, "chaos") && !typed(prompt, "c")) add(`--chaos ${chaos}`);
      if (Number(weird) > 0 && !typed(prompt, "weird") && !typed(prompt, "w")) add(`--weird ${weird}`);
    }
    return { prompt, attachments };
  }
  if (ar && !shaped) add(`--ar ${ar}`);
  if (Number(chaos) > 0 && !typed(prompt, "chaos") && !typed(prompt, "c")) add(`--chaos ${chaos}`);
  if (Number(weird) > 0 && !typed(prompt, "weird") && !typed(prompt, "w")) add(`--weird ${weird}`);
  return { prompt, attachments };
}

// -------------------------------------------------------------------------------- feed

function insert(job) {
  const known = store.jobs.has(job.id);
  store.jobs.set(job.id, job);
  if (!known) {
    // Newest first; a job from another tab or a chat app lands where its time puts it.
    const at = store.order.findIndex((id) => store.jobs.get(id).created < job.created);
    if (at === -1) store.order.push(job.id);
    else store.order.splice(at, 0, job.id);
  }
}

export async function loadFeed({ more = false } = {}) {
  if (store.loading || (more && !store.next)) return;
  store.loading = true;
  emit();
  try {
    const page = await api.get("/api/jobs", { before: more ? store.next : undefined, q: store.search, limit: 24 });
    if (!more) {
      store.jobs = new Map();
      store.order = [];
    }
    for (const job of page.jobs) insert(job);
    store.next = page.next;
  } catch (error) {
    toast(error.message, "error");
  } finally {
    store.loading = false;
    emit();
  }
}

export function putJob(job) {
  if (!job) return;
  if (store.search && !store.jobs.has(job.id)) {
    store.jobs.set(job.id, job); // known for the lightbox, but not shown in a filtered feed
    return emit();
  }
  insert(job);
  emit();
}

/** Remember a job without adding it to the feed: one looked at from the gallery, say. */
export function keepJob(job) {
  if (!job) return;
  if (store.order.includes(job.id)) insert(job);
  else store.jobs.set(job.id, job);
  emit();
}

export async function refreshJob(id) {
  const answer = await attempt(() => api.get(`/api/jobs/${id}`));
  if (answer) keepJob(answer.job);
  return answer;
}

export async function refreshApp() {
  const state = await attempt(() => api.get("/api/state"));
  if (state) {
    store.app = state;
    emit();
  }
}

export async function saveSettings(changes) {
  const state = await attempt(() => api.put("/api/settings", changes));
  if (state) {
    store.app = state;
    emit();
  }
}

export async function refreshUploads() {
  const answer = await attempt(() => api.get("/api/uploads"));
  if (answer) {
    store.uploads = answer.uploads;
    emit();
  }
}

export async function refreshLookbooks() {
  const answer = await attempt(() => api.get("/api/lookbooks"));
  if (answer) {
    store.lookbooks = answer.lookbooks;
    emit();
  }
  return answer && answer.lookbooks;
}

export async function refreshFolders() {
  const answer = await attempt(() => api.get("/api/folders"));
  if (answer) {
    store.folders = answer.folders;
    emit();
  }
}

/** Start jobs from a prompt and put them at the top of the feed. */
export async function imagine(prompt, attachments = []) {
  const answer = await attempt(() => api.post("/api/imagine", { prompt, attachments }));
  if (!answer) return null;
  for (const job of answer.jobs) putJob(job);
  return answer.jobs;
}

/**
 * Press one of the buttons under an image. Returns the server's answer, or undefined on an error.
 * `attachments` are pictures that go with the action, as `@1`, `@2` in its words.
 */
export async function act(jobId, kind, { index, value, prompt, attachments } = {}) {
  const body = { kind, index, value, prompt, attachments };
  const answer = await attempt(() => api.post(`/api/jobs/${jobId}/act`, body));
  if (!answer) return undefined;
  for (const job of answer.jobs) putJob(job);
  if (answer.job) keepJob(answer.job);
  if (answer.jobs.length) toast(answer.jobs.length === 1 ? "Started — it is at the top of Create." : "Started.");
  else if (answer.message && kind !== "favorite") toast(answer.message.replace(/`/g, ""));
  return answer;
}

// ------------------------------------------------------------------------- live events

const ENDED = new Set(["finished", "failed", "cancelled"]);
let source = null;

export function connect() {
  if (source) return;
  source = new EventSource("/api/events");
  source.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.type === "hello") {
      if (store.app) store.app.queue = event.queue;
      return emit();
    }
    if (event.type === "resync") {
      store.live = new Map();
      return void loadFeed();
    }
    if (!event.job) return;
    if (ENDED.has(event.type)) store.live.delete(event.job.id);
    else {
      const before = store.live.get(event.job.id) || {};
      store.live.set(event.job.id, {
        type: event.type,
        // "loading" and "thinking" carry no progress of their own: keep the bar where it was.
        fraction: event.type === "progress" || event.type === "preview" ? event.fraction : before.fraction || 0,
        position: event.position,
        preview: event.preview || before.preview,
        // The pictures its videos are going to start on, by the number of the video.
        starts: event.starts || before.starts,
      });
    }
    if (store.app) store.app.queue = [...store.live.keys()].length;
    putJob(event.job);
  };
  source.onerror = () => {
    // The browser reconnects on its own; when it does, what was missed is fetched again.
    source.onopen = () => {
      source.onopen = null;
      store.live = new Map();
      loadFeed();
    };
  };
}

// ----------------------------------------------------------------------------- routing

export const ROUTES = ["create", "organize", "lookbooks", "styles"];

export function readRoute() {
  const name = location.hash.replace(/^#\/?/, "").split(/[/?]/)[0];
  store.route = ROUTES.includes(name) ? name : "create";
  emit();
}

export function go(route) {
  location.hash = `#/${route}`;
}

/**
 * Open an image large. `sequence` is the list of { job, index } the arrow keys walk through;
 * without it they walk through the Create feed.
 */
export function openLightbox(job, index, sequence = null) {
  store.lightbox = { job, index, sequence: sequence || (store.lightbox && store.lightbox.sequence) || null };
  emit();
}

export function closeLightbox() {
  store.lightbox = null;
  emit();
}
