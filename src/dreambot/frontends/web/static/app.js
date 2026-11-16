// The Dreambot web app. No build step: these are plain ES modules, and the one library they use
// (Preact with htm, in ./vendor) is loaded the same way.

import { html, render, useEffect } from "./vendor/preact-htm.js";
import { CreatePage } from "./create.js";
import { Lightbox } from "./lightbox.js";
import { LookbooksPage } from "./lookbooks.js";
import { OrganizePage } from "./organize.js";
import { StylesPage } from "./styles.js";
import { connect, loadFeed, readRoute, refreshApp, useStore } from "./store.js";
import { Icon, Toasts } from "./ui.js";

const PAGES = {
  create: { title: "Create", icon: "create", page: CreatePage },
  organize: { title: "Organize", icon: "organize", page: OrganizePage },
  lookbooks: { title: "Lookbooks", icon: "lookbook", page: LookbooksPage },
  styles: { title: "Styles", icon: "styles", page: StylesPage },
};

function Sidebar() {
  const { route, app, live } = useStore();
  const working = live.size;
  return html`<nav class="sidebar">
    <a class="brand" href="#/create">Dreambot</a>
    ${Object.entries(PAGES).map(
      ([name, info]) => html`<a key=${name} href=${`#/${name}`} class=${`nav-link ${route === name ? "is-current" : ""}`} aria-current=${route === name ? "page" : undefined}>
        <${Icon} name=${info.icon} /><span>${info.title}</span>
      </a>`,
    )}
    <div class="grow"></div>
    <div class="sidebar-foot small muted">
      ${working > 0 && html`<div class="working"><span class="pulse"></span>${working} in progress</div>`}
      ${app && html`<div title="The model a plain prompt uses">${app.model || "No model installed"}</div>`}
      ${app && app.backend === "fake" && html`<div class="tag">placeholder images</div>`}
      ${app && html`<div>v${app.version} · local</div>`}
    </div>
  </nav>`;
}

function App() {
  const { route } = useStore();
  useEffect(() => {
    readRoute();
    window.addEventListener("hashchange", readRoute);
    refreshApp();
    loadFeed();
    connect();
    return () => window.removeEventListener("hashchange", readRoute);
  }, []);
  useEffect(() => {
    document.title = `${PAGES[route].title} · Dreambot`;
    window.scrollTo({ top: 0 });
  }, [route]);
  const Page = PAGES[route].page;
  return html`<div class="shell">
    <${Sidebar} />
    <main><${Page} /></main>
    <${Lightbox} />
    <${Toasts} />
  </div>`;
}

render(html`<${App} />`, document.getElementById("app"));
