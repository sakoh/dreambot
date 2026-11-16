// Talking to the Dreambot server. Every call resolves to parsed JSON or rejects with an Error
// whose message is meant to be shown to the person as it is.

function query(params) {
  if (!params) return "";
  const pairs = Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== "");
  return pairs.length ? "?" + new URLSearchParams(pairs).toString() : "";
}

async function request(method, path, body, { blob = false } = {}) {
  const init = { method, headers: {}, credentials: "same-origin" };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new Error("Dreambot is not answering. Is it still running on this Mac?");
  }
  if (response.status === 401) {
    // The session cookie is gone (cleared, or Dreambot was reinstalled). The page itself hands
    // out a new one, so load it again — once, in case cookies are blocked altogether.
    if (!sessionStorage.getItem("dreambot-reloaded")) {
      sessionStorage.setItem("dreambot-reloaded", "1");
      location.reload();
    }
    throw new Error("This page lost its session. Reload it.");
  }
  sessionStorage.removeItem("dreambot-reloaded");
  const json = (response.headers.get("content-type") || "").includes("json");
  if (!response.ok) {
    const answer = json ? await response.json() : null;
    throw new Error((answer && answer.error) || `That did not work (${response.status}).`);
  }
  return blob ? response.blob() : response.json();
}

export const api = {
  get: (path, params) => request("GET", path + query(params)),
  post: (path, body = {}) => request("POST", path, body),
  put: (path, body = {}) => request("PUT", path, body),
  patch: (path, body = {}) => request("PATCH", path, body),
  del: (path) => request("DELETE", path),
  blob: (path, body = {}) => request("POST", path, body, { blob: true }),
  upload(files) {
    const form = new FormData();
    for (const file of files) form.append("file", file, file.name || "picture.png");
    return request("POST", "/api/uploads", form);
  },
};

/** The address of a picture at a thumbnail width the server makes (256, 512 or 1024). */
export const thumb = (url, width = 512) => `${url}${url.includes("?") ? "&" : "?"}w=${width}`;
