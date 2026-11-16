# Vendored libraries

The web app has no build step, so the one library it uses is kept here as shipped.

| File | What | Version | License |
| --- | --- | --- | --- |
| `preact-htm.js` | [htm](https://github.com/developit/htm)'s `preact/standalone` module: [Preact](https://preactjs.com), its hooks and htm in one file | htm 3.1.1 (bundles Preact 10) | htm: Apache-2.0 (`LICENSE-htm.txt`); Preact: MIT (`LICENSE-preact.txt`) |

To update it: `npm pack htm`, and copy `package/preact/standalone.module.js` over `preact-htm.js`.
