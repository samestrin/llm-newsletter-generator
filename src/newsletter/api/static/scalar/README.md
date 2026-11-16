# Vendored Scalar API reference

`standalone.js` is the published browser bundle of `@scalar/api-reference` **1.78.0**,
copied here so `/api/docs` loads nothing from another origin at run time. It is MIT
licensed, like this service. It is the same pinned version and the same bytes as
llm-pdf-ocr-api's `src/llm_pdf_ocr/web/scalar/standalone.js`.

- Pinned version: `@scalar/api-reference@1.78.0`
- Downloaded from: `https://cdn.jsdelivr.net/npm/@scalar/api-reference@1.78.0/dist/browser/standalone.js`
- SHA-256: `954b54c4e9e38abe6db9bbd504acd76aae3be473117a1b63a87b34a0298c629b`

The page that loads it is `../docs.html`, served by `GET /api/docs` in `../../app.py`;
the bundle itself is served at `GET /api/docs/scalar.js`. The configuration in the page
turns off Scalar's request proxy, its web fonts, its telemetry and its "open in client"
button, so the reference works with no network at all.

The bundle ends with `//# sourceMappingURL=standalone.js.map`. That map is not vendored,
so a browser with DevTools open asks for `/api/docs/scalar.js.map` and gets a 404 from
this origin. It is not an outside request and the page does not need it. The file is
left byte for byte as published rather than edited to drop that line.

To move to a newer version: download the same URL with the new version number, replace
`standalone.js`, and update the version and the SHA-256 above. Do not edit the bundle.
`tests/test_api_reference.py` checks the digest and that the page names no other origin.
