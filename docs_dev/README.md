#  Documentation

 The documentation server is built using [MkDocs](https://www.mkdocs.org/) with the
 [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

---

##  Development

To build and serve the documentation locally:

1. Install MkDocs with the Material theme:

   ```bash
   pip install mkdocs-material
   ```
2. Serve the documentation locally:

``` bash
mkdocs serve -a localhost:8000
```
This will start a local server (at http://localhost:8000) where you can preview your changes in real-time.

## Guidelines

- All documentation is written in Markdown (.md) files.

- Keep content concise, clear, and informative.

- Use consistent structure and formatting for readability.

- The documentation covers the two open-source libraries:

    - `src/mlconfgen` — Python library (`docs/python/`)

    - `js` — JavaScript / ONNX Runtime package (`docs/javascript/`)

    - shared pages: installation and weights (`docs/getting_started/`), model architecture and benchmarks (`docs/model/`)

- Navigation is defined explicitly in `mkdocs.yml`; add new pages there.

- Validate with `mkdocs build --strict` before publishing — it fails on broken internal links.

CI builds static HTML to the repository root **`../docs`**. `-d` is resolved relative to this directory (where `mkdocs.yml` lives), so use `../docs` — not `docs` (that would overwrite source `docs/`).

From the repo root:

```bash
pip install mkdocs-material
mkdocs build --strict -f docs_dev/mkdocs.yml -d ../docs
```

Or from this directory:

```bash
mkdocs build --strict -d ../docs
```

## Server Template

The pages of the production inference-server documentation this site was templated from
(Molecule Designer, Admin UI, server API, sample integrations and their screenshots) are kept
in `./_server_template` for reference and are not part of the build.