# Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/) and the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

Markdown sources live in `docs/`. The published static site is committed at the
repository root as **`../docs`** and served by GitHub Pages (**Settings → Pages →
Deploy from a branch → `/docs`**). An empty `.nojekyll` file disables GitHub’s
Jekyll step so MkDocs HTML is served as-is.

---

## Development

1. Install MkDocs with the Material theme:

   ```bash
   pip install mkdocs-material
   ```

2. Live preview:

   ```bash
   mkdocs serve -a localhost:8000
   ```

## Publish (update the committed site)

After changing sources under `docs/`, rebuild into the repo-root `docs/` folder and
commit those files (CI only verifies the build; it does not push Pages):

```bash
# from repository root — note ../docs (relative to docs_dev/, not "docs")
pip install mkdocs-material
mkdocs build --strict -f docs_dev/mkdocs.yml -d ../docs
```

Or from this directory:

```bash
mkdocs build --strict -d ../docs
```

Then commit the updated `docs/` tree (including `.nojekyll`).

## Guidelines

- All documentation is written in Markdown (`.md`) files.
- Keep content concise, clear, and informative.
- Use consistent structure and formatting for readability.
- The documentation covers the two open-source libraries:
  - `src/mlconfgen` — Python library (`docs/python/`)
  - `js` — JavaScript / ONNX Runtime package (`docs/javascript/`)
  - shared pages: installation and weights (`docs/getting_started/`), model architecture and benchmarks (`docs/model/`)
- Navigation is defined explicitly in `mkdocs.yml`; add new pages there.
- Validate with `mkdocs build --strict` before publishing — it fails on broken internal links.

## Server Template

The pages of the production inference-server documentation this site was templated from
(Molecule Designer, Admin UI, server API, sample integrations and their screenshots) are kept
in `./_server_template` for reference and are not part of the build.
