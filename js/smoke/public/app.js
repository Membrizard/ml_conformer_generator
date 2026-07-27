const form = document.getElementById("form");
const runBtn = document.getElementById("run");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");
const cards = document.getElementById("cards");
const meta = document.getElementById("meta");

let RDKit = null;
let modelsReady = false;
const viewers3d = [];

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = kind ? `status ${kind}` : "status";
}

function refreshStatus() {
  if (!modelsReady) return;
  const bits = ["Models ready"];
  bits.push(RDKit ? `RDKit ${RDKit.version()}` : "RDKit loading…");
  bits.push(window.$3Dmol ? "3Dmol ready" : "3Dmol missing");
  setStatus(bits.join(" · "), RDKit && window.$3Dmol ? "ok" : "");
}

async function initRdkit() {
  if (typeof initRDKitModule !== "function") {
    throw new Error("RDKit script failed to load (check network / CDN).");
  }
  RDKit = await initRDKitModule({
    locateFile: () =>
      "https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.wasm",
  });
  refreshStatus();
}

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (!data.ready) {
      setStatus(`Models not ready: ${data.error || "unknown error"}`, "bad");
      runBtn.disabled = true;
      return;
    }
    modelsReady = true;
    refreshStatus();
  } catch (err) {
    setStatus(`Status check failed: ${err.message}`, "bad");
  }
}

/** Render a Molfile / SDF block to SVG via RDKit.js. */
function molfileToSvg(molfile, width = 280, height = 220) {
  if (!RDKit) return `<p class="draw-error">RDKit not loaded</p>`;

  const block = String(molfile).replace(/\$\$\$\$\s*$/, "").trim();
  // Generated samples often fail RDKit sanitization (valence / aromaticity).
  let mol = RDKit.get_mol(block, JSON.stringify({ sanitize: false }));
  if (!mol) mol = RDKit.get_mol(block);
  if (!mol) {
    return `<p class="draw-error">RDKit could not parse Molfile</p>`;
  }

  try {
    if (typeof mol.set_new_coords === "function") {
      mol.set_new_coords(true);
    } else if (typeof mol.get_new_coords === "function") {
      mol.get_new_coords(true);
    }

    const details = JSON.stringify({
      width,
      height,
      bondLineWidth: 1.2,
      addStereoAnnotation: false,
    });
    const svg =
      typeof mol.get_svg_with_highlights === "function"
        ? mol.get_svg_with_highlights(details)
        : mol.get_svg(width, height);
    return svg || `<p class="draw-error">Empty SVG from RDKit</p>`;
  } catch (err) {
    return `<p class="draw-error">Draw failed: ${err.message}</p>`;
  } finally {
    mol.delete();
  }
}

function asSdf(molfile) {
  const block = String(molfile).replace(/\$\$\$\$\s*$/, "").trimEnd();
  return `${block}\n$$$$\n`;
}

/** Mount an interactive 3Dmol.js viewer into `element`. */
function mount3dViewer(element, molfile) {
  if (!window.$3Dmol) {
    element.innerHTML = `<p class="draw-error">3Dmol.js not loaded</p>`;
    return null;
  }

  element.innerHTML = "";
  try {
    const viewer = $3Dmol.createViewer(element, {
      backgroundColor: "#ffffff",
      antialias: true,
    });
    viewer.addModel(asSdf(molfile), "sdf");
    viewer.setStyle(
      {},
      {
        stick: { radius: 0.14, colorscheme: "default" },
        sphere: { scale: 0.22, colorscheme: "default" },
      },
    );
    viewer.zoomTo();
    viewer.render();
    // Ensure canvas matches layout after the card is visible.
    requestAnimationFrame(() => {
      viewer.resize();
      viewer.render();
    });
    return viewer;
  } catch (err) {
    element.innerHTML = `<p class="draw-error">3D view failed: ${err.message}</p>`;
    return null;
  }
}

function clearViewers() {
  for (const viewer of viewers3d) {
    try {
      viewer.clear();
    } catch {
      // ignore
    }
  }
  viewers3d.length = 0;
}

function renderMoleculeCard(mol) {
  const card = document.createElement("article");
  card.className = "card";
  card.innerHTML = `
    <div class="card-head">
      <h3>Sample ${mol.index + 1}</h3>
      <p class="stats">${mol.nAtoms} atoms · ${mol.nBonds} bonds</p>
    </div>
    <div class="views">
      <figure class="view">
        <figcaption>2D · RDKit</figcaption>
        <div class="drawing drawing-2d"></div>
      </figure>
      <figure class="view">
        <figcaption>3D · 3Dmol <span class="hint">drag to rotate</span></figcaption>
        <div class="drawing drawing-3d"></div>
      </figure>
    </div>
    <details>
      <summary>Molfile V2000</summary>
      <pre></pre>
    </details>
  `;
  card.querySelector(".drawing-2d").innerHTML = molfileToSvg(mol.molfile);
  card.querySelector("pre").textContent = mol.molfile;

  const viewerEl = card.querySelector(".drawing-3d");
  const viewer = mount3dViewer(viewerEl, mol.molfile);
  if (viewer) viewers3d.push(viewer);

  return card;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  runBtn.disabled = true;
  runBtn.textContent = "Generating…";
  results.hidden = true;
  clearViewers();
  cards.innerHTML = "";

  const body = {
    referenceContext: [
      Number(form.c0.value),
      Number(form.c1.value),
      Number(form.c2.value),
    ],
    nAtoms: Number(form.nAtoms.value),
    nSamples: Number(form.nSamples.value),
    variance: Number(form.variance.value),
    diffusionSteps: Number(form.diffusionSteps.value),
    keepLargestFragment: form.keepLargestFragment.checked,
    filterInvalid: form.filterInvalid.checked,
  };

  try {
    if (!RDKit) await initRdkit();

    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

    const validBit =
      data.nRequested != null
        ? `${data.nValid}/${data.nRequested} valid · `
        : "";
    meta.textContent = `${validBit}${data.molecules.length} shown · ${(data.elapsedMs / 1000).toFixed(1)}s`;
    for (const mol of data.molecules) {
      cards.appendChild(renderMoleculeCard(mol));
    }
    results.hidden = false;
    // Second resize pass once the results section is shown.
    requestAnimationFrame(() => {
      for (const viewer of viewers3d) {
        try {
          viewer.resize();
          viewer.render();
        } catch {
          // ignore
        }
      }
    });
    refreshStatus();
  } catch (err) {
    setStatus(`Generation failed: ${err.message}`, "bad");
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Generate";
  }
});

await Promise.all([
  loadStatus(),
  initRdkit().catch((err) => {
    setStatus(`RDKit failed to load: ${err.message}`, "bad");
  }),
]);
refreshStatus();
