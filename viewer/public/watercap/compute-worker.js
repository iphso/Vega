// Web worker for the watercap page's "Compute" button: runs the campaign's own
// resolved_coarse tier (scripts/watercap_browser.py) under Pyodide. The Python
// sources come from data/watercap/pybundle.json (built by watercap_browser.py --bundle).
//   in:  {type: 'init'} | {type: 'compute', design, row}
//   out: {type: 'ready', seconds} | {type: 'log', line} | {type: 'result', result} | {type: 'error', error}
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const HOME = '/home/pyodide';

async function setup(py, bundle) {
  for (const [path, src] of Object.entries(bundle.files)) {
    const full = `${HOME}/${path}`;
    py.FS.mkdirTree(full.slice(0, full.lastIndexOf('/')));
    py.FS.writeFile(full, src);
  }
  py.runPython(`
import os, sys
os.environ["STONELAB_SRC"] = "${HOME}/StoneLab/src"
sys.path[:0] = ["${HOME}/vega/scripts", "${HOME}/StoneLab/src"]
import warnings; warnings.filterwarnings("ignore")
import watercap_browser, watercap_resolved, export_watercap_viewer_data
`);
}

function run(py, design, row) {
  const compute = py.globals.get('watercap_browser').compute;
  return JSON.parse(compute(JSON.stringify(design), JSON.stringify(row || {})));
}

if (typeof self !== 'undefined' && typeof importScripts === 'function') {
  let py = null;
  const post = (m) => self.postMessage(m);
  self.onmessage = async (ev) => {
    const m = ev.data;
    try {
      if (m.type === 'init') {
        const t0 = performance.now();
        importScripts(PYODIDE + 'pyodide.js');
        post({ type: 'log', line: '@stage load' });
        py = await loadPyodide({ indexURL: PYODIDE });
        py.setStdout({ batched: (line) => post({ type: 'log', line }) });
        await py.loadPackage(['numpy', 'scipy']);
        const bundle = await (await fetch(m.bundleUrl)).json();
        await setup(py, bundle);
        py.runPython('import watercap_browser');
        post({ type: 'ready', seconds: (performance.now() - t0) / 1e3, built: bundle.built });
      } else if (m.type === 'compute') {
        post({ type: 'result', id: m.id, result: run(py, m.design, m.row) });
      }
    } catch (e) {
      post({ type: 'error', id: m.id, error: String(e && e.message || e) });
    }
  };
} else if (typeof module !== 'undefined') {
  module.exports = { setup, run };
}
