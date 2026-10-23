// Resolved tier worker: the campaign's own Python (Vega scripts/watercap_browser.workbench ->
// watercap_resolved.simulate -> stonelab spray_cfd / ehd / ehd_particles) under Pyodide, with the
// grid solves on wasm-compute-kit's kernels.wasm (stonelab.gridsolve -> wck). Pyodide's files are
// served next to the page (pyodide/), so the first run downloads ~37 MB and later runs reuse it.
//   in:  {type: "run", id, design, aux, preset, seed, base, bundle?}   (base: the page's absolute
//        directory URL -- this worker runs from a blob: URL, so relative paths would not resolve)
//   out: {type: "progress", id, stage: "load" | "spray" | "corona", ...} | {type: "done", id, result} | {type: "error", id, message}
let py = null, job = 0;

// ---- wckPool: kernel calls side by side (wck's Wasm backend .map), one kernels.wasm instance per
// helper worker. Python blocks in Atomics.wait while the helpers run, so results come back through
// a SharedArrayBuffer -- which needs a cross-origin isolated page (COOP + COEP); without one the
// pool stays off and wck runs the same chunks one after another (same results).
const HELPER = `
let ex = null;
onmessage = e => {
  const m = e.data;
  if (m.type === "init") { ex = new WebAssembly.Instance(new WebAssembly.Module(m.bytes), {}).exports; postMessage({ type: "ready" }); return; }
  const ctl = new Int32Array(m.sab, 0, 4), allocs = [];
  try {
    const conv = m.args.map(a => {
      if (a.n !== undefined) return a.n;
      const nb = Math.max(a.a.byteLength, 8), p = ex.wck_alloc(nb, 16) >>> 0;
      if (!p) throw new Error("wck pool: alloc of " + nb + " bytes");
      allocs.push([p, nb]);
      new Uint8Array(ex.memory.buffer, p, a.a.byteLength).set(new Uint8Array(a.a.buffer, a.a.byteOffset, a.a.byteLength));
      return p;
    });
    const r = ex[m.name](...conv);
    new Float64Array(m.sab, m.retOff, 1)[0] = Number(r);
    let k = 0;
    m.args.forEach((a, i) => {
      if (!a.out) return;
      new Uint8Array(m.sab, m.outOffs[k++], a.a.byteLength).set(new Uint8Array(ex.memory.buffer, conv[i], a.a.byteLength));
    });
  } catch (err) {
    Atomics.store(ctl, 1, 1);
    console.error("wck pool job", m.name, err);
  } finally {
    for (const [p, nb] of allocs) ex.wck_free(p, nb, 16);
    Atomics.add(ctl, 0, 1);
    Atomics.notify(ctl, 0);
  }
};`;

const wckPool = {
  ready: false, size: 0, workers: [],
  /** Start n helpers on the kernels (bytes); false (pool off) without cross-origin isolation. */
  async init(bytes, n) {
    if (typeof SharedArrayBuffer === "undefined" || !self.crossOriginIsolated || n < 2) return false;
    const url = URL.createObjectURL(new Blob([HELPER], { type: "text/javascript" }));
    this.workers = Array.from({ length: n }, () => new Worker(url));
    await Promise.all(this.workers.map(w => new Promise((res, rej) => {
      w.onmessage = e => { if (e.data.type === "ready") res(); };
      w.onerror = rej;
      w.postMessage({ type: "init", bytes });
    })));
    this.size = n; this.ready = true;
    return true;
  },
  /** jobs: per call, its arguments ({n: number} | {a: TypedArray, out: bool}); blocks until all are done. */
  run(name, jobs) {
    let off = 16;
    const outOffs = [], retOffs = [];
    for (const job of jobs) {
      const oo = [];
      for (const a of job) if (a.out) { oo.push(off); off += (a.a.byteLength + 7) & ~7; }
      outOffs.push(oo); retOffs.push(off); off += 8;
    }
    const sab = new SharedArrayBuffer(off), ctl = new Int32Array(sab, 0, 4);
    jobs.forEach((job, j) => this.workers[j % this.size].postMessage({ name, args: job, sab, outOffs: outOffs[j], retOff: retOffs[j] }));
    const t0 = performance.now();
    for (;;) {
      const done = Atomics.load(ctl, 0);
      if (done >= jobs.length) break;
      if (performance.now() - t0 > 600e3) throw new Error("wck pool: jobs did not finish in 10 min");
      Atomics.wait(ctl, 0, done, 1000);
    }
    if (Atomics.load(ctl, 1)) throw new Error("wck pool: a job failed (see the console)");
    return {
      outs: jobs.map((job, j) => { let k = 0; return job.filter(a => a.out).map(a => new a.a.constructor(sab, outOffs[j][k++], a.a.length)); }),
      rets: retOffs.map(o => new Float64Array(sab, o, 1)[0]),
    };
  },
};
self.wckPool = wckPool;
const HOME = "/home/pyodide";
const post = m => self.postMessage(m);

async function boot(base, bundle, id) {
  const t0 = performance.now();
  const say = line => post({ type: "progress", id, stage: "load", line, seconds: (performance.now() - t0) / 1e3 });
  say("loading Python (Pyodide 0.27.7)");
  importScripts(base + "pyodide/pyodide.js");
  const p = await loadPyodide({ indexURL: base + "pyodide/" });
  say("loading numpy and scipy");
  await p.loadPackage(["numpy", "scipy"]);
  for (const [path, src] of Object.entries(bundle.files)) {
    const full = `${HOME}/${path}`;
    p.FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
    p.FS.writeFile(full, src);
  }
  p.runPython(`
import sys
sys.path[:0] = ["${HOME}/vega/scripts", "${HOME}/StoneLab/src", "${HOME}/wasm-compute-kit/python"]
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import watercap_browser`);
  p.setStdout({ batched: line => progress(line) });
  const b64 = p.runPython("from wck import _kernels_wasm as _k\n_k.KERNELS_WASM_BASE64");
  const n = Math.min(navigator.hardwareConcurrency || 4, 8);
  const pooled = await wckPool.init(Uint8Array.from(atob(b64), c => c.charCodeAt(0)), n);
  say(`Python ready in ${((performance.now() - t0) / 1e3).toFixed(1)} s; ` +
      (pooled ? `parcels run on ${n} threads` : "one thread (this page is not cross-origin isolated)"));
  return p;
}

// the Python's verbose lines -> the page's progress messages
function progress(line) {
  let m = /^\s*it (\d+): w_out=(\S+) flow_res=(\S+)/.exec(line);
  if (m) { post({ type: "progress", id: job, stage: "spray", spray: { outer: +m[1], wOut: +m[2], flowRes: +m[3] } }); return; }
  m = /^\s*period (\d+): I=(\S+) uA.*collected so far (\S+),/.exec(line);
  if (m) post({ type: "progress", id: job, stage: "corona", period: +m[1], current: +m[2] * 1e-6, collected: +m[3] });
}

self.onmessage = async e => {
  const m = e.data;
  if (m.type !== "run") return;
  job = m.id;
  try {
    py ??= await boot(m.base, m.bundle, m.id);
    const out = py.globals.get("watercap_browser").workbench(JSON.stringify(m.design), JSON.stringify(m.aux), m.preset || "resolved_coarse", m.seed || 0);
    const r = JSON.parse(out);
    if (!r.ok) throw new Error(r.error);
    post({ type: "done", id: m.id, result: r });
  } catch (err) {
    post({ type: "error", id: m.id, message: err instanceof Error ? err.message : String(err) });
  }
};
