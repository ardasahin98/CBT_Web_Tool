// Web worker: runs the Python metric code (cbt_core.py) with Pyodide.
// All computation happens in the user's browser; no data is uploaded.

const PYODIDE_BASE = new URL("../vendor/pyodide/", self.location).href;
const ROOT = new URL("../", self.location).href;

let pyodide = null;

function post(type, payload = {}) { self.postMessage({ type, ...payload }); }

async function init() {
  post("progress", { step: "Loading Python runtime", pct: 5 });
  importScripts(PYODIDE_BASE + "pyodide.js");
  pyodide = await loadPyodide({ indexURL: PYODIDE_BASE });
  post("progress", { step: "Loading NumPy and SciPy", pct: 40 });
  await pyodide.loadPackage(["numpy", "scipy"]);
  post("progress", { step: "Loading metric code and model coefficients", pct: 85 });
  const [code, models] = await Promise.all([
    fetch(ROOT + "py/cbt_core.py", { cache: "no-cache" }).then(r => r.text()),
    fetch(ROOT + "data/models.json").then(r => r.text()),
  ]);
  pyodide.FS.writeFile("/home/pyodide/cbt_core.py", code);
  pyodide.globals.set("_models_json", models);
  pyodide.runPython(`
import sys, json, math
import numpy as np
sys.path.insert(0, "/home/pyodide")
import cbt_core
_analysts = cbt_core.load_models(_models_json)
del _models_json

def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return o if math.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o

def _run_for_js(txt, nm):
    res, ser = cbt_core.run(txt, nm)
    keys = list(ser.keys())
    buf = np.stack([np.asarray(ser[k], dtype=np.float32) for k in keys]).tobytes()
    return json.dumps(_clean(res)), keys, buf
`);
  post("ready", {});
}

async function runTest(id, name, text) {
  try {
    const fn = pyodide.globals.get("_run_for_js");
    const out = fn(text, name);
    const [json, keysPy, bufPy] = out.toJs({ depth: 1 });
    const keys = keysPy.toJs ? keysPy.toJs() : Array.from(keysPy);
    const bytes = bufPy.toJs ? bufPy.toJs() : bufPy;
    const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    [out, fn, keysPy, bufPy].forEach(p => p && p.destroy && p.destroy());
    self.postMessage({ type: "result", id, json, keys, buffer: ab }, [ab]);
  } catch (err) {
    let msg = String(err && err.message ? err.message : err);
    const m = msg.match(/(ValueError|Exception|IndexError|ZeroDivisionError|TypeError): (.*)$/m);
    if (m) msg = m[2];
    post("error", { id, message: msg });
  }
}

self.onmessage = async (e) => {
  const d = e.data;
  if (d.type === "init") {
    try { await init(); } catch (err) { post("fatal", { message: String(err) }); }
  } else if (d.type === "run") {
    await runTest(d.id, d.name, d.text);
  }
};
