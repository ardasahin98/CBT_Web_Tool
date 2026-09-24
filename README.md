# CBT Tool (web)

Static website that computes CDSS test metrics and estimates cyclic behavior type (CBT)
for every loading cycle, following Sahin et al. (in prep.). The Python metric code runs
in the browser with Pyodide. There is no server, and uploaded files never leave the user's computer.

## Folder layout

| Path | Contents |
|---|---|
| `index.html`, `css/`, `js/app.js` | Tool page (Plotly charts) |
| `about.html` | Input format, method and citation page |
| `js/worker.js` | Web worker that starts Pyodide and calls the Python code |
| `py/cbt_core.py` | Metric and CBT code. The top half is copied verbatim from `ngl_def_v2.py` (`Data_Smooth` … `ru_calc`, `Ns_calculation`, `three_point_curvature`); the bottom half adds hull ratio, Ratio_Δγ, κγ, CBT prediction and the analyst combination |
| `data/models.json` | Posterior samples (4 chains × 1,000) of the final model for each of the 10 analysts |
| `template/`, `examples/` | Input template and three example tests |
| `vendor/` | Pyodide 0.29.3 (NumPy 2.2.5, SciPy 1.14.1) and Plotly 2.35.2, hosted locally so the site does not depend on a CDN |
| `tools/build_models.py` | Rebuilds `models.json` from `../Bayesian_Results` |

## Run locally

Browsers block web workers on `file://` pages, so serve the folder:

```
cd web
python -m http.server 8000
```

Then open http://localhost:8000.

## Publish (GitHub Pages, free)

1. Create a GitHub repository (for example `cbt-tool`) and push the contents of `web/` to it.
2. In the repository, open Settings → Pages and choose "Deploy from a branch", branch `main`, folder `/ (root)`.
3. The site will be at `https://<user>.github.io/cbt-tool/`. `.nojekyll` is already included.

Any static host works too: Netlify, Cloudflare Pages, or a UCLA web server.

## Implementation notes

- Units: time (s, min, h) and stress (kPa, MPa, psf, psi, atm) are labels only; all metrics use ratios. Shear strain entered as a decimal is multiplied by 100 in `read_template` (`strain_scale`), and results are reported in percent.

- CBT for analyst *i*: every posterior sample gives μ_s = θ_s·[1, X]. Then μ_i = mean(μ_s) and
  σ_i² = var(μ_s) + mean(σ̂_s²). Analysts are combined with equal weights using
  μ_c = Σ w μ_i and σ_c² = Σ w σ_i² + Σ w (μ_i − μ_c)². The result is back-transformed with expit.
- Default evaluation cycle is the last cycle; the γ_DA = 9% cycle is marked. Tests that stop between 6% and 9% are flagged. Below 6%, no CBT is reported ("CBT for fine-grained soils cannot be assessed").
- κγ is computed once for the whole record. The batch script `Metric_Calculator.ipynb` called
  `Ns_calculation(Data, data_num)` (the test index); the tool passes the number of cycles instead.
- Posterior means and SDs in `models.json` reproduce Table 2 of the paper exactly.
