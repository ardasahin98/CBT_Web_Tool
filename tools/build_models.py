"""Rebuild data/models.json from the PyMC posterior samples in ../Bayesian_Results.

Run from the web/ folder:  python tools/build_models.py
Edit FINAL_MODELS if an analyst's retained metric set changes. Coefficient
columns in the .npy files follow the metric order in the file name; they are
re-ordered here to [intercept, hull, ru, dgamma, kappa] (metrics present only).
"""
import json, os
import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "Bayesian_Results")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "models.json")
FINAL_MODELS = {
    "AJ": "AJ_ru_max_cycle_Ratio", "AS": "AS_Ratio_Str_Acc_Curvature_ru_max_cycle",
    "AWS": "AWS_ru_max_cycle_Ratio", "JPS": "JPS_ru_max_cycle_Ratio_Delta_S_FC_S",
    "KJU": "KJU_ru_max_cycle_Ratio_Delta_S_FC_S", "KOC": "KOC_ru_max_cycle_Str_Acc_Curvature_Ratio",
    "SJB": "SJB_ru_max_cycle_Str_Acc_Curvature_Ratio", "SLK": "SLK_ru_max_cycle_Ratio",
    "TME": "TME_ru_max_cycle_Str_Acc_Curvature_Ratio", "VR": "VR_ru_max_cycle_Ratio_Delta_S_FC_S",
}
TOKENS = {"ru_max_cycle": "ru", "Str_Acc_Curvature": "kappa", "Delta_S_FC_S": "dgamma", "Ratio": "hull"}
CANON = ["hull", "ru", "dgamma", "kappa"]

out = {}
for a, f in FINAL_MODELS.items():
    rest, order = f[len(a) + 1:], []
    while rest:
        t = next(t for t in sorted(TOKENS, key=len, reverse=True) if rest.startswith(t))
        order.append(TOKENS[t]); rest = rest[len(t):].lstrip("_")
    b = np.load(os.path.join(SRC, f + "_beta.npy")).reshape(-1, len(order) + 1)
    s = np.load(os.path.join(SRC, f + "_sigma.npy")).reshape(-1)
    metrics = [m for m in CANON if m in order]
    B = b[:, [0] + [1 + order.index(m) for m in metrics]]
    out[a] = {"metrics": metrics, "theta": np.round(B, 5).tolist(), "sigma": np.round(s, 5).tolist()}
    print(a, metrics, "mean", np.round(B.mean(0), 3), "sd", np.round(B.std(0), 3), "sigma", round(float(s.mean()), 3))
json.dump({"coef_order": "[intercept]+metrics", "n_samples": int(len(s)), "analysts": out},
          open(OUT, "w"), separators=(",", ":"))
print("wrote", OUT)
