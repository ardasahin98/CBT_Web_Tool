"""cbt_core.py - CDSS test metrics and Cyclic Behavior Type (CBT) estimation.

Sahin et al. - Cyclic Behavior Type Assessment Conditioned on Laboratory Test
Metrics for Fine-grained Soils.  Only the metrics used by the CBT models are
computed (r_u,max, Ratio_hull, Ratio_dgamma, kappa_gamma).  Data_Smooth,
NumberOfCycles, ru_calc, Ns_calculation and three_point_curvature are copied
verbatim from the tested ngl_def_v2.py.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

def Data_Smooth(Raw_Data):

    # Creates a copy of the raw data as smooth data
    Smooth_Data = {"Test_Type": np.copy(Raw_Data["Test_Type"]),
                   "Time" : np.copy(Raw_Data["Time"]), 
                   "Stress" : [], 
                   "Cell_Pressure" : [],
                   "Volumetric_Strain" : np.copy(Raw_Data["Volumetric_Strain"]), 
                   "Pore_Pressure" : np.copy(Raw_Data["Pore_Pressure"]),
                   "Normal_Stress" : [], 
                   "Strain" : [], 
                   "New_Time" : [],
                   "Effective_Stress": np.copy(Raw_Data["Effective_Stress"])}
    
    Test_Type = Smooth_Data["Test_Type"][0]
    if Test_Type == "DSS":
        Stress = np.copy(Raw_Data["Shear_Stress"])
        Strain = np.copy(Raw_Data["Shear_Strain"])
        Normal_stress = np.copy(Raw_Data["Normal_Stress"])
        Smooth_Data["Normal_Stress"] = Normal_stress
    else: 
        Stress = np.copy(Raw_Data["Deviator_Stress"])
        Strain = np.copy(Raw_Data["Axial_Strain"])
        Cell_pressure = np.copy(Raw_Data["Cell_Pressure"])
        Smooth_Data["Cell_Pressure"] = Cell_pressure

    # Assigns the Stress - Strain values to the Smooth Data just in case
    Smooth_Data["Stress"] = Stress
    Smooth_Data["Strain"] = Strain

    # Assign Cell and Pore Pressure values
    Pore_Pressure = Smooth_Data["Pore_Pressure"]
    Cell_Pressure = Smooth_Data["Cell_Pressure"]

    # Takes time from the raw data
    Time = np.copy(Raw_Data["Time"])
    Num = len(NumberOfCycles(Smooth_Data))-1
    if Num <= 0:

        Smooth_Data["Stress"] = Stress
        Smooth_Data["Strain"] = Strain
        Smooth_Data["New_Time"] = Time
        return Smooth_Data
    
    else:
        try:
            
            # Creates a time space with maximum time value in the raw data
            new_time = np.linspace(0, np.max(Time), Num*1000)

            # Make monotonic cubic spline
            if Test_Type == "TX":
                try:
                    # For TX cubic spline is all for Stress Strain Cell and Pore Pressures
                    pi_strain = PchipInterpolator(Time, Strain)
                    pi_stress = PchipInterpolator(Time, Stress)
                    pi_pore = PchipInterpolator(Time, Pore_Pressure)
                    pi_cell = PchipInterpolator(Time, Cell_Pressure)
                    New_strain = pi_strain(new_time)
                    New_stress = pi_stress(new_time)
                    New_pore = pi_pore(new_time)
                    New_cell = pi_cell(new_time)
                except ValueError:
                    Time_new = np.linspace(0, np.max(Time), len(Time))
                    pi_strain = PchipInterpolator(Time_new, Strain)
                    pi_stress = PchipInterpolator(Time_new, Stress)
                    pi_pore = PchipInterpolator(Time_new, Pore_Pressure)
                    pi_cell = PchipInterpolator(Time_new, Cell_Pressure)
                    New_strain = pi_strain(new_time)
                    New_stress = pi_stress(new_time)
                    New_pore = pi_pore(new_time)
                    New_cell = pi_cell(new_time)
                Smooth_Data["Pore_Pressure"] = New_pore
                Smooth_Data["Cell_Pressure"] = New_cell
            # This part is for DSS
            else:
                try:
                    # For DSS only Stress and Strain is connected with Cubic Spline
                    pi_strain = PchipInterpolator(Time, Strain)
                    pi_stress = PchipInterpolator(Time, Stress)
                    pi_norm_stress = PchipInterpolator(Time, Normal_stress)
                    New_strain = pi_strain(new_time)
                    New_stress = pi_stress(new_time)
                    New_norm_stress = pi_norm_stress(new_time)
                except ValueError:
                    Time_new = np.linspace(0, np.max(Time), len(Time))
                    pi_strain = PchipInterpolator(Time_new, Strain)
                    pi_stress = PchipInterpolator(Time_new, Stress)
                    pi_norm_stress = PchipInterpolator(Time_new, Normal_stress)
                    New_strain = pi_strain(new_time)
                    New_stress = pi_stress(new_time)
                    New_norm_stress = pi_norm_stress(new_time)
                Smooth_Data["Normal_Stress"] = New_norm_stress
            # Assigns new interpolated values to the Smooth_Data dictionary
            Smooth_Data["Stress"] = New_stress
            Smooth_Data["Strain"] = New_strain    
            Smooth_Data["New_Time"] = new_time
        except Exception as e:
            print("Interpolation failed:", e)
            Smooth_Data["Stress"] = Stress
            Smooth_Data["Strain"] = Strain   

    return Smooth_Data

def NumberOfCycles(Data):

    Cycle_Bounds = [0]
    Instance_Num = 0
    # Determines the stress type based on test type 
    Stress = Data["Stress"]
    max_stress = np.max(np.abs(Stress))
    
    try:
        for i in range(1, len(Stress)-1):
            
            # if Stress[i] > 0.0001 and Stress[i+1] < 0.0001:
            #     Instance_Num += 1
            #     continue

            if Stress[i] < 0 and Stress[i+1] > 0:
                Instance_Num += 1
                continue

            elif Instance_Num == 1:
                m_stress = np.abs(np.min(Stress[Cycle_Bounds[-1]:i]))
                
                threshold = 0.3
                if m_stress < threshold *max_stress:
                    Instance_Num = 0
                    continue
                else:
                    Cycle_Bounds.append(i)
                    
                    Instance_Num = 0
            

        last_value = Cycle_Bounds[-1]
        l1 = Cycle_Bounds[-2]
        last_portion_stress = Stress[last_value:]
        max_stress_last = np.max(last_portion_stress)
        min_stress_last = np.min(last_portion_stress)
        max_stress_gen = np.max(Stress[l1:last_value])
        min_stress_gen = np.min(Stress[l1:last_value])
        acceptance = 0.95

        if max_stress_last > max_stress_gen * acceptance:
            if min_stress_last < min_stress_gen * acceptance:
                Cycle_Bounds.append(len(Stress) - 1)

    except IndexError:
        Instance_Num = 1

    return Cycle_Bounds


def ru_calc(Data):

    # A dictionary is created to store the output data
    ru_calc = {"CSR": [], "ru_values" : [], "x_axis_values" : [], "y_axis_values" : [], "Limiting_N" : 0, "N/N_l": []}

    # This part considers the calculations for the triaxial test
    if Data["Test_Type"][0] == "TX":

        # Copy of each data from the smoothened data is taken for variables
        Pore_Pressure = np.copy(Data["Pore_Pressure"])
        Cell_Pressure = np.copy(Data["Cell_Pressure"])
        Effective_Stress  = np.copy(Data["Effective_Stress"])
        Deviator_Stress = np.copy(Data["Stress"])
        
        # There are 2 cases which are the pore pressure is total values with including back pressure
        # The second one are the effective pore pressures excluding the back pressure 
        if Effective_Stress.size == 0:
            Back_Pressure = Pore_Pressure[0]
            Effective_Pore_Pressure = np.copy(Pore_Pressure) - Back_Pressure
        else:
            Back_Pressure = Cell_Pressure[0] - Effective_Stress[0]
            Effective_Pore_Pressure = np.copy(Pore_Pressure)
        
        # Initial effective stress is calculated by using cell and back pressure values 
        Initial_Effective_Stress = Cell_Pressure[0] - Back_Pressure

        # Based on Initial effective stress and pore pressure ru value is calculated 
        #ru_calc["ru_values"] = Effective_Pore_Pressure/Initial_Effective_Stress

        # p value is calculated by using Effective cell pressure by subtracting back pressure
        # The equation is based on sigma_1 = sd + cell, sigma_2 = cell, sigma_3 = cell
        p = ((Cell_Pressure-Back_Pressure)*3+Deviator_Stress)/3

        # Effective pore pressure which is the excluded version of the back pressure is subtracted to calculate p_prime
        p_prime = p - Effective_Pore_Pressure
        
        # For the x axis of one plot p_prime/p_prime_c is calculated
        ru_calc["x_axis_values"] = p_prime/p_prime[0]

        # q value is calculated based on sigma_1 = sd + cell, sigma_3 = cell
        q = Deviator_Stress

        # For the x axis of one plot q/2p_prime_c is calculated
        ru_calc["y_axis_values"] =q/(2*p_prime[0])

        ru_calc["ru_values"] = (p_prime[0] - p_prime) / p_prime[0]

        # CSR value is assigned as q/p_prime
        ru_calc["CSR"] = q/(2*p_prime[0])

    # This part is for the Direct Simple Shear Test
    else:
        
        # Normal and Shear stress values are copied from the raw data instead of the smoothened data
        Normal_Stress = np.copy(Data["Normal_Stress"])
        try:
            Shear_Stress = np.copy(Data["Stress"])
        except:  # noqa: E722
            Shear_Stress = np.copy(Data["Shear_Stress"])

        # The insterted normal stress values are effective values
        Effective_Vertical_Stress = Normal_Stress

        # Pore pressure values are calculated based on the first value of the effective vertical stress which there is no pore pressure
        Pore_Pressure = Effective_Vertical_Stress[0] - Normal_Stress

        # Necessary calculations are performed similar to the definitions for each parameter explained in the triaxial part and assigned to the necessary values
        ru_calc["CSR"] = Shear_Stress/Effective_Vertical_Stress[0]
        ru_calc["ru_values"] = Pore_Pressure/Effective_Vertical_Stress[0]
        ru_calc["x_axis_values"] = Effective_Vertical_Stress/Effective_Vertical_Stress[0]
        ru_calc["y_axis_values"] = ru_calc["CSR"]

    return ru_calc

def Ns_calculation(Data, Num): 
    Strain = np.asarray(Data["Strain"])
    Time = np.asarray(Data["New_Time"])

    if Data["Test_Type"][0] == "TX":

        strain_list = np.zeros(len(Strain), dtype=float)

        Threshold_Strain = 5
        # Compute cumulative max and min
        cum_max = np.maximum.accumulate(Strain)
        cum_min = np.minimum.accumulate(Strain)
        strain_list = cum_max - cum_min

        N_s_index = np.argmin(np.abs(strain_list - Threshold_Strain))
        N_s = N_s_index/1000

        if N_s < Num:
            N_s_index = np.argmin(np.abs(strain_list - Threshold_Strain))
            N_s_time = Time[N_s_index]
            N_over_N_s = Time/N_s_time
            gama_over_gama_s = strain_list/Threshold_Strain
            N_s = N_s_index/1000
        else: 
            N_s = "Did not reached"
            N_over_N_s = "Did not reached"
            gama_over_gama_s = strain_list
        
    else: 
        
        strain_list = np.zeros(len(Strain), dtype=float)

        Threshold_Strain = 3
        # Compute cumulative max of absolute values
        cum_max_abs = np.maximum.accumulate(np.abs(Strain))
        cum_min_abs = np.minimum.accumulate(np.abs(Strain))
        strain_list = np.maximum(cum_max_abs, cum_min_abs)
        
        N_s_index = np.argmin(np.abs(strain_list - Threshold_Strain))
        N_s = N_s_index/1000

        if N_s < Num:
            N_s_index = np.argmin(np.abs(strain_list - Threshold_Strain))
            N_s_time = Time[N_s_index]
            N_over_N_s = Time/N_s_time
            gama_over_gama_s = strain_list/Threshold_Strain
            N_s = N_s_index/1000
        else: 
            N_s = "Did not reached"
            N_over_N_s = "Did not reached"
            gama_over_gama_s = strain_list

    N_s_dict = [gama_over_gama_s, N_over_N_s, N_s]

    return N_s_dict

def three_point_curvature(x_s, y_s):

    # Compute spacing
    h1 = x_s[1] - x_s[0]  # Distance between first and second point
    h2 = x_s[2] - x_s[1]  # Distance between second and third point

    # First derivative (slope at the middle point approximation)
    y_prime = (y_s[2] - y_s[0]) / (x_s[2] - x_s[0])

    # Second derivative using non-uniform finite difference
    y_double_prime = 2 * ((y_s[2] - y_s[1]) / h2 - (y_s[1] - y_s[0]) / h1) / (h1 + h2)

    # Compute curvature
    curvature = y_double_prime / (1 + y_prime ** 2) ** (3 / 2)

    return curvature


# =============================================================================
#  Web-tool layer (new).  Everything above this line is copied verbatim from
#  ngl_def_v2.py so the metrics are identical to those used in the paper.
# =============================================================================
from scipy.spatial import ConvexHull
from scipy.integrate import simpson

GAMMA_DA_TARGET = 9.0          # % double-amplitude shear strain (gamma_DA,9)
GAMMA_DA_MIN = 6.0             # % below this, CBT is not assessed
METRIC_KEYS = ["hull", "ru", "dgamma", "kappa"]

TEMPLATE_COLUMNS = ["Time (s)", "Shear Stress (kPa)", "Vertical Effective Stress (kPa)",
                    "Shear Strain (%)", "Vertical Strain (%)"]


# -----------------------------------------------------------------------------
#  Input
# -----------------------------------------------------------------------------
def read_template(text):
    """Parse the CDSS template CSV. Columns are used by position:
    time (s), shear stress (kPa), vertical effective stress (kPa),
    shear strain (%) and, optionally, vertical strain (%).
    A header row is required; lines starting with '#' are ignored.
    NGL export files (DSSS_TIME, DSSS_TAU, ...) have the same layout."""
    import csv
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ValueError("The file is empty.")
    rows = []
    for rec in csv.reader(lines[1:]):
        vals = []
        for v in rec[:5]:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(np.nan)
        if len(vals) >= 4 and np.all(np.isfinite(vals[:4])):
            rows.append(vals + [np.nan] * (5 - len(vals)))
    if len(rows) < 50:
        raise ValueError("Fewer than 50 valid data rows were found. The file must have at least "
                         "4 numeric columns: time, shear stress, vertical effective stress and shear strain.")
    arr = np.asarray(rows, float)
    t = arr[:, 0]
    if np.any(np.diff(t) < 0):
        raise ValueError("Time must be non-decreasing.")
    n = len(arr)
    vol = np.nan_to_num(arr[:, 4])
    return {"Test_Type": ["DSS"], "Time": t, "Shear_Stress": arr[:, 1],
            "Normal_Stress": arr[:, 2], "Shear_Strain": arr[:, 3],
            "Volumetric_Strain": vol, "Pore_Pressure": np.zeros(n),
            "New_Time": [], "Effective_Stress": np.zeros(n)}


# -----------------------------------------------------------------------------
#  Metrics
# -----------------------------------------------------------------------------
def hull_ratio(Data, l1, l2):
    """Ratio_hull = A_loop / A_hull (Metric_Calculator, cell 3)."""
    stress_ind = np.asarray(Data["Stress"][l1:l2])
    strain_ind = np.asarray(Data["Strain"][l1:l2])
    combined = np.stack((strain_ind, stress_ind), axis=1)
    hull = ConvexHull(combined)
    area_curve = simpson(stress_ind, x=strain_ind)
    area_hull = hull.volume
    return area_curve / area_hull, hull.vertices


def strain_curvature(Data, Num):
    """kappa_gamma over the whole record (Metric_Calculator, cell 0).
    Note: the original batch script passed the test index instead of the
    number of cycles to Ns_calculation; the number of cycles is used here."""
    N_s_dict = Ns_calculation(Data, Num)
    N_s = N_s_dict[2]
    if N_s != "Did not reached":
        gama_over_gama_s = N_s_dict[0]
        N_over_N_s = N_s_dict[1]
    else:
        gama_over_gama_s = N_s_dict[0]
        N_over_N_s = Data["New_Time"]
    max_N_over_N_s = np.max(N_over_N_s)
    idx = [int(np.argmin(np.abs(N_over_N_s - max_N_over_N_s * p))) for p in (0.1, 0.5, 0.9)]
    x_s = [float(N_over_N_s[i]) for i in idx]
    y_s = [float(gama_over_gama_s[i]) for i in idx]
    curvature = three_point_curvature(x_s, y_s)
    return {"kappa": float(curvature), "N_s": None if N_s == "Did not reached" else float(N_s),
            "x_s": x_s, "y_s": y_s,
            "N_over_N_s": np.asarray(N_over_N_s, float), "g_over_g_s": np.asarray(gama_over_gama_s, float),
            "reached": N_s != "Did not reached"}


def _f(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def analyze(raw):
    """Compute all per-cycle metrics for one CDSS record."""
    Data = Data_Smooth(raw)
    B = NumberOfCycles(Data)
    Num = len(B) - 1
    if Num < 2:
        raise ValueError("Fewer than two loading cycles were detected.")
    stress = np.asarray(Data["Stress"], float)
    strain = np.asarray(Data["Strain"], float)
    time = np.asarray(Data["New_Time"], float)
    ru_d = ru_calc(Data)
    ru = np.asarray(ru_d["ru_values"], float)
    csr = np.asarray(ru_d["CSR"], float)
    sv0 = float(Data["Normal_Stress"][0])

    curv = strain_curvature(Data, Num)
    first_cycle_max_strain = float(np.max(np.abs(strain[B[0]:B[1]])))

    cycles = []
    for i in range(1, Num + 1):
        l1, l2 = B[i - 1], B[i]
        s = strain[l1:l2]
        row = {"cycle": i, "l1": int(l1), "l2": int(l2),
               "gamma_SA": _f(np.max(np.abs(s))), "gamma_DA": _f(np.max(s) - np.min(s)),
               "CSR_cycle": _f(np.max(np.abs(csr[l1:l2]))),
               "ru": _f(np.max(ru[l1:l2]))}
        try:
            hr, verts = hull_ratio(Data, l1, l2)
            row["hull"] = _f(min(hr, 1.0))   # Ratio_hull limited to 1
            row["hull_idx"] = (np.asarray(verts) + l1).tolist()
        except Exception:
            row["hull"] = None
            row["hull_idx"] = []
        if i == 1:
            before = 0.0
        else:
            before = float(np.max(np.abs(strain[B[i - 2]:l1])))
        row["str_acc"] = _f(np.max(np.abs(s)) - before)
        row["dgamma"] = _f(row["str_acc"] / first_cycle_max_strain) if first_cycle_max_strain > 0 else None
        row["kappa"] = _f(curv["kappa"])
        cycles.append(row)

    target = next((c["cycle"] for c in cycles if c["gamma_DA"] is not None and c["gamma_DA"] >= GAMMA_DA_TARGET), None)
    gamma_da_max = max((c["gamma_DA"] or 0.0) for c in cycles)

    return {"n_cycles": Num, "sv0": sv0, "bounds": [int(b) for b in B],
            "target_cycle": target, "reached_target": target is not None,
            "gamma_DA_max": gamma_da_max, "assessable": gamma_da_max >= GAMMA_DA_MIN,
            "gamma_DA_min": GAMMA_DA_MIN,
            "first_cycle_max_strain": first_cycle_max_strain,
            "curvature": {k: curv[k] for k in ("kappa", "N_s", "x_s", "y_s", "reached")},
            "cycles": cycles,
            "_series": {"time": time, "stress": stress, "strain": strain, "ru": ru,
                        "sv_norm": np.asarray(ru_d["x_axis_values"], float),
                        "csr": csr, "NNs": curv["N_over_N_s"], "gg_s": curv["g_over_g_s"]}}


# -----------------------------------------------------------------------------
#  CBT prediction
# -----------------------------------------------------------------------------
_MODELS = {}


def load_models(json_text):
    import json
    m = json.loads(json_text)
    _MODELS.clear()
    for a, d in m["analysts"].items():
        _MODELS[a] = {"metrics": d["metrics"], "theta": np.asarray(d["theta"], float),
                      "sigma": np.asarray(d["sigma"], float)}
    return list(_MODELS.keys())


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(40)
_GH_W = _GH_W / _GH_W.sum()


def _cbt_stats(mu, sd):
    """Summary of CBT = expit(Y), Y ~ N(mu, sd)."""
    mean = float(np.sum(_GH_W * _expit(mu + sd * _GH_X)))
    return {"mu_T": float(mu), "sigma_T": float(sd), "median": float(_expit(mu)),
            "p16": float(_expit(mu - sd)), "p84": float(_expit(mu + sd)),
            "p05": float(_expit(mu - 1.645 * sd)), "p95": float(_expit(mu + 1.645 * sd)),
            "mean": mean}


def predict_cycle(metrics, analysts=None, weights=None):
    """Predictive CBT distribution for one set of metrics.

    For analyst i, every posterior sample s gives mu_s = theta_s . [1, X].
    The analyst's predictive logit-space distribution has
        mu_i      = mean_s(mu_s)
        sigma_i^2 = var_s(mu_s) + mean_s(sigma_hat_s^2)
    Analysts are combined with the logic-tree moments (Brandenberg et al. 2024):
        mu_c      = sum w_i mu_i
        sigma_c^2 = sum w_i sigma_i^2 + sum w_i (mu_i - mu_c)^2
    """
    analysts = analysts or list(_MODELS.keys())
    out = {"analysts": {}, "missing": []}
    mus, sds, ws = [], [], []
    for k, a in enumerate(analysts):
        mdl = _MODELS[a]
        x = [metrics.get(m) for m in mdl["metrics"]]
        if any(v is None for v in x):
            out["missing"].append(a)
            continue
        X = np.concatenate([[1.0], np.asarray(x, float)])
        mu_s = mdl["theta"] @ X
        mu_i = float(mu_s.mean())
        var_param = float(mu_s.var())
        var_model = float(np.mean(mdl["sigma"] ** 2))
        sd_i = float(np.sqrt(var_param + var_model))
        st = _cbt_stats(mu_i, sd_i)
        st["sd_param"] = float(np.sqrt(var_param))
        st["sd_model"] = float(np.sqrt(var_model))
        out["analysts"][a] = st
        mus.append(mu_i); sds.append(sd_i)
        ws.append(1.0 if weights is None else float(weights.get(a, 1.0)))
    if mus:
        w = np.asarray(ws) / np.sum(ws)
        mus = np.asarray(mus); sds = np.asarray(sds)
        mu_c = float(np.sum(w * mus))
        within = float(np.sum(w * sds ** 2))
        between = float(np.sum(w * (mus - mu_c) ** 2))
        st = _cbt_stats(mu_c, np.sqrt(within + between))
        st["sd_within"] = float(np.sqrt(within))
        st["sd_between"] = float(np.sqrt(between))
        out["combined"] = st
    else:
        out["combined"] = None
    return out


# -----------------------------------------------------------------------------
#  Entry point used by the web worker
# -----------------------------------------------------------------------------
_LAST = {}


def run(text, name="test", weights=None):
    raw = read_template(text)
    res = analyze(raw)
    for c in res["cycles"]:
        if res["assessable"]:
            m = {k: c.get(k) for k in METRIC_KEYS}
            c["cbt"] = predict_cycle(m, weights=weights)
        else:
            c["cbt"] = None
    _LAST[name] = res
    ser = res.pop("_series")
    res["name"] = name
    res["analyst_names"] = list(_MODELS.keys())
    res["analyst_metrics"] = {a: _MODELS[a]["metrics"] for a in _MODELS}
    return res, ser
