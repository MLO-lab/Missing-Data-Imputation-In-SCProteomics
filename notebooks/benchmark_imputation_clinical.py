# --- Cell 1: cap CPU thread/process usage, imports, environment setup, and the shared random seed ---------
import os


def _cluster_cpu_budget(default=4):
    """How many CPU threads/workers this job should actually use.

    Priority: an explicit IMPUTATION_CPU_BUDGET override (for interactive use
    on an unmanaged, possibly-shared node- cpu , or the gpu
    prototyping nodes - where Slurm sets none of the variables below) beats
    the Slurm allocation (when this is an actual sbatch job on gpu-unlimited),
    which beats this process's own CPU affinity, which beats a conservative
    default. Also returns where the number came from, for the printed message
    below."""
    override = os.environ.get("IMPUTATION_CPU_BUDGET")
    if override:
        try:
            return max(1, int(override)), "IMPUTATION_CPU_BUDGET override"
        except ValueError:
            print(f"IMPUTATION_CPU_BUDGET={override!r} isn't a valid integer - ignoring it.")

    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val:
            try:
                return max(1, int(val)), var
            except ValueError:
                pass
    val = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if val:
        import re
        m = re.match(r"(\d+)", val)
        if m:
            return max(1, int(m.group(1))), "SLURM_JOB_CPUS_PER_NODE"

    try:
        return max(1, len(os.sched_getaffinity(0))), "CPU affinity (no Slurm allocation, no override detected)"
    except AttributeError:
        return max(1, os.cpu_count() or default), "os.cpu_count() (no Slurm allocation, no override detected)"


# Must run before numpy/scipy/sklearn/torch/rpy2 are imported below - each of
# these reads its thread count once, at import/initialization time, so setting
# the env vars any later has no effect. Left uncapped, they each default to
# using every core visible on the node, not just the ones Slurm actually gave
# this job - on a shared node (a GPU node especially, where other jobs' own
# CPU-side work depends on not being starved) that over-subscription is
# exactly the problem this avoids.
N_JOBS, _cpu_budget_source = _cluster_cpu_budget()
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = str(N_JOBS)
print(f"CPU budget for this session: {N_JOBS} threads/workers (source: {_cpu_budget_source})")
if _cpu_budget_source.startswith(("CPU affinity", "os.cpu_count")):
    print(
        "  no Slurm allocation detected - this looks like an unmanaged node "
        "(cpu, or a gpu prototyping session). If sharing it "
        "with others, set IMPUTATION_CPU_BUDGET to a smaller number "
        "before starting this kernel, e.g. `export IMPUTATION_CPU_BUDGET=16`."
    )


def _split_cpu_budget(n_jobs, max_workers_cap=None):
    """Splits the CPU budget above into (n_scenario_workers, threads_per_worker).
    """
    if n_jobs <= 1:
        return 1, 1
    if n_jobs <= 8:
        return n_jobs, 1
    n_workers = max(2, n_jobs // 3)
    if max_workers_cap:
        n_workers = min(n_workers, max_workers_cap)
    threads_per_worker = max(1, n_jobs // n_workers)
    return n_workers, threads_per_worker


N_SCENARIO_WORKERS, THREADS_PER_WORKER = _split_cpu_budget(N_JOBS)
print(f"Scenario-level parallelism: {N_SCENARIO_WORKERS} concurrent scenario worker(s) x "
      f"{THREADS_PER_WORKER} thread(s) each (budget: {N_JOBS}). Override N_SCENARIO_WORKERS / "
      f"THREADS_PER_WORKER directly (after this cell runs, before Cell 16) to change the split.")

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mudata as mu
from scipy.stats import norm, truncnorm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, Ridge

mu.set_options(pull_on_update=False)
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)

RNG_SEED = 0

# R/rpy2 guard (BPCA, DreamAI, ADMIN): try to import these once up front;
# if the R side isn't available, those three methods are simply skipped
# rather than treated as a hard failure. R picks up the same OMP_NUM_THREADS /
# OPENBLAS_NUM_THREADS env vars set above (as long as it's linked against
# OpenBLAS or MKL), since those are set before rpy2 initializes R below.
RUN_BPCA, RUN_DREAMAI = True, True
BPCA_AVAILABLE, DREAMAI_AVAILABLE = False, False
if RUN_BPCA:
    try:
        import rpy2.rinterface as ri
        from rpy2.robjects.packages import importr
        if not ri.embedded.isinitialized():
            ri.initr()
        importr("pcaMethods")
        BPCA_AVAILABLE = True
        print("R + pcaMethods available: BPCA enabled.")
    except Exception as exc:
        print(f"BPCA disabled ({type(exc).__name__}: {exc}). Requires R + Bioconductor 'pcaMethods' + rpy2.")
if RUN_DREAMAI:
    try:
        import rpy2.rinterface as ri
        from rpy2.robjects.packages import importr
        if not ri.embedded.isinitialized():
            ri.initr()
        importr("DreamAI")
        DREAMAI_AVAILABLE = True
        print("R + DreamAI available: DreamAI/ADMIN enabled.")
    except Exception as exc:
        print(f"DreamAI/ADMIN disabled ({type(exc).__name__}: {exc}). Requires R + the DreamAI package + rpy2.")

# PyTorch guard (DAE, VAE): same idea - skip gracefully if torch isn't installed
RUN_DEEP_LEARNING = True
DEEP_LEARNING_AVAILABLE = False
if RUN_DEEP_LEARNING:
    try:
        import torch
        import torch.nn as nn
        torch.set_num_threads(N_JOBS)  # PyTorch has its own thread pool, separate from the BLAS env vars above
        DEEP_LEARNING_AVAILABLE = True
        print("PyTorch available: DAE/VAE enabled.")
    except Exception as exc:
        print(f"DAE/VAE disabled ({type(exc).__name__}: {exc}). Requires PyTorch.")


# --- Cell 2: locate and load the processed data ---------
try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()

import os

BASE_DIR = THIS_DIR.parent  # the project root, e.g. /home/rnjue/Imputationv2

CANDIDATE_PROCESSED_DIRS = []
if os.environ.get("IMPUTATION_OUTPUT_DIR"):
    CANDIDATE_PROCESSED_DIRS.append(Path(os.environ["IMPUTATION_OUTPUT_DIR"]).expanduser())
CANDIDATE_PROCESSED_DIRS += [
    BASE_DIR / "data" / "processed",
    Path.home() / "reuben_imputation" / "processed",
    Path("/projects/datasets_BIO/reuben_imputation/processed"),
    THIS_DIR / "processed",
]
PROCESSED_DIR = next((d for d in CANDIDATE_PROCESSED_DIRS if d.exists()), None)
if PROCESSED_DIR is None:
    raise FileNotFoundError(
        "no 'processed' directory found among: " + ", ".join(str(d) for d in CANDIDATE_PROCESSED_DIRS) +
        ". Run read_multiomic_Data_clinical.ipynb through Cell 14 first, or set IMPUTATION_OUTPUT_DIR."
    )


try:
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    probe = RESULTS_DIR / ".write_test"
    probe.touch(); probe.unlink()
except (PermissionError, OSError):
    RESULTS_DIR = Path.home() / "reuben_imputation" / "imputation_results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print("processed dir:", PROCESSED_DIR, "| results dir:", RESULTS_DIR)

mdata = mu.read_h5mu(PROCESSED_DIR / "dlbcl_mudata_filtered.h5mu")
crc_df = pd.read_csv(PROCESSED_DIR / "crc_proteomics_samples_x_features.csv", index_col=0)


def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# --- Cell 3: build the three benchmark matrices (features x samples, per point 1) ---
rna_adata = mdata.mod["1_rna"]
assert (rna_adata.obs_names == mdata.obs_names).all()
profiled_mask = mdata.obs["rna_profiled"].to_numpy()

rna_full = pd.DataFrame(
    to_dense(rna_adata.X)[profiled_mask].T, index=rna_adata.var_names, columns=rna_adata.obs_names[profiled_mask]
)
prot_full = pd.DataFrame(to_dense(mdata.mod["2_prot"].X).T, index=mdata.mod["2_prot"].var_names, columns=mdata.obs_names)
crc_full = crc_df.T  # transpose back to features x samples, matching this benchmark's convention (see point 1)

N_TOP_FEATURES = 500  # this can be raised...I fix to 500 for time budget 


def top_variable_features(df, n):
    if df.shape[0] <= n:
        return df
    top_idx = df.var(axis=1, skipna=True).sort_values(ascending=False).index[:n]
    return df.loc[df.index.isin(top_idx)]


MATRICES = {
    "dlbcl_rna": top_variable_features(rna_full, N_TOP_FEATURES),
    "dlbcl_protein": top_variable_features(prot_full, N_TOP_FEATURES),
    "crc_protein": top_variable_features(crc_full, N_TOP_FEATURES),
}

# Scale check (point 2 above): report aggregate stats only, never raw values.
for name, df in MATRICES.items():
    v = df.values
    print(
        f"{name}: {df.shape} (features x samples) | "
        f"min={np.nanmin(v):.2f} max={np.nanmax(v):.2f} mean={np.nanmean(v):.2f} | "
        f"{100 * np.isnan(v).mean():.1f}% missing"
    )


# --- Cell 3b: calibrate the MNAR detection-limit curve from each matrix's own
# REAL missingness, before any artificial masking is applied. Use the probit model,
# Phi(a + b*Y), fit by maximum likelihood against each matrix's own real
# observed/missing pattern (Karpievitch et al. 2009; Lazar et al. 2016). Y is
# raw log2 abundance, not a standardized z-score, so a and b live on the scale
# the data itself is already on.
from scipy.stats import norm as _mnar_norm
from scipy.optimize import minimize as _mnar_minimize


def fit_mnar_probit_params(X, min_features=20):
    """Calibrate Phi(a + b*Y) from a matrix's own real (pre-masking)
    missingness. Y is a feature's own observed-mean log2 abundance, used as
    the abundance proxy since a missing cell's true value is unknown by
    definition. Falls back to (a=0, b=0) - an uninformative, MCAR-like
    weighting - when there isn't enough real missingness to fit from.
    """
    vals = X.values.astype(float)
    n_features, n_samples = vals.shape
    observed = np.isfinite(vals)
    missing = ~observed

    feat_mean = np.full(n_features, np.nan)
    has_obs = observed.any(axis=1)
    feat_mean[has_obs] = np.nanmean(np.where(observed, vals, np.nan)[has_obs], axis=1)

    usable_rows = has_obs
    n_usable = usable_rows.sum()
    n_with_any_missing = (missing[usable_rows].any(axis=1)).sum()

    diagnostics = {
        "n_features_total": n_features, "n_features_usable": int(n_usable),
        "n_features_with_missing": int(n_with_any_missing),
    }

    if n_usable < min_features or n_with_any_missing < 5:
        diagnostics["fit_status"] = "insufficient_data_fallback_to_mcar_like"
        return 0.0, 0.0, diagnostics

    x = np.repeat(feat_mean[usable_rows], n_samples)
    y = missing[usable_rows].ravel().astype(float)

    x_mean, x_std = x.mean(), x.std()
    if x_std == 0:
        diagnostics["fit_status"] = "zero_variance_abundance_fallback_to_mcar_like"
        return 0.0, 0.0, diagnostics
    xs = (x - x_mean) / x_std

    def neg_log_lik(params):
        a_s, b_s = params
        p = _mnar_norm.cdf(a_s + b_s * xs)
        eps = 1e-9
        p = np.clip(p, eps, 1 - eps)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    p0 = y.mean()
    a0 = _mnar_norm.ppf(np.clip(p0, 1e-3, 1 - 1e-3))
    res = _mnar_minimize(
        neg_log_lik, x0=[a0, 0.0], method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
    )
    if not res.success:
        diagnostics["fit_status"] = f"optimizer_failed_fallback_to_mcar_like ({res.message})"
        return 0.0, 0.0, diagnostics

    a_s, b_s = res.x
    b = b_s / x_std
    a = a_s - b_s * x_mean / x_std

    pred_p = _mnar_norm.cdf(a + b * x)
    ll_fit = -neg_log_lik(res.x)
    if 0 < p0 < 1:
        ll_null = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
        pseudo_r2 = float(1 - ll_fit / ll_null) if ll_null != 0 else float("nan")
    else:
        pseudo_r2 = float("nan")
    diagnostics.update({
        "fit_status": "ok", "a": float(a), "b": float(b),
        "empirical_missing_rate": float(y.mean()),
        "predicted_missing_rate": float(pred_p.mean()),
        "pseudo_r2_mcfadden": pseudo_r2,
    })
    return float(a), float(b), diagnostics


MNAR_PROBIT_PARAMS = {}
print("\nCalibrating MNAR detection-limit curve Phi(a + b*Y) per matrix, from each matrix's own real missingness:")
for name, df in MATRICES.items():
    a_fit, b_fit, diag = fit_mnar_probit_params(df)
    MNAR_PROBIT_PARAMS[name] = (a_fit, b_fit)
    extra = (
        f" | empirical missing rate={diag['empirical_missing_rate']:.3f} "
        f"predicted={diag['predicted_missing_rate']:.3f} "
        f"pseudo-R2={diag['pseudo_r2_mcfadden']:.3f}"
    ) if diag["fit_status"] == "ok" else ""
    print(
        f"  {name}: a={a_fit:.4f} b={b_fit:.4f} | {diag['fit_status']} | "
        f"features usable={diag['n_features_usable']}/{diag['n_features_total']} "
        f"(with missing: {diag['n_features_with_missing']})" + extra
    )
print(
    "\nNote: a negative b means lower-abundance values are more likely to be missing "
    "(a real detection-limit pattern). b close to 0 means abundance carries little or no "
    "information about missingness for that matrix (e.g. structural missingness, like "
    "dlbcl_rna's un-profiled samples), and the mask then falls back to an essentially MCAR "
    "weighting for that matrix, which is the correct behavior for that data, not a bug."
)


# --- Cell 4: generating masks (MCAR / MNAR / mixed) -----------
def generate_mcar_mask(X, masking_fraction=0.30, rng=None, exclude_mask=None):
    if rng is None:
        rng = np.random.default_rng()
    eligible = X.notna().values
    if exclude_mask is not None:
        eligible = eligible & ~exclude_mask.values
    positions = np.argwhere(eligible)
    n_to_mask = int(round(masking_fraction * len(positions)))
    if n_to_mask == 0:
        return pd.DataFrame(False, index=X.index, columns=X.columns)
    selected = rng.choice(len(positions), size=n_to_mask, replace=False)
    sel = positions[selected]
    mask = np.zeros(X.shape, dtype=bool)
    mask[sel[:, 0], sel[:, 1]] = True
    return pd.DataFrame(mask, index=X.index, columns=X.columns)


def generate_mnar_weighted_positions(X, n_to_mask, a, b, rng=None, exclude_mask=None):
    if rng is None:
        rng = np.random.default_rng()
    eligible = X.notna().values
    if exclude_mask is not None:
        eligible = eligible & ~exclude_mask.values
    positions = np.argwhere(eligible)
    if n_to_mask == 0:
        return np.empty((0, 2), dtype=int)
    values = X.values[eligible]  # raw log2 abundance, this entry's own real value
    weights = _mnar_norm.cdf(a + b * values)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        weights = np.ones_like(weights)
        total = weights.sum()
    weights = weights / total
    selected = rng.choice(len(positions), size=n_to_mask, replace=False, p=weights)
    return positions[selected]


def generate_mnar_mask(X, masking_fraction=0.30, a=0.0, b=0.0, rng=None, exclude_mask=None):
    if rng is None:
        rng = np.random.default_rng()
    eligible = X.notna().values
    if exclude_mask is not None:
        eligible = eligible & ~exclude_mask.values
    n_to_mask = int(round(masking_fraction * eligible.sum()))
    sel_pos = generate_mnar_weighted_positions(X, n_to_mask, a, b, rng=rng, exclude_mask=exclude_mask)
    mask = np.zeros(X.shape, dtype=bool)
    if len(sel_pos) > 0:
        mask[sel_pos[:, 0], sel_pos[:, 1]] = True
    return pd.DataFrame(mask, index=X.index, columns=X.columns)


def generate_mixed_mask(X, masking_fraction=0.30, mnar_fraction=0.50, a=0.0, b=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    eligible = X.notna().values
    n_eligible = eligible.sum()
    total_to_mask = int(round(masking_fraction * n_eligible))
    n_mnar = int(round(total_to_mask * mnar_fraction))
    n_mcar = total_to_mask - n_mnar

    mnar_positions = generate_mnar_weighted_positions(X, n_mnar, a, b, rng=rng)
    mnar_excl = np.zeros(X.shape, dtype=bool)
    if len(mnar_positions) > 0:
        mnar_excl[mnar_positions[:, 0], mnar_positions[:, 1]] = True
    mnar_excl_df = pd.DataFrame(mnar_excl, index=X.index, columns=X.columns)

    mcar_positions = np.empty((0, 2), dtype=int)
    if n_mcar > 0:
        available = np.argwhere(eligible & ~mnar_excl_df.values)
        n_mcar_clamped = min(n_mcar, len(available))
        mcar_positions = available[rng.choice(len(available), size=n_mcar_clamped, replace=False)]

    mask = np.zeros(X.shape, dtype=bool)
    mechanism = np.full(X.shape, "", dtype=object)
    if len(mnar_positions) > 0:
        mask[mnar_positions[:, 0], mnar_positions[:, 1]] = True
        mechanism[mnar_positions[:, 0], mnar_positions[:, 1]] = "MNAR"
    if len(mcar_positions) > 0:
        mask[mcar_positions[:, 0], mcar_positions[:, 1]] = True
        mechanism[mcar_positions[:, 0], mcar_positions[:, 1]] = "MCAR"
    return (
        pd.DataFrame(mask, index=X.index, columns=X.columns),
        pd.DataFrame(mechanism, index=X.index, columns=X.columns),
    )


def build_masked_scenario(row, X_truth, a=0.0, b=0.0):
    rng = np.random.default_rng(int(row["Seed"]))
    mask, mechanism = generate_mixed_mask(
        X_truth, masking_fraction=float(row["Masking_Fraction"]), mnar_fraction=float(row["MNAR_Fraction"]),
        a=a, b=b, rng=rng,
    )
    return X_truth.mask(mask), mask, mechanism


# --- Cell 5: metrics for scoring imputation accuracy -------
def calculate_metrics(truth, prediction, mask):
    y_true_mat, y_pred_mat, mask_mat = truth.values, prediction.values, mask.values
    y_true, y_pred = y_true_mat[mask_mat], y_pred_mat[mask_mat]
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    errors = y_true - y_pred
    rmse = np.sqrt(np.mean(errors ** 2)) if len(errors) else np.nan
    mae = np.mean(np.abs(errors)) if len(errors) else np.nan
    pearson_r = (np.corrcoef(y_true, y_pred)[0, 1]
                 if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
                 else np.nan)  # guards the same zero-variance case cor_vals already checks below

    nrmse_vals, cor_vals = [], []
    min_masked_per_feature = 5
    for i in range(y_true_mat.shape[0]):
        row_mask = mask_mat[i]
        if row_mask.sum() < min_masked_per_feature:
            continue
        yt, yp = y_true_mat[i, row_mask], y_pred_mat[i, row_mask]
        v = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[v], yp[v]
        if len(yt) < min_masked_per_feature:
            continue
        feat_range = np.nanmax(y_true_mat[i]) - np.nanmin(y_true_mat[i])
        if not np.isfinite(feat_range) or feat_range <= 0:
            continue
        row_rmse = np.sqrt(np.mean((yt - yp) ** 2))
        nrmse_vals.append(row_rmse / feat_range)
        if len(yt) > 1 and np.std(yt) > 0 and np.std(yp) > 0:
            cor_vals.append(np.corrcoef(yt, yp)[0, 1])

    return {
        "RMSE": rmse, "MAE": mae, "Pearson_r": pearson_r,
        "NRMSE": float(np.mean(nrmse_vals)) if nrmse_vals else np.nan,
        "Cor_proteinwise": float(np.mean(cor_vals)) if cor_vals else np.nan,
        "N_Evaluated": len(y_true), "N_Features_NRMSE": len(nrmse_vals),
    }


# --- Cell 6: shared helpers, baseline substitution, and KNN/SVD/SoftImpute -----

def mean_impute(Y):
    row_means = Y.mean(axis=1, skipna=True)
    return Y.T.fillna(row_means).T


def median_impute(Y):
    row_medians = Y.median(axis=1, skipna=True)
    return Y.T.fillna(row_medians).T


def mindet_impute(Y, q=0.01):
    """Deterministic left-censored fill, in the style of Perseus/DEP's 'MinDet':
    each sample's missing entries are filled with that same sample's own
    q-th quantile of observed values, standing in for a per-sample
    detection floor."""
    return Y.fillna(Y.quantile(q, axis=0))


def minprob_impute(Y, q=0.01, sigma_scale=0.3, rng=None):
    """Probabilistic left-censored fill: missing entries are drawn from
    N(quantile_q(observed), sigma_scale * sd(observed)), with the mean and
    spread of that normal estimated separately for each sample."""
    if rng is None:
        rng = np.random.default_rng()
    result = Y.copy()
    for col in result.columns:
        observed = result[col].dropna()
        if observed.empty:
            continue
        missing_mask = result[col].isna()
        n_missing = int(missing_mask.sum())
        if n_missing == 0:
            continue
        mu = observed.quantile(q)
        sigma = max(observed.std() * sigma_scale, 1e-6)
        result.loc[missing_mask, col] = rng.normal(loc=mu, scale=sigma, size=n_missing)
    return result


def knn_impute(Y, n_neighbors=10):
    """Distance-weighted KNN across samples. Each feature is standardized
    first so that no single feature's variance ends up dominating the
    distance metric."""
    row_mean = Y.mean(axis=1, skipna=True)
    row_std = Y.std(axis=1, skipna=True).replace(0, 1.0)
    Z = Y.sub(row_mean, axis=0).div(row_std, axis=0)

    fully_missing_rows = Z.index[Z.isna().all(axis=1)]
    Z_usable = Z.drop(index=fully_missing_rows)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    Z_filled_arr = imputer.fit_transform(Z_usable.T).T
    Z_filled = pd.DataFrame(Z_filled_arr, index=Z_usable.index, columns=Y.columns)
    if len(fully_missing_rows) > 0:
        Z_filled = Z_filled.reindex(Y.index)
    return Z_filled.mul(row_std, axis=0).add(row_mean, axis=0)


def _initial_fill(Y, center=False):
    """Initial row-mean imputation used to seed SVD/softImpute iterations.

    center=True additionally subtracts each feature's own row mean before
    returning, and returns that vector of row means as a fourth element so
    the caller can add it back after reconstruction. Features differ
    substantially in average log2 abundance, and on the raw (uncentered)
    scale that abundance-baseline difference dominates the matrix's singular
    value spectrum - svd_impute's fixed-rank truncation tolerates this (it
    always keeps a fixed *count* of components), but softimpute's shrinkage
    threshold is set as a fraction of the leading singular value, so on the
    uncentered matrix it ends up larger than nearly every other singular
    value in the spectrum and collapses the reconstruction to effective rank
    1, discarding almost all genuine covariation structure. Verified
    directly on this project's own matrices. 
    """
    row_means = Y.mean(axis=1, skipna=True)
    fully_missing_rows = row_means.isna().values
    grand_mean = row_means.mean()
    row_means_filled = row_means.fillna(grand_mean)
    filled = Y.T.fillna(row_means_filled).T.to_numpy(dtype=float, copy=True)
    if not center:
        return filled, Y.notna().values, fully_missing_rows
    row_means_arr = row_means_filled.to_numpy()
    centered = filled - row_means_arr[:, None]
    return centered, Y.notna().values, fully_missing_rows, row_means_arr


def _blank_out_unidentifiable_rows(X_filled, fully_missing_rows, index, columns):
    result = pd.DataFrame(X_filled, index=index, columns=columns)
    if fully_missing_rows.any():
        result.loc[fully_missing_rows] = np.nan
    return result


def svd_impute(Y, n_components=10, max_iter=100, tol=1e-5):
    """Iterative truncated-SVD ('hard-impute') low-rank reconstruction.

    Row-centered before decomposing (see `_initial_fill`'s docstring) so the
    singular value spectrum reflects genuine cross-feature covariation rather
    than being dominated by each feature's own average abundance; the row
    means are added back to the final reconstruction."""
    X_filled, observed, fully_missing_rows, row_means = _initial_fill(Y, center=True)
    for _ in range(max_iter):
        old = X_filled.copy()
        U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
        r = min(n_components, len(s))
        reconstruction = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
        X_filled[~observed] = reconstruction[~observed]
        denom = np.linalg.norm(old)
        if (np.linalg.norm(X_filled - old) / denom if denom > 0 else 0.0) < tol:
            break
    X_filled = X_filled + row_means[:, None]
    return _blank_out_unidentifiable_rows(X_filled, fully_missing_rows, Y.index, Y.columns)


def _soft_threshold_svd(M, lam):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ np.diag(np.maximum(s - lam, 0)) @ Vt


def softimpute(Y, shrink_frac=0.10, max_iter=200, tol=1e-5):
    """Nuclear-norm-regularized ('soft-impute', Mazumder et al., 2010) low-rank
    reconstruction. `lam` is calibrated as a fraction of the row-centered
    matrix's own leading singular value (see `_initial_fill`'s docstring) -
    on the uncentered scale that leading singular value is inflated by each
    feature's own average abundance rather than genuine covariation."""
    X_filled, observed, fully_missing_rows, row_means = _initial_fill(Y, center=True)
    Y_centered_vals = Y.values - row_means[:, None]
    lam = shrink_frac * np.linalg.svd(X_filled, compute_uv=False)[0]
    for _ in range(max_iter):
        old = X_filled.copy()
        reconstruction = _soft_threshold_svd(X_filled, lam)
        X_filled[~observed] = reconstruction[~observed]
        X_filled[observed] = Y_centered_vals[observed]
        denom = np.linalg.norm(old)
        if (np.linalg.norm(X_filled - old) / denom if denom > 0 else 0.0) < tol:
            break
    X_filled = X_filled + row_means[:, None]
    return _blank_out_unidentifiable_rows(X_filled, fully_missing_rows, Y.index, Y.columns), lam


# --- Cell 7: PPCA via EM on incomplete data --------
def ppca_impute(Y, n_components=10, max_iter=100, tol=1e-6, ridge_frac=1e-2):
    """Probabilistic PCA imputation via Roweis' (1998) EM algorithm.

    The E-/M-step per-sample and per-feature least-squares solves below were
    previously regularized by a fixed `np.eye(K) * 1e-6` - an absolute
    constant unrelated to the scale of the Gram matrix being inverted.
    Whenever a feature (or sample) has few observed entries relative to K
    (=10 by default), that Gram matrix is close to singular and 1e-6 is
    nowhere near enough regularization, producing individual reconstruction
    errors far outside the plausible range for this data and dominating the
    overall RMSE even though most predictions are reasonable. Scaling the
    ridge to a fraction of the local Gram matrix's own trace (standard
    Tikhonov regularization) fixes this.
    """
    P, N = Y.shape
    obs = Y.notna().values
    row_mean = Y.mean(axis=1, skipna=True).values
    row_mean = np.where(np.isnan(row_mean), np.nanmean(row_mean), row_mean)
    Yc = Y.values - row_mean[:, None]  # centered; NaNs remain NaN where missing

    K = n_components
    filled0 = np.where(obs, np.nan_to_num(Yc), 0.0)
    U, s, Vt = np.linalg.svd(filled0, full_matrices=False)
    C = U[:, :K] * s[:K]
    Z = Vt[:K, :]

    def _ridge_for(gram):
        return (ridge_frac * (np.trace(gram) / K if K > 0 else 1.0) + 1e-8) * np.eye(K)

    prev_recon = None
    for _ in range(max_iter):
        # E-step: per-sample least squares on observed features only
        Z = np.zeros((K, N))
        for n in range(N):
            o = obs[:, n]
            if not o.any():
                continue
            Co = C[o, :]
            gram = Co.T @ Co
            Z[:, n] = np.linalg.solve(gram + _ridge_for(gram), Co.T @ Yc[o, n])

        # M-step: per-feature least squares on observed samples only
        C_new = np.zeros((P, K))
        for p in range(P):
            o = obs[p, :]
            if not o.any():
                C_new[p, :] = np.nan  # flagged and blanked out below
                continue
            Zo = Z[:, o]
            gram = Zo @ Zo.T
            C_new[p, :] = np.linalg.solve(gram + _ridge_for(gram), Zo @ Yc[p, o])
        C = C_new

        recon = np.where(np.isnan(C).any(axis=1, keepdims=True), np.nan, C) @ Z
        if prev_recon is not None:
            finite = np.isfinite(recon) & np.isfinite(prev_recon)
            denom = np.linalg.norm(prev_recon[finite])
            if (np.linalg.norm(recon[finite] - prev_recon[finite]) / denom if denom > 0 else 0.0) < tol:
                prev_recon = recon
                break
        prev_recon = recon

    fully_missing_rows = np.isnan(C).any(axis=1)
    recon = np.nan_to_num(prev_recon) + row_mean[:, None]
    X_filled = np.where(obs, Y.values, recon)
    return _blank_out_unidentifiable_rows(X_filled, fully_missing_rows, Y.index, Y.columns)


# --- Cell 8: Regression / RandomForest / MissForest ---------
# All three methods predict each feature that has missing entries from a
# shared ~20-dimensional PCA summary of the current best-guess matrix,
# refitting that summary on every iteration.
def _low_dim_predictors(Y_filled, n_components=20):
    grand_mean = np.nanmean(Y_filled.values)
    Y_for_pca = Y_filled.fillna(grand_mean)
    return PCA(n_components=n_components, random_state=0).fit_transform(Y_for_pca.T.values)


def _regression_family_impute(Y, estimator_factory, n_components=20, max_iter=1, tol=1e-3, warm_start=None):
    Y_filled = (warm_start if warm_start is not None else svd_impute(Y, n_components=n_components)).copy()
    features_with_missing = Y.index[Y.isna().any(axis=1)]

    prev_filled = None
    for _ in range(max_iter):
        Z = _low_dim_predictors(Y_filled, n_components=n_components)
        new_filled = Y_filled.copy()
        for feat in features_with_missing:
            row_obs = Y.loc[feat].notna().values
            if row_obs.sum() < n_components + 2:
                continue
            X_missing = Z[~row_obs]
            if len(X_missing) == 0:
                continue
            model = estimator_factory()
            model.fit(Z[row_obs], Y.loc[feat].values[row_obs])
            new_filled.loc[feat, ~row_obs] = model.predict(X_missing)

        if prev_filled is not None:
            denom = np.linalg.norm(prev_filled.values)
            change = np.linalg.norm(new_filled.values - prev_filled.values) / denom if denom > 0 else 0.0
            Y_filled = new_filled
            if change < tol:
                break
        else:
            Y_filled = new_filled
        prev_filled = Y_filled
    return Y_filled


def regression_impute(Y, n_components=20, alpha=5.0):
    return _regression_family_impute(Y, lambda: Ridge(alpha=alpha), n_components=n_components, max_iter=1)


def random_forest_impute(Y, n_components=20, n_estimators=50, max_depth=8):
    return _regression_family_impute(
        Y, lambda: RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=N_JOBS),
        n_components=n_components, max_iter=1,
    )


def missforest_impute(Y, n_components=20, n_estimators=50, max_depth=8, max_iter=6, tol=1e-3):
    return _regression_family_impute(
        Y, lambda: RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=N_JOBS),
        n_components=n_components, max_iter=max_iter, tol=tol,
    )


# --- Cell 9: LLS, EM_MVN (MLE/EM), and MICE ---------------
def lls_impute(Y, k=30, ridge=1.0):
    result = Y.copy()
    values = Y.values
    filled, observed, _ = _initial_fill(Y)
    features_with_missing = [p for p in range(Y.shape[0]) if np.isnan(values[p]).any()]

    corr = Y.T.corr(min_periods=5).values.copy()
    corr = np.nan_to_num(corr, nan=-np.inf)
    np.fill_diagonal(corr, -np.inf)

    for p in features_with_missing:
        row_obs = observed[p]
        row_mis = ~row_obs
        if row_mis.sum() == 0:
            continue
        # Fallback to the same row-mean fill every other baseline-seeded method
        # (SVD, SoftImpute, ...) already uses via _initial_fill, instead of
        # leaving these entries unimputed (NaN), when there isn't enough
        # signal to trust a local regression: too few observed values for
        # this feature this scenario, or too few usable correlated
        # neighbors. calculate_metrics() silently drops any NaN prediction
        # from scoring - so leaving these NaN doesn't make LLS "fail
        # honestly" on its hardest cases, it makes LLS quietly not compete
        # on them at all while every other method still gets graded there.
        # Concretely, on dlbcl_protein (~34% real baseline missingness on
        # top of the artificial masking), these two conditions fired often
        # enough that LLS was scored on ~4-5% fewer features than every
        # other method, all in LLS's own favor, before it was fixed.
        if row_obs.sum() < 5:
            result.iloc[p, np.where(row_mis)[0]] = filled[p, row_mis]
            continue
        neighbor_order = np.argsort(-corr[p])
        neighbors = [n for n in neighbor_order if corr[p, n] > -np.inf and n != p][:k]
        if len(neighbors) < 3:
            result.iloc[p, np.where(row_mis)[0]] = filled[p, row_mis]
            continue
        A_train = filled[np.ix_(neighbors, np.where(row_obs)[0])].T
        b = values[p, row_obs]
        AtA = A_train.T @ A_train + ridge * np.eye(A_train.shape[1])
        coef = np.linalg.solve(AtA, A_train.T @ b)
        A_pred = filled[np.ix_(neighbors, np.where(row_mis)[0])].T
        result.iloc[p, np.where(row_mis)[0]] = A_pred @ coef
    return result


def em_mvn_impute(Y, max_iter=8, tol=1e-4, shrinkage=None):
    X = Y.T.values.copy()
    n, p = X.shape
    obs_mask = ~np.isnan(X)
    col_mean_init = np.nanmean(X, axis=0)
    fully_missing_cols = np.isnan(col_mean_init)
    col_mean_init = np.where(fully_missing_cols, np.nanmean(col_mean_init), col_mean_init)
    X_filled = np.where(obs_mask, X, col_mean_init[None, :])
    mu = X_filled.mean(axis=0)

    for _ in range(max_iter):
        centered = X_filled - mu
        if shrinkage is None:
            cov = LedoitWolf().fit(centered).covariance_
        else:
            emp = (centered.T @ centered) / n
            cov = (1 - shrinkage) * emp + shrinkage * np.eye(p) * np.trace(emp) / p
        cov_inv = np.linalg.pinv(cov)

        X_new = X_filled.copy()
        for i in range(n):
            miss = ~obs_mask[i]
            if not miss.any() or (~miss).sum() == 0:
                continue
            obs = ~miss
            Sigma_mm_inv = np.linalg.pinv(cov_inv[np.ix_(miss, miss)])
            beta = -Sigma_mm_inv @ cov_inv[np.ix_(miss, obs)]
            X_new[i, miss] = mu[miss] + beta @ (X_filled[i, obs] - mu[obs])

        change = np.linalg.norm(X_new - X_filled) / (np.linalg.norm(X_filled) + 1e-12)
        X_filled, mu = X_new, X_new.mean(axis=0)
        if change < tol:
            break

    result = pd.DataFrame(X_filled.T, index=Y.index, columns=Y.columns)
    result[Y.notna()] = Y
    fully_missing_rows = Y.isna().all(axis=1).values
    if fully_missing_rows.any():
        result.loc[fully_missing_rows] = np.nan
    return result


def mice_impute(Y, n_nearest_features=30, max_iter=10, random_state=0):
    # sklearn's IterativeImputer silently drops any fully-missing column
    # from its output array (documented behavior).
    # At the highest masking/MNAR corners of the grid, a protein that's
    # already about 34% missing in reality can end up wholly missing once
    # the extra MNAR-weighted mask is layered on, so completed.T ends up
    # with fewer rows than Y.index, and the DataFrame reconstruction below
    # used to raise "Shape of passed values is (N, M), indices imply
    # (500, M)". We apply the same seed-with-grand-mean fix used
    # throughout this file (dreamai_impute, svd_impute, ppca_impute, ...)
    # via the shared _blank_out_unidentifiable_rows helper: seed those
    # columns with just enough of a value that sklearn never sees a
    # fully-NaN column to drop, then blank that row back to NaN in the
    # final output.
    fully_missing_features = Y.isna().all(axis=1).values
    Y_seeded = Y.copy()
    if fully_missing_features.any():
        grand_mean = Y.values[Y.notna().values].mean()
        Y_seeded.loc[Y.index[fully_missing_features]] = Y_seeded.loc[Y.index[fully_missing_features]].fillna(grand_mean)

    X = Y_seeded.T.values
    imputer = IterativeImputer(
        estimator=BayesianRidge(), max_iter=max_iter, n_nearest_features=n_nearest_features,
        random_state=random_state, sample_posterior=False, skip_complete=True,
    )
    completed = imputer.fit_transform(X)
    result = pd.DataFrame(completed.T, index=Y.index, columns=Y.columns)
    result[Y.notna()] = Y
    return _blank_out_unidentifiable_rows(result.values, fully_missing_features, Y.index, Y.columns)


# --- Cell 10: MsImpute (data-driven-rank SoftImpute) and CF --------
def msimpute_impute(Y, candidate_ranks=(2, 4, 6, 8, 10, 15, 20), holdout_frac=0.10, max_iter=150, tol=1e-5, rng=None):
    """MsImpute-style data-driven-rank SoftImpute (see `softimpute`'s docstring in Cell 6 for
    why the row-centering below matters - the same rationale applies here, since this reuses
    `_soft_threshold_svd` and calibrates every candidate `lam` off `_initial_fill`'s leading
    singular value)."""
    if rng is None:
        rng = np.random.default_rng()
    observed = Y.notna().values
    obs_positions = np.argwhere(observed)
    n_holdout = min(max(50, int(holdout_frac * len(obs_positions))), len(obs_positions) - 1)
    holdout_idx = rng.choice(len(obs_positions), size=n_holdout, replace=False)
    holdout_pos = obs_positions[holdout_idx]

    cv_arr = Y.to_numpy(copy=True)
    cv_arr[holdout_pos[:, 0], holdout_pos[:, 1]] = np.nan
    Y_cv = pd.DataFrame(cv_arr, index=Y.index, columns=Y.columns)

    def _fit(Y_in, lam):
        X_filled, obs, fully_missing_rows, row_means = _initial_fill(Y_in, center=True)
        Y_in_centered_vals = Y_in.values - row_means[:, None]
        for _ in range(max_iter):
            old = X_filled.copy()
            recon = _soft_threshold_svd(X_filled, lam)
            X_filled[~obs] = recon[~obs]
            X_filled[obs] = Y_in_centered_vals[obs]
            denom = np.linalg.norm(old)
            if (np.linalg.norm(X_filled - old) / denom if denom > 0 else 0.0) < tol:
                break
        return X_filled + row_means[:, None], fully_missing_rows

    best_rank, best_err = None, np.inf
    for r in candidate_ranks:
        s_max_cv = np.linalg.svd(_initial_fill(Y_cv, center=True)[0], compute_uv=False)[0]
        X_filled, _ = _fit(Y_cv, (r / max(candidate_ranks)) * 0.15 * s_max_cv)
        pred = X_filled[holdout_pos[:, 0], holdout_pos[:, 1]]
        truth = Y.values[holdout_pos[:, 0], holdout_pos[:, 1]]
        err = np.sqrt(np.nanmean((pred - truth) ** 2))
        if err < best_err:
            best_err, best_rank = err, r

    s_max = np.linalg.svd(_initial_fill(Y, center=True)[0], compute_uv=False)[0]
    X_filled, fully_missing_rows = _fit(Y, (best_rank / max(candidate_ranks)) * 0.15 * s_max)
    return _blank_out_unidentifiable_rows(X_filled, fully_missing_rows, Y.index, Y.columns)


def cf_impute(Y, n_factors=10, n_epochs=300, lr=0.05, reg=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    P, C = Y.shape
    obs_p, obs_c = np.where(Y.notna().values)
    obs_v = Y.values[obs_p, obs_c]
    mu = obs_v.mean()

    feature_bias, sample_bias = np.zeros(P), np.zeros(C)
    feature_emb = rng.normal(scale=0.05, size=(P, n_factors))
    sample_emb = rng.normal(scale=0.05, size=(C, n_factors))
    n_obs = len(obs_v)

    for _ in range(n_epochs):
        pred = mu + feature_bias[obs_p] + sample_bias[obs_c] + np.sum(feature_emb[obs_p] * sample_emb[obs_c], axis=1)
        err = obs_v - pred

        grad_pb = np.zeros(P); np.add.at(grad_pb, obs_p, err)
        grad_cb = np.zeros(C); np.add.at(grad_cb, obs_c, err)
        feature_bias += lr * (grad_pb / n_obs - reg * feature_bias)
        sample_bias += lr * (grad_cb / n_obs - reg * sample_bias)

        grad_pe = np.zeros_like(feature_emb)
        grad_ce = np.zeros_like(sample_emb)
        np.add.at(grad_pe, obs_p, err[:, None] * sample_emb[obs_c])
        np.add.at(grad_ce, obs_c, err[:, None] * feature_emb[obs_p])
        feature_emb += lr * (grad_pe / n_obs - reg * feature_emb)
        sample_emb += lr * (grad_ce / n_obs - reg * sample_emb)

    recon = mu + feature_bias[:, None] + sample_bias[None, :] + feature_emb @ sample_emb.T
    X_filled = Y.values.copy()
    missing = Y.isna().values
    X_filled[missing] = recon[missing]
    fully_missing_rows = Y.isna().all(axis=1).values
    return _blank_out_unidentifiable_rows(X_filled, fully_missing_rows, Y.index, Y.columns)


# --- Cell 11: baseline-substitution / left-censored family (protein matrices only) ---
# Zero, HalfMin, LOD, QRILC.
def zero_impute(Y):
    return Y.fillna(0.0)


def half_min_impute(Y):
    row_half_min = Y.min(axis=1, skipna=True) / 2.0
    return Y.T.fillna(row_half_min).T


def lod_impute(Y, lod_value=None):
    """
    Fixed, global limit-of-detection substitution: every missing value is
    replaced with the same floor value, which defaults to this matrix's
    own smallest observed value.
    """
    if lod_value is None:
        lod_value = float(np.nanmin(Y.values))
    return Y.fillna(lod_value)


def qrilc_impute(Y, q_tail=0.30, rng=None):
    """
    Approximate QRILC: for each feature, fit a truncated normal
    distribution to the left tail of its observed values, then draw that
    feature's missing entries from the fitted distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    result = Y.copy()
    global_sd = np.nanstd(Y.values)
    for feat in result.index:
        observed = result.loc[feat].dropna()
        missing_mask = result.loc[feat].isna()
        n_missing = int(missing_mask.sum())
        if n_missing == 0 or observed.empty:
            continue
        sorted_obs = np.sort(observed.values)
        n_obs = len(sorted_obs)
        n_tail = max(3, int(np.ceil(q_tail * n_obs)))
        tail = sorted_obs[:n_tail]
        pp = (np.arange(1, n_tail + 1) - 0.5) / n_obs
        z = norm.ppf(pp)
        if n_tail >= 3 and np.std(z) > 0:
            slope, intercept = np.polyfit(z, tail, 1)
            sigma_hat, mu_hat = max(slope, 1e-3), intercept
        else:
            mu_hat, sigma_hat = tail.mean(), (global_sd if global_sd > 0 else 1e-3)
        a, b = -np.inf, (sorted_obs[0] - mu_hat) / sigma_hat
        result.loc[feat, missing_mask] = truncnorm.rvs(a, b, loc=mu_hat, scale=sigma_hat, size=n_missing, random_state=rng)
    return result


# --- Cell 12: BPCA and DreamAI/ADMIN, R/rpy2-backed ---------
def bpca_impute(Y, n_components=10):
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, pandas2ri
    from rpy2.robjects.conversion import localconverter

    # pcaMethods' checkData() rejects any matrix that has a fully-missing
    # row (a protein with zero observed values anywhere). At the highest
    # masking/MNAR corners of the grid, dlbcl_protein's real ~34% baseline
    # missingness plus the artificial mask can push some proteins into
    # that state, raising "pcaMethods checkData() failed." We apply the
    # same seed-with-grand-mean fix that dreamai_impute (right below)
    # already uses, via the shared _blank_out_unidentifiable_rows helper
    # used throughout this file.
    fully_missing_rows = Y.isna().all(axis=1).values
    Y_seeded = Y.copy()
    if fully_missing_rows.any():
        grand_mean = Y.values[Y.notna().values].mean()
        Y_seeded.loc[Y.index[fully_missing_rows]] = Y_seeded.loc[Y.index[fully_missing_rows]].fillna(grand_mean)

    X_for_r = Y_seeded.T.copy()
    X_for_r.index = X_for_r.index.astype(str)
    X_for_r.columns = X_for_r.columns.astype(str)
    with localconverter(default_converter + pandas2ri.converter):
        r_dataframe = ro.conversion.py2rpy(X_for_r)

    bpca_function = ro.r("""
        function(x, ncomp) {
            x <- as.matrix(x); x[is.nan(x)] <- NA_real_; storage.mode(x) <- "double"
            check <- checkData(x, verbose = FALSE)
            if (!isTRUE(check)) stop("pcaMethods checkData() failed.")
            fit <- pca(x, method = "bpca", nPcs = ncomp, completeObs = TRUE)
            completeObs(fit)
        }
    """)
    bpca_result = bpca_function(r_dataframe, n_components)
    with localconverter(default_converter + pandas2ri.converter):
        X_imputed = ro.conversion.rpy2py(bpca_result)
    X_imputed = pd.DataFrame(X_imputed, index=X_for_r.index, columns=X_for_r.columns).T
    X_imputed = X_imputed.loc[Y.index, Y.columns]
    return _blank_out_unidentifiable_rows(X_imputed.values, fully_missing_rows, Y.index, Y.columns)


def dreamai_impute(Y, k=10, maxiter_MF=10, ntree=100, maxiter_ADMIN=30, gamma=50, iter_SpectroFM=40,
                    method=("KNN", "MissForest", "ADMIN", "Birnn", "SpectroFM", "RegImpute"),
                    progress_callback=None, resume_check=None):
    """Runs each algorithm in `method` as its OWN separate DreamAI::DreamAI() call (one method
    at a time), instead of one call bundling all of them - then combines the resulting complete
    matrices into the exact same "Ensemble" a single multi-method call would produce.

    "Ensemble" is a plain elementwise mean across the requested methods' completed matrices 
    (ensemble <- ensemble + d.impute.<method> for each one, then ensemble <- ensemble / n.method) 
    - no ranking, no weighting. Here we summing per-method results and divide by len(method)
    to reproduce exactly that, just restructured so each sub-algorithm's cost - and any failure -
    is paid and observable on its own, instead of hidden behind one atomic multi-method call that
    only reports success/failure (and only checkpoints, one level up in the pilot cell) once
    everything has finished.

    Note on randomness: each sub-call below re-seeds with the same `seed` independently - the R
    closure already did `set.seed(seed)` per algorithm internally even in the old bundled call,
    so this is no less reproducible, just no longer bit-for-bit identical to whatever RNG state a
    single continuous 6-method R call would have consumed in sequence.

    progress_callback(method_name, X_hat_method_or_None, seconds), if given, is called once a
    freshly-computed sub-method finishes (X_hat_method) or raises (None) - NOT called for a
    sub-method resume_check already supplied from a checkpoint, so a resumed run doesn't
    overwrite a real timing with 0.0s.

    resume_check(method_name), if given, is called before each sub-method runs; return an
    already-imputed DataFrame (same shape/index/columns as Y) to reuse it and skip the R call
    entirely, or None to compute it fresh. Lets a caller (e.g. the pilot cell) resume a previously
    interrupted DreamAI run without redoing sub-methods that already finished.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, pandas2ri
    from rpy2.robjects.conversion import localconverter

    fully_missing_rows = Y.isna().all(axis=1).values
    Y_seeded = Y.copy()
    if fully_missing_rows.any():
        grand_mean = Y.values[Y.notna().values].mean()
        Y_seeded.loc[Y.index[fully_missing_rows]] = Y_seeded.loc[Y.index[fully_missing_rows]].fillna(grand_mean)

    X_for_r = Y_seeded.copy()
    X_for_r.index = X_for_r.index.astype(str)
    X_for_r.columns = X_for_r.columns.astype(str)
    with localconverter(default_converter + pandas2ri.converter):
        r_matrix = ro.conversion.py2rpy(X_for_r)

    dreamai_function = ro.r("""
        function(x, k, maxiter_MF, ntree, maxiter_ADMIN, gamma, iter_SpectroFM, method_vec, seed) {
            set.seed(seed); x <- as.matrix(x); storage.mode(x) <- "double"
            imp <- DreamAI::DreamAI(x, k = k, maxiter_MF = maxiter_MF, ntree = ntree,
                                     maxnodes = NULL, maxiter_ADMIN = maxiter_ADMIN,
                                     tol = 10^(-2), gamma_ADMIN = NA, gamma = gamma,
                                     CV = FALSE, fillmethod = "row_mean",
                                     maxiter_RegImpute = 10, conv_nrmse = 1e-6,
                                     iter_SpectroFM = iter_SpectroFM, method = method_vec, out = c("Ensemble"))
            imp$Ensemble
        }
    """)
    seed = int(RNG_SEED) % (2**31 - 1)

    method_matrices = []
    for method_name in method:
        cached = resume_check(method_name) if resume_check is not None else None
        if cached is not None:
            method_matrices.append(cached)
            continue
        t0 = time.time()
        try:
            single_result = dreamai_function(
                r_matrix, k, maxiter_MF, ntree, maxiter_ADMIN, gamma, iter_SpectroFM,
                ro.StrVector([method_name]), seed,
            )
            with localconverter(default_converter + pandas2ri.converter):
                X_hat_method = ro.conversion.rpy2py(single_result)
            X_hat_method = pd.DataFrame(np.asarray(X_hat_method), index=X_for_r.index, columns=X_for_r.columns)
        except Exception:
            if progress_callback is not None:
                progress_callback(method_name, None, time.time() - t0)
            raise
        if progress_callback is not None:
            progress_callback(method_name, X_hat_method, time.time() - t0)
        method_matrices.append(X_hat_method)

    # Reproduces DreamAI's own "Ensemble" exactly - elementwise mean across the requested
    # methods' completed matrices (see docstring above).
    stacked = np.stack([m.values for m in method_matrices], axis=0)
    X_imputed = pd.DataFrame(stacked.mean(axis=0), index=X_for_r.index, columns=X_for_r.columns)
    X_imputed.index, X_imputed.columns = Y.index, Y.columns
    return _blank_out_unidentifiable_rows(X_imputed.values, fully_missing_rows, Y.index, Y.columns)


def admin_impute(Y, gamma=50, maxiter_ADMIN=30):
    return dreamai_impute(Y, gamma=gamma, maxiter_ADMIN=maxiter_ADMIN, method=("ADMIN",))


# --- Cell 13: DAE / VAE (PIMMS-style autoencoders), PyTorch-backed ----------
def _standardize_for_dl(Y):
    row_mean = Y.mean(axis=1, skipna=True)
    row_std = Y.std(axis=1, skipna=True).replace(0, 1.0)
    return Y.sub(row_mean, axis=0).div(row_std, axis=0), row_mean, row_std


if DEEP_LEARNING_AVAILABLE:
    class _MLPAutoencoder(nn.Module):
        def __init__(self, n_features, hidden=64, latent=16, variational=False, dropout=0.3):
            super().__init__()
            self.variational = variational
            self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout))
            if variational:
                self.fc_mu = nn.Linear(hidden, latent)
                self.fc_logvar = nn.Linear(hidden, latent)
            else:
                self.fc_z = nn.Linear(hidden, latent)
            self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

        def forward(self, x):
            h = self.encoder(x)
            if self.variational:
                mu, logvar = self.fc_mu(h), self.fc_logvar(h)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                return self.decoder(z), mu, logvar
            z = self.fc_z(h)
            return self.decoder(z), None, None

    def _train_autoencoder(Z_np, obs_np, variational, n_epochs=300, lr=1e-3, weight_decay=1e-4,
                            corruption=0.20, beta=0.5, hidden=64, latent=16, dropout=0.3, seed=0):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        n_samples, n_features = Z_np.shape
        X = torch.tensor(np.nan_to_num(Z_np, nan=0.0), dtype=torch.float32)
        obs = torch.tensor(obs_np, dtype=torch.bool)

        model = _MLPAutoencoder(n_features, hidden=hidden, latent=latent, variational=variational, dropout=dropout)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(n_epochs):
            model.train()
            noise_mask = torch.tensor(rng.random(size=(n_samples, n_features)) < corruption) & obs
            x_in = X.clone()
            x_in[noise_mask] = 0.0
            recon, mu, logvar = model(x_in)
            loss = ((recon - X) ** 2)[obs].mean()
            if variational:
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            recon, _, _ = model(X)
        return recon.numpy()

    def _autoencoder_impute(Y, variational, **kwargs):
        Z, row_mean, row_std = _standardize_for_dl(Y)
        obs_np = Y.notna().values.T
        recon = _train_autoencoder(Z.values.T, obs_np, variational=variational, **kwargs)
        recon_df = pd.DataFrame(recon.T, index=Y.index, columns=Y.columns)
        X_filled_z = Z.where(Y.notna(), recon_df)
        X_filled = X_filled_z.mul(row_std, axis=0).add(row_mean, axis=0)
        fully_missing_rows = Y.isna().all(axis=1).values
        if fully_missing_rows.any():
            X_filled.loc[fully_missing_rows] = np.nan
        return X_filled

    def dae_impute(Y, **kwargs):
        return _autoencoder_impute(Y, variational=False, **kwargs)

    def vae_impute(Y, **kwargs):
        return _autoencoder_impute(Y, variational=True, **kwargs)


# --- Cell 14: method groups, per-matrix applicability, and FAST/MEDIUM/SLOW tiers
rng = np.random.default_rng(RNG_SEED)

GENERAL_METHODS = {  # applies to all three matrices
    "Mean": lambda Y: mean_impute(Y),
    "Median": lambda Y: median_impute(Y),
    "KNN": lambda Y: knn_impute(Y, n_neighbors=10),
    "SVD": lambda Y: svd_impute(Y, n_components=10),
    "SoftImpute": lambda Y: softimpute(Y, shrink_frac=0.10)[0],
    "PPCA": lambda Y: ppca_impute(Y, n_components=10),
    "Regression": lambda Y: regression_impute(Y, n_components=20, alpha=5.0),
    "RandomForest": lambda Y: random_forest_impute(Y, n_components=20, n_estimators=20, max_depth=6),
    "MissForest": lambda Y: missforest_impute(Y, n_components=20, n_estimators=20, max_depth=6, max_iter=4),
    "MICE": lambda Y: mice_impute(Y, n_nearest_features=30, max_iter=10),
    "CF": lambda Y: cf_impute(Y, n_factors=10, n_epochs=300, rng=rng),
    "MsImpute": lambda Y: msimpute_impute(Y, rng=rng),
    "EM_MVN": lambda Y: em_mvn_impute(Y, max_iter=6),
    "LLS": lambda Y: lls_impute(Y, k=30),
}
LEFT_CENSORED_METHODS = {  # protein matrices only
    "MinDet": lambda Y: mindet_impute(Y),
    "MinProb": lambda Y: minprob_impute(Y, rng=rng),
    "QRILC": lambda Y: qrilc_impute(Y, rng=rng),
    "LOD": lambda Y: lod_impute(Y),
    "Zero": lambda Y: zero_impute(Y),
    "HalfMin": lambda Y: half_min_impute(Y),
}
R_BACKED_METHODS = {}  # protein matrices only, if available
if BPCA_AVAILABLE:
    R_BACKED_METHODS["BPCA"] = lambda Y: bpca_impute(Y, n_components=10)
if DREAMAI_AVAILABLE:
    # progress_callback/resume_check default to None here - the full-grid cell calls this the
    # same way it always has (fn(Y_masked), no extra kwargs). The pilot cell (Cell 15) is the only 
    # caller that passes them, to get per-sub-method checkpointing on the slow DreamAI ensemble specifically 
    # - see dreamai_impute()'s docstring.
    R_BACKED_METHODS["DreamAI"] = lambda Y, progress_callback=None, resume_check=None: dreamai_impute(
        Y, progress_callback=progress_callback, resume_check=resume_check)
    R_BACKED_METHODS["ADMIN"] = lambda Y: admin_impute(Y)
DEEP_LEARNING_METHODS = {}  # all three matrices, if available
if DEEP_LEARNING_AVAILABLE:
    DEEP_LEARNING_METHODS["DAE"] = lambda Y: dae_impute(Y, n_epochs=200)
    DEEP_LEARNING_METHODS["VAE"] = lambda Y: vae_impute(Y, n_epochs=200)

PROTEIN_MATRICES = {"dlbcl_protein", "crc_protein"}


def methods_for_matrix(matrix_name):
    methods = {**GENERAL_METHODS, **DEEP_LEARNING_METHODS}
    if matrix_name in PROTEIN_MATRICES:
        methods = {**methods, **LEFT_CENSORED_METHODS, **R_BACKED_METHODS}
    return methods


# Speed tiers, reflecting each method's intrinsic computational cost
FAST_NAMES = {"Mean", "Median", "MinDet", "MinProb", "KNN", "SVD", "SoftImpute", "Zero", "HalfMin", "LOD", "QRILC", "LLS", "BPCA"}
MEDIUM_NAMES = {"PPCA", "Regression", "MICE", "CF", "MsImpute"}
SLOW_NAMES = {"RandomForest", "MissForest", "EM_MVN", "DreamAI", "ADMIN", "DAE", "VAE"}
SLOW_GRID_MASKING = {0.30}
SLOW_GRID_MNAR = {0.00, 0.25, 0.50, 0.75, 1.00}


def applicable_methods(matrix_name, row, full_grid=False):
    all_methods = methods_for_matrix(matrix_name)
    if full_grid:
        return all_methods
    methods = {n: f for n, f in all_methods.items() if n in FAST_NAMES}
    if int(row["Replicate"]) == 1:
        methods.update({n: f for n, f in all_methods.items() if n in MEDIUM_NAMES})
        if float(row["Masking_Fraction"]) in SLOW_GRID_MASKING and float(row["MNAR_Fraction"]) in SLOW_GRID_MNAR:
            methods.update({n: f for n, f in all_methods.items() if n in SLOW_NAMES})
    return methods


for name in MATRICES:
    print(name, "->", len(methods_for_matrix(name)), "applicable methods:", sorted(methods_for_matrix(name)))


# --- Cell 15: pilot scenario per matrix, to validate and get real timing -------
# Before committing to the full grid, run one representative scenario per
# matrix (10% masking, pure MCAR) through every applicable method. This
# gives real per-method timing on this data, which is what determines how
# big a grid is actually feasible to run.

# Checkpointed: each (matrix, method) result gets written to a CSV as
# soon as it finishes, so an interrupted kernel - or one slow method
# (BPCA/DreamAI/DAE/VAE are the likely culprits) hanging - can pick back
# up from there instead of re-running everything that already completed.
# The checkpoint is keyed on the scenario's own parameters too, not just
# matrix/method, so editing PILOT_ROW later and re-running is recognized
# as a different scenario - the old checkpoint is then treated as stale
# and ignored, rather than silently mixing results from two different configs.
#
# DreamAI checkpointing: It's one
# Method entry ("DreamAI") but internally bundles six R sub-algorithms (KNN,
# MissForest, ADMIN, Birnn, SpectroFM, RegImpute) - if the kernel died mid-ensemble, 
# NOTHING was saved and a resume finished. dreamai_impute() (Cell 12) runs those six 
# individually and exposes progress_callback/resume_check hooks; the block below 
# wires those to this same pilot_rows/PILOT_CHECKPOINT_PATH machinery, under Method 
# names "DreamAI_KNN", "DreamAI_MissForest", etc., so each sub-algorithm's own RMSE/timing 
# is checkpointed (and its completed matrix saved) the moment it finishes - a resume skips 
# only the sub-methods that already succeeded, same as it always could for every other 
# top-level method. The final "DreamAI" row (the elementwise-mean Ensemble across all six) 
# is unaffected downstream: same Method name, same meaning, evaluate_imputation_methods_clinical.ipynb sees 
# no difference.

PILOT_ROW = pd.Series({"Masking_Fraction": 0.30, "MNAR_Fraction": 0.0, "Replicate": 1, "Seed": RNG_SEED})
PILOT_CHECKPOINT_PATH = RESULTS_DIR / "pilot_checkpoint.csv"
PILOT_MATRICES_DIR = RESULTS_DIR / "pilot_imputed_matrices"
PILOT_MATRICES_DIR.mkdir(exist_ok=True)
_PILOT_SCENARIO_COLS = ["Masking_Fraction", "MNAR_Fraction", "Replicate", "Seed"]


def _pilot_scenario_key(row):
    return tuple(float(row[c]) for c in _PILOT_SCENARIO_COLS)


def _pilot_matrix_path(matrix_name, method_name):
    return PILOT_MATRICES_DIR / f"pilot_{matrix_name}_imputed_{method_name}.csv"


if PILOT_CHECKPOINT_PATH.exists():
    _checkpoint_df = pd.read_csv(PILOT_CHECKPOINT_PATH)
    _stale = not _checkpoint_df.empty and (
        tuple(_checkpoint_df[_PILOT_SCENARIO_COLS].iloc[0]) != _pilot_scenario_key(PILOT_ROW)
    )
    if _stale:
        print("pilot checkpoint is for a different PILOT_ROW config - ignoring it, starting fresh.")
        pilot_rows = []
    else:
        pilot_rows = _checkpoint_df.to_dict("records")
        print(f"resuming pilot from checkpoint: {len(pilot_rows)} (matrix, method) result(s) already done.")
else:
    pilot_rows = []

# Indexed by (matrix, method) -> row position, so re-running a method
# that was already checkpointed (but whose matrix wasn't saved) updates
# its existing row in place instead of appending a duplicate.
_pilot_row_index = {(r["Matrix"], r["Method"]): i for i, r in enumerate(pilot_rows)}


def _checkpoint_pilot_row(row_dict):
    """Write one (matrix, method) result into pilot_rows/_pilot_row_index and flush the
    whole checkpoint to disk - shared by the generic per-method loop below and by
    DreamAI's per-sub-method progress_callback, so both checkpoint through the exact
    same path."""
    key = (row_dict["Matrix"], row_dict["Method"])
    if key in _pilot_row_index:
        pilot_rows[_pilot_row_index[key]] = row_dict
    else:
        pilot_rows.append(row_dict)
        _pilot_row_index[key] = len(pilot_rows) - 1
    pd.DataFrame(pilot_rows).to_csv(PILOT_CHECKPOINT_PATH, index=False)


def _make_dreamai_hooks(_matrix_name, _X, _mask):
    """DreamAI-specific progress_callback/resume_check, closed over one matrix's truth/mask
    so each of DreamAI's six sub-methods can be scored and checkpointed independently. Called
    once per matrix, below, right before that matrix's "DreamAI" entry runs."""
    def _resume_check(sub_name):
        sub_method_name = f"DreamAI_{sub_name}"
        sub_path = _pilot_matrix_path(_matrix_name, sub_method_name)
        if (_matrix_name, sub_method_name) in _pilot_row_index and sub_path.exists():
            print(f"    DreamAI/{sub_name:10s} (skipped - already checkpointed)")
            return pd.read_csv(sub_path, index_col="Feature_ID")
        return None

    def _progress_callback(sub_name, X_hat_sub, sub_seconds):
        sub_method_name = f"DreamAI_{sub_name}"
        if X_hat_sub is None:
            sm = {"Matrix": _matrix_name, "Method": sub_method_name, "Seconds": sub_seconds,
                  "RMSE": np.nan, "Error": "sub-method failed (see the exception/R output above)"}
            print(f"    !! DreamAI/{sub_name} failed after {sub_seconds:.2f}s")
        else:
            sm = calculate_metrics(_X, X_hat_sub, _mask)
            sm.update({"Matrix": _matrix_name, "Method": sub_method_name, "Seconds": sub_seconds})
            X_hat_sub.rename_axis("Feature_ID").to_csv(_pilot_matrix_path(_matrix_name, sub_method_name))
            print(f"    DreamAI/{sub_name:10s} RMSE={sm.get('RMSE', float('nan')):.4f}  "
                  f"({sub_seconds:.2f}s) [checkpointed]")
        sm.update({c: PILOT_ROW[c] for c in _PILOT_SCENARIO_COLS})
        _checkpoint_pilot_row(sm)

    return _progress_callback, _resume_check


for matrix_name, X in MATRICES.items():
    print(f"\n=== pilot: {matrix_name} {X.shape} ===")
    _a, _b = MNAR_PROBIT_PARAMS[matrix_name]
    Y_masked, mask, mechanism = build_masked_scenario(PILOT_ROW, X, a=_a, b=_b)

    for method_name, fn in methods_for_matrix(matrix_name).items():
        matrix_path = _pilot_matrix_path(matrix_name, method_name)
        if (matrix_name, method_name) in _pilot_row_index and matrix_path.exists():
            print(f"  {method_name:12s} (skipped - checkpoint + saved matrix both present)")
            continue
        t0 = time.time()
        try:
            if method_name == "DreamAI":
                _progress_callback, _resume_check = _make_dreamai_hooks(matrix_name, X, mask)
                X_hat = fn(Y_masked, progress_callback=_progress_callback, resume_check=_resume_check)
            else:
                X_hat = fn(Y_masked)
            m = calculate_metrics(X, X_hat, mask)
            m.update({"Method": method_name, "Seconds": time.time() - t0})
            X_hat.rename_axis("Feature_ID").to_csv(matrix_path)  # for the evaluation notebook
        except Exception as exc:
            m = {"Method": method_name, "Seconds": time.time() - t0, "RMSE": np.nan, "Error": str(exc)}
            print(f"  !! {method_name} failed: {type(exc).__name__}: {exc}")
        m.update({"Matrix": matrix_name, **{c: PILOT_ROW[c] for c in _PILOT_SCENARIO_COLS}})
        _checkpoint_pilot_row(m)
        print(f"  {method_name:12s} RMSE={m.get('RMSE', float('nan')):.4f}  ({m['Seconds']:.2f}s)")

pilot_results = pd.DataFrame(pilot_rows)
pilot_summaries = {
    matrix_name: pilot_results[pilot_results["Matrix"] == matrix_name].set_index("Method")
    for matrix_name in MATRICES
}


# --- Cell 16: experiment design (masking-fraction x --
# MNAR-fraction x replicate factorial), plus the run_full_benchmark() definition.
# The one-off checkpoint repair and execution loop are the two cells
# below this one - kept separate so the repair always runs before anything
# gets (re)computed under a stale checkpoint, regardless of which grid was used before...the reason 
# for doing this is because I had to reduce the masking fracttions; removed 0.1, and 0.50. 
MASKING_FRACTIONS = [0.30]  
                                   
                                   
MNAR_FRACTIONS = [0.00, 0.25, 0.50, 0.75, 1.00]
N_REPLICATES = 5  # -> len(MASKING_FRACTIONS) * len(MNAR_FRACTIONS) * N_REPLICATES scenarios

FULL_GRID_FOR_ALL_METHODS = True   # True runs every method on every scenario; that's a big jump in runtime,
                                   # so better when False (off)

_rows = []
_scenario_id = 0
for mf in MASKING_FRACTIONS:
    for nf in MNAR_FRACTIONS:
        for rep in range(1, N_REPLICATES + 1):
            _scenario_id += 1
            _rows.append({"Scenario_ID": _scenario_id, "Masking_Fraction": mf, "MNAR_Fraction": nf,
                           "Replicate": rep, "Seed": RNG_SEED * 10_000 + _scenario_id})
experiment_design = pd.DataFrame(_rows)
print("experiment design:", experiment_design.shape)

# Scenarios are fully independent of each other (each builds its own masked
# matrix and scores its own applicable methods against it), so they now run
# N_SCENARIO_WORKERS-at-a-time via joblib/loky instead of one at a time - see
# Cell 1's _split_cpu_budget for why. R-backed methods (BPCA/DreamAI/ADMIN)
# are the one exception: rpy2's embedded R interpreter isn't safe to
# reinitialize repeatedly across worker processes, so those three still run
# sequentially in this main process/kernel, after the parallel pass -
# rebuilding their scenario's mask a second time there is negligible next to
# an R call's own runtime.
import joblib
from sklearn.utils.parallel import Parallel, delayed

R_BACKED_METHOD_NAMES = set(R_BACKED_METHODS.keys())

_RNG_DEPENDENT_FACTORY = {
    "CF": lambda rng: (lambda Y: cf_impute(Y, n_factors=10, n_epochs=300, rng=rng)),
    "MsImpute": lambda rng: (lambda Y: msimpute_impute(Y, rng=rng)),
    "MinProb": lambda rng: (lambda Y: minprob_impute(Y, rng=rng)),
    "QRILC": lambda rng: (lambda Y: qrilc_impute(Y, rng=rng)),
}


def _reseed_rng_methods(method_items, seed):
    """Replaces CF/MsImpute/MinProb/QRILC's closures (normally over the single
    shared, mutable `rng` from Cell 14, advanced call-to-call as the old
    sequential loop ran) with copies bound to their own scenario-seeded
    generator instead.

    This matters once scenarios run in separate worker processes: each worker
    gets its own pickled copy of whatever `rng` state existed when the task
    was shipped off, so without this, every scenario would silently draw the
    *same* frozen initial random values for these four methods instead of
    scenario-distinct ones - worse than the original sequential behavior
    (where at least call order gave each one different, if not reproducibly
    ordered, draws). Seeding from the scenario's own `Seed` column mirrors
    what `build_masked_scenario` already does for the mask itself, which
    makes this both correctly parallel *and* actually reproducible per
    scenario - interrupting and resuming this cell previously changed these
    four methods' results even in the original sequential code, since the
    shared `rng`'s state depended on exactly how many prior calls happened to
    consume it first.
    """
    out = dict(method_items)
    for name, factory in _RNG_DEPENDENT_FACTORY.items():
        if name in out:
            out[name] = factory(np.random.default_rng(seed))
    return out


def _score_one(X, Y_masked, mask, mcar_only, mnar_only, row, name, fn, matrix_name=None):
    try:
        X_hat = fn(Y_masked)
        overall = calculate_metrics(X, X_hat, mask)
        rec = {"Scenario_ID": row["Scenario_ID"], "Method": name,
               "Masking_Fraction": row["Masking_Fraction"], "MNAR_Fraction": row["MNAR_Fraction"],
               "Replicate": row["Replicate"], **overall}
        if mcar_only.values.any():
            rec["RMSE_MCAR_subset"] = calculate_metrics(X, X_hat, mcar_only)["RMSE"]
        if mnar_only.values.any():
            rec["RMSE_MNAR_subset"] = calculate_metrics(X, X_hat, mnar_only)["RMSE"]
    except Exception as exc:
        rec = {"Scenario_ID": row["Scenario_ID"], "Method": name,
               "Masking_Fraction": row["Masking_Fraction"], "MNAR_Fraction": row["MNAR_Fraction"],
               "Replicate": row["Replicate"], "RMSE": np.nan, "Error": f"{type(exc).__name__}: {exc}"}
        if matrix_name is not None:
            print(f"    !! {matrix_name}/{name} failed on scenario {row['Scenario_ID']}: {exc}")
    return rec


def _run_scenario_methods(X_values, X_index, X_columns, row_dict, method_items, threads_per_worker, a, b):
    """Runs one scenario's masked matrix through {name: fn} entirely inside one
    worker process. Sets this worker's own PyTorch/NumExpr thread budget here
    as a best-effort extra measure - but note this does NOT reliably cap
    OpenBLAS/OpenMP: this function's own os.environ assignment below runs only
    after this fresh worker process has already imported numpy/pandas/sklearn
    while unpickling its own arguments (they're referenced as globals in this
    closure), and those libraries read OMP_NUM_THREADS/OPENBLAS_NUM_THREADS
    once, at that import/init moment - inheriting whatever value the *parent*
    kernel process had (Cell 1's full-node N_JOBS budget), not the smaller
    value this line tries to set."""
    import os
    import warnings
    # Cell 1's warnings.filterwarnings("ignore") only touches *this* kernel
    # process's filter state.
    warnings.filterwarnings("ignore")
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = str(threads_per_worker)
    global N_JOBS
    N_JOBS = threads_per_worker   # so random_forest_impute/missforest_impute's own
                                  # n_jobs=N_JOBS uses this worker's budget, not the node's
    try:
        import torch
        torch.set_num_threads(threads_per_worker)
    except Exception:
        pass

    X = pd.DataFrame(X_values, index=X_index, columns=X_columns)
    row = pd.Series(row_dict)
    method_items = _reseed_rng_methods(method_items, int(row["Seed"]))
    Y_masked, mask, mechanism = build_masked_scenario(row, X, a=a, b=b)
    mcar_only = pd.DataFrame(mechanism.values == "MCAR", index=mask.index, columns=mask.columns)
    mnar_only = pd.DataFrame(mechanism.values == "MNAR", index=mask.index, columns=mask.columns)

    return [_score_one(X, Y_masked, mask, mcar_only, mnar_only, row, name, fn)
            for name, fn in method_items.items()]


def run_full_benchmark(matrix_name, X, checkpoint_path, n_workers=None, threads_per_worker=None):
    n_workers = N_SCENARIO_WORKERS if n_workers is None else n_workers
    threads_per_worker = THREADS_PER_WORKER if threads_per_worker is None else threads_per_worker
    a, b = MNAR_PROBIT_PARAMS[matrix_name]

    if checkpoint_path.exists():
        all_results = pd.read_csv(checkpoint_path).to_dict("records")
        print(f"{matrix_name}: resuming from checkpoint, {len(all_results)} results already done.")
    else:
        all_results = []
    already_done = {(r["Scenario_ID"], r["Method"]) for r in all_results}

    X_values, X_index, X_columns = X.values, X.index, X.columns

    parallel_tasks, sequential_tasks = [], []
    for _, row in experiment_design.iterrows():
        active = applicable_methods(matrix_name, row, full_grid=FULL_GRID_FOR_ALL_METHODS)
        remaining = {n: f for n, f in active.items() if (row["Scenario_ID"], n) not in already_done}
        if not remaining:
            continue
        remaining_parallel = {n: f for n, f in remaining.items() if n not in R_BACKED_METHOD_NAMES}
        remaining_sequential = {n: f for n, f in remaining.items() if n in R_BACKED_METHOD_NAMES}
        if remaining_parallel:
            parallel_tasks.append((row, remaining_parallel))
        if remaining_sequential:
            sequential_tasks.append((row, remaining_sequential))

    print(f"{matrix_name}: {len(parallel_tasks)} scenario(s) with parallelizable methods "
          f"({n_workers} worker(s) x {threads_per_worker} thread(s) each), "
          f"{len(sequential_tasks)} scenario(s) with R-backed methods left (sequential).")

    # Stream results back and checkpoint per-scenario as each one finishes,
    # rather than waiting for an entire wave of n_workers scenarios to finish
    # before saving any of them.
    with joblib.parallel_config(backend="loky", inner_max_num_threads=threads_per_worker):
        parallel = Parallel(n_jobs=n_workers, return_as="generator_unordered")
        tasks = (
            delayed(_run_scenario_methods)(
                X_values, X_index, X_columns, row.to_dict(), method_items, threads_per_worker, a, b
            )
            for row, method_items in parallel_tasks
        )
        for i, records in enumerate(parallel(tasks), start=1):
            for rec in records:
                all_results.append(rec)
                already_done.add((rec["Scenario_ID"], rec["Method"]))
            pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)
            print(f"  {matrix_name}: {i}/{len(parallel_tasks)} scenario(s) done (parallel pass)")

    for row, method_items in sequential_tasks:
        method_items = _reseed_rng_methods(method_items, int(row["Seed"]))
        Y_masked, mask, mechanism = build_masked_scenario(row, X, a=a, b=b)
        mcar_only = pd.DataFrame(mechanism.values == "MCAR", index=mask.index, columns=mask.columns)
        mnar_only = pd.DataFrame(mechanism.values == "MNAR", index=mask.index, columns=mask.columns)
        for name, fn in method_items.items():
            rec = _score_one(X, Y_masked, mask, mcar_only, mnar_only, row, name, fn, matrix_name=matrix_name)
            all_results.append(rec)
            already_done.add((row["Scenario_ID"], name))
            pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(all_results)


# --- Cell 17: One-off cleanup: repair existing checkpoints for the reduced 2-masking-fraction
# grid, and clear stale SVD/SoftImpute/PPCA/MsImpute/LLS results after those methods'
# implementations were corrected'. It does no harm on subsequent run. 

BUGFIXED_METHOD_NAMES = {"SVD", "SoftImpute", "PPCA", "MsImpute", "LLS"}
_scenario_lookup = {
    (round(float(r["Masking_Fraction"]), 10), round(float(r["MNAR_Fraction"]), 10), int(r["Replicate"])):
        int(r["Scenario_ID"])
    for _, r in experiment_design.iterrows()
}


def _repair_checkpoint(df):
    keep_mask, new_scenario_id = [], []
    for _, r in df.iterrows():
        key = (round(float(r["Masking_Fraction"]), 10), round(float(r["MNAR_Fraction"]), 10), int(r["Replicate"]))
        sid = _scenario_lookup.get(key)
        keep = sid is not None and r["Method"] not in BUGFIXED_METHOD_NAMES
        keep_mask.append(keep)
        new_scenario_id.append(sid if keep else None)
    out = df[pd.Series(keep_mask, index=df.index)].copy()
    out["Scenario_ID"] = [sid for sid, k in zip(new_scenario_id, keep_mask) if k]
    return out


for matrix_name in MATRICES:
    ckpt_path = RESULTS_DIR / f"benchmark_checkpoint_{matrix_name}.csv"
    if not ckpt_path.exists():
        continue
    df = pd.read_csv(ckpt_path)
    before = len(df)
    df = _repair_checkpoint(df)
    df.to_csv(ckpt_path, index=False)
    print(f"benchmark_checkpoint_{matrix_name}.csv: {before} -> {len(df)} rows kept "
          f"(dropped: masking fractions outside {MASKING_FRACTIONS}, and all "
          f"{sorted(BUGFIXED_METHOD_NAMES)} rows)")

if PILOT_CHECKPOINT_PATH.exists():
    _ckpt = pd.read_csv(PILOT_CHECKPOINT_PATH)
    _before = len(_ckpt)
    _ckpt = _ckpt[~_ckpt["Method"].isin(BUGFIXED_METHOD_NAMES)]
    _ckpt.to_csv(PILOT_CHECKPOINT_PATH, index=False)
    print(f"pilot_checkpoint.csv: removed {_before - len(_ckpt)} row(s) for "
          f"{sorted(BUGFIXED_METHOD_NAMES)}, {len(_ckpt)} remain")

_removed = 0
for method_name in BUGFIXED_METHOD_NAMES:
    for matrix_name in MATRICES:
        f = _pilot_matrix_path(matrix_name, method_name)
        if f.exists():
            f.unlink()
            _removed += 1
print(f"pilot_imputed_matrices/: removed {_removed} matrix file(s) for {sorted(BUGFIXED_METHOD_NAMES)}")


# --- Cell 17b: prune checkpoint rows that no longer match a method's CURRENT tier -----------
# applicability, not just rows outside the current masking-fraction grid. 
_design_by_scenario = {int(r["Scenario_ID"]): r for _, r in experiment_design.iterrows()}


def _prune_inapplicable_rows(df, matrix_name):
    keep_mask = []
    n_dropped_by_method = {}
    for _, r in df.iterrows():
        design_row = _design_by_scenario.get(int(r["Scenario_ID"]))
        if design_row is None:
            keep_mask.append(True)  # already handled by Cell 17's Scenario_ID repair
            continue
        # Only prune rows for methods this notebook's own applicable_methods() actually
        # manages (the 25-method roster). Anything else; MOFA+_native and its variants,
        # added to these same checkpoint files by evaluate_mofaplus_multiomic_clinical.ipynb's
        # own full-grid cell - isn't in that roster at all, so applicable_methods() would
        # incorrectly report it as never-applicable and purge every one of its rows.
        if r["Method"] not in methods_for_matrix(matrix_name):
            keep_mask.append(True)
            continue
        ok = r["Method"] in applicable_methods(matrix_name, design_row, full_grid=FULL_GRID_FOR_ALL_METHODS)
        keep_mask.append(ok)
        if not ok:
            n_dropped_by_method[r["Method"]] = n_dropped_by_method.get(r["Method"], 0) + 1
    out = df[pd.Series(keep_mask, index=df.index)].copy()
    return out, n_dropped_by_method


for matrix_name in MATRICES:
    ckpt_path = RESULTS_DIR / f"benchmark_checkpoint_{matrix_name}.csv"
    if not ckpt_path.exists():
        continue
    df = pd.read_csv(ckpt_path)
    before = len(df)
    df, n_dropped_by_method = _prune_inapplicable_rows(df, matrix_name)
    if n_dropped_by_method:
        df.to_csv(ckpt_path, index=False)
        detail = ", ".join(f"{m}: {n}" for m, n in sorted(n_dropped_by_method.items()))
        print(f"benchmark_checkpoint_{matrix_name}.csv: {before} -> {len(df)} rows "
              f"(dropped rows no longer applicable under the current tier for their method: {detail})")
    else:
        print(f"benchmark_checkpoint_{matrix_name}.csv: {before} rows, already consistent with current tiering - no changes")


# -------- Cell 18: Run one matrix at a time ------------- 
# each call resumes from its own checkpoint if interrupted, 
# so it's safe to run matrices in separate sessions instead of all at once.
# (warnings are suppressed inside _run_scenario_methods - see Cell 16 -
# since that's the process they actually get raised in)
results = {}
for matrix_name, X in MATRICES.items():
    results[matrix_name] = run_full_benchmark(matrix_name, X, RESULTS_DIR / f"benchmark_checkpoint_{matrix_name}.csv")


# --- Cell 19: aggregate the results and rank methods (once Cell 16 - 17 has been run) ----------
all_results = pd.concat(
 [df.assign(Matrix=name) for name, df in results.items()], ignore_index=True
 )
all_results.to_csv(RESULTS_DIR / "benchmark_results_full.csv", index=False)
overall_ranking = (
 all_results.groupby(["Matrix", "Method"])["RMSE"]
 .agg(["mean", "std", "count"]).sort_values(["Matrix", "mean"])
 )
overall_ranking
