# --- Cell 1: imports, environment guards, seed -------------------------------
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

# R/rpy2 guard (BPCA, DreamAI, ADMIN)- attempted once; if unavailable,
# those three methods are simply skipped. 
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

# PyTorch guard (DAE, VAE)
RUN_DEEP_LEARNING = True
DEEP_LEARNING_AVAILABLE = False
if RUN_DEEP_LEARNING:
    try:
        import torch
        import torch.nn as nn
        DEEP_LEARNING_AVAILABLE = True
        print("PyTorch available: DAE/VAE enabled.")
    except Exception as exc:
        print(f"DAE/VAE disabled ({type(exc).__name__}: {exc}). Requires PyTorch.")


# --- Cell 2: locate + load the processed data --------------------------------
try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()

import os

# Same env-var-first, then-candidate-list pattern as read_multiomic_data.py's
# resolve_data_file() / resolve_output_dir(). 
CANDIDATE_PROCESSED_DIRS = []
if os.environ.get("IMPUTATION_OUTPUT_DIR"):
    CANDIDATE_PROCESSED_DIRS.append(Path(os.environ["IMPUTATION_OUTPUT_DIR"]).expanduser())
CANDIDATE_PROCESSED_DIRS += [
    Path.home() / "reuben_imputation" / "processed",
    Path("/projects/datasets_BIO/reuben_imputation/processed"),
    (THIS_DIR / "..").resolve() / "data" / "processed",
    THIS_DIR / "processed",
]
PROCESSED_DIR = next((d for d in CANDIDATE_PROCESSED_DIRS if d.exists()), None)
if PROCESSED_DIR is None:
    raise FileNotFoundError(
        "no 'processed' directory found among: " + ", ".join(str(d) for d in CANDIDATE_PROCESSED_DIRS) +
        ". Run read_multiomic_data.py through Cell 14 first, or set IMPUTATION_OUTPUT_DIR."
    )

# RESULTS_DIR: try alongside PROCESSED_DIR first fall back to the home-dir
# location if PROCESSED_DIR was instead found on a read-only mount. 
try:
    RESULTS_DIR = PROCESSED_DIR.parent / "imputation_results"
    RESULTS_DIR.mkdir(exist_ok=True)
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



# --- Cell 3: build the 3 benchmark matrices (features x samples, per point 1)----
rna_adata = mdata.mod["1_rna"]
assert (rna_adata.obs_names == mdata.obs_names).all()
profiled_mask = mdata.obs["rna_profiled"].to_numpy()

rna_full = pd.DataFrame(
    to_dense(rna_adata.X)[profiled_mask].T, index=rna_adata.var_names, columns=rna_adata.obs_names[profiled_mask]
)
prot_full = pd.DataFrame(to_dense(mdata.mod["2_prot"].X).T, index=mdata.mod["2_prot"].var_names, columns=mdata.obs_names)
crc_full = crc_df.T  # back to features x samples for this benchmark

N_TOP_FEATURES = 500  # raise once the pipeline's confirmed working


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

# Scale check (point 2 above) - aggregate only, never raw values.
for name, df in MATRICES.items():
    v = df.values
    print(
        f"{name}: {df.shape} (features x samples) | "
        f"min={np.nanmin(v):.2f} max={np.nanmax(v):.2f} mean={np.nanmean(v):.2f} | "
        f"{100 * np.isnan(v).mean():.1f}% missing"
    )
print(
    "\nIf a matrix's max is well above ~30, it's very unlikely to already be log2-scale "
    "(this dataset's log2 CRC values top out under 30) -- log2-transform it before "
    "continuing, for the reason in the markdown above."
)


# --- Cell 4: masking (MCAR / MNAR / mixed) ----------
def generate_mcar_mask(X, masking_fraction=0.20, rng=None, exclude_mask=None):
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


def generate_mnar_weighted_positions(X, n_to_mask, slope=-1.5, rng=None, exclude_mask=None):
    if rng is None:
        rng = np.random.default_rng()
    eligible = X.notna().values
    if exclude_mask is not None:
        eligible = eligible & ~exclude_mask.values
    positions = np.argwhere(eligible)
    if n_to_mask == 0:
        return np.empty((0, 2), dtype=int)
    values = X.values[eligible]  # assumed already on the right analysis scale
    mean_v, std_v = values.mean(), values.std()
    z = np.zeros_like(values) if std_v == 0 else (values - mean_v) / std_v
    weights = np.exp(slope * z)
    weights = weights / weights.sum()
    selected = rng.choice(len(positions), size=n_to_mask, replace=False, p=weights)
    return positions[selected]


def generate_mnar_mask(X, masking_fraction=0.20, slope=-1.5, rng=None, exclude_mask=None):
    if rng is None:
        rng = np.random.default_rng()
    eligible = X.notna().values
    if exclude_mask is not None:
        eligible = eligible & ~exclude_mask.values
    n_to_mask = int(round(masking_fraction * eligible.sum()))
    sel_pos = generate_mnar_weighted_positions(X, n_to_mask, slope=slope, rng=rng, exclude_mask=exclude_mask)
    mask = np.zeros(X.shape, dtype=bool)
    if len(sel_pos) > 0:
        mask[sel_pos[:, 0], sel_pos[:, 1]] = True
    return pd.DataFrame(mask, index=X.index, columns=X.columns)


def generate_mixed_mask(X, masking_fraction=0.20, mnar_fraction=0.50, slope=-1.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    eligible = X.notna().values
    n_eligible = eligible.sum()
    total_to_mask = int(round(masking_fraction * n_eligible))
    n_mnar = int(round(total_to_mask * mnar_fraction))
    n_mcar = total_to_mask - n_mnar

    mnar_positions = generate_mnar_weighted_positions(X, n_mnar, slope=slope, rng=rng)
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


def build_masked_scenario(row, X_truth, slope=-1.5):
    rng = np.random.default_rng(int(row["Seed"]))
    mask, mechanism = generate_mixed_mask(
        X_truth, masking_fraction=float(row["Masking_Fraction"]), mnar_fraction=float(row["MNAR_Fraction"]),
        slope=slope, rng=rng,
    )
    return X_truth.mask(mask), mask, mechanism

# --- Cell 5: evaluation metrics -------
def calculate_metrics(truth, prediction, mask):
    y_true_mat, y_pred_mat, mask_mat = truth.values, prediction.values, mask.values
    y_true, y_pred = y_true_mat[mask_mat], y_pred_mat[mask_mat]
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    errors = y_true - y_pred
    rmse = np.sqrt(np.mean(errors ** 2)) if len(errors) else np.nan
    mae = np.mean(np.abs(errors)) if len(errors) else np.nan
    pearson_r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan

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


# --- Cell 6: shared helpers + baseline substitution + KNN/SVD/SoftImpute -----
def mean_impute(Y):
    row_means = Y.mean(axis=1, skipna=True)
    return Y.T.fillna(row_means).T


def median_impute(Y):
    row_medians = Y.median(axis=1, skipna=True)
    return Y.T.fillna(row_medians).T


def mindet_impute(Y, q=0.01):
    """Deterministic left-censored fill (Perseus/DEP-style 'MinDet'): each
    sample's missing entries get that sample's own q-th quantile of
    observed values -- a per-sample apparent detection floor."""
    return Y.fillna(Y.quantile(q, axis=0))


def minprob_impute(Y, q=0.01, sigma_scale=0.3, rng=None):
    """Probabilistic left-censored fill ('MinProb', Lazar et al. 2016):
    draws from N(quantile_q(observed), sigma_scale * sd(observed)) per
    sample."""
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
    """Distance-weighted KNN across samples, per-feature standardized first
    so no feature's variance dominates the distance metric."""
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


def _initial_fill(Y):
    row_means = Y.mean(axis=1, skipna=True)
    fully_missing_rows = row_means.isna().values
    grand_mean = row_means.mean()
    row_means_filled = row_means.fillna(grand_mean)
    filled = Y.T.fillna(row_means_filled).T.to_numpy(dtype=float, copy=True)
    return filled, Y.notna().values, fully_missing_rows


def _blank_out_unidentifiable_rows(X_filled, fully_missing_rows, index, columns):
    result = pd.DataFrame(X_filled, index=index, columns=columns)
    if fully_missing_rows.any():
        result.loc[fully_missing_rows] = np.nan
    return result


def svd_impute(Y, n_components=10, max_iter=100, tol=1e-5):
    X_filled, observed, fully_missing_rows = _initial_fill(Y)
    for _ in range(max_iter):
        old = X_filled.copy()
        U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
        r = min(n_components, len(s))
        reconstruction = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
        X_filled[~observed] = reconstruction[~observed]
        denom = np.linalg.norm(old)
        if (np.linalg.norm(X_filled - old) / denom if denom > 0 else 0.0) < tol:
            break
    return _blank_out_unidentifiable_rows(X_filled, fully_missing_rows, Y.index, Y.columns)


def _soft_threshold_svd(M, lam):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ np.diag(np.maximum(s - lam, 0)) @ Vt


def softimpute(Y, shrink_frac=0.10, max_iter=200, tol=1e-5):
    X_filled, observed, fully_missing_rows = _initial_fill(Y)
    lam = shrink_frac * np.linalg.svd(X_filled, compute_uv=False)[0]
    for _ in range(max_iter):
        old = X_filled.copy()
        reconstruction = _soft_threshold_svd(X_filled, lam)
        X_filled[~observed] = reconstruction[~observed]
        X_filled[observed] = Y.values[observed]
        denom = np.linalg.norm(old)
        if (np.linalg.norm(X_filled - old) / denom if denom > 0 else 0.0) < tol:
            break
    return _blank_out_unidentifiable_rows(X_filled, fully_missing_rows, Y.index, Y.columns), lam


# --- Cell 7: PPCA (incomplete-data EM, Roweis 1998) -- ported verbatim ------
def ppca_impute(Y, n_components=10, max_iter=100, tol=1e-6):
    P, N = Y.shape
    obs = Y.notna().values
    row_mean = Y.mean(axis=1, skipna=True).values
    row_mean = np.where(np.isnan(row_mean), np.nanmean(row_mean), row_mean)
    Yc = Y.values - row_mean[:, None]

    K = n_components
    filled0 = np.where(obs, np.nan_to_num(Yc), 0.0)
    U, s, Vt = np.linalg.svd(filled0, full_matrices=False)
    C = U[:, :K] * s[:K]
    eye_k = np.eye(K) * 1e-6

    prev_recon = None
    for _ in range(max_iter):
        Z = np.zeros((K, N))
        for n in range(N):
            o = obs[:, n]
            if not o.any():
                continue
            Co = C[o, :]
            Z[:, n] = np.linalg.solve(Co.T @ Co + eye_k, Co.T @ Yc[o, n])

        C_new = np.zeros((P, K))
        for p in range(P):
            o = obs[p, :]
            if not o.any():
                C_new[p, :] = np.nan
                continue
            Zo = Z[:, o]
            C_new[p, :] = np.linalg.solve(Zo @ Zo.T + eye_k, Zo @ Yc[p, o])
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
# All three predict each feature-with-missing-entries from a shared ~20-dim
# PCA summary of the current best-guess matrix, refit each iteration-(raw feature-on-feature regression is unstable
# and, at N_TOP_FEATURES per-feature models, too slow).
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
        Y, lambda: RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=-1),
        n_components=n_components, max_iter=1,
    )


def missforest_impute(Y, n_components=20, n_estimators=50, max_depth=8, max_iter=6, tol=1e-3):
    return _regression_family_impute(
        Y, lambda: RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=-1),
        n_components=n_components, max_iter=max_iter, tol=tol,
    )

# --- Cell 9: LLS, EM_MVN (MLE/EM), MICE ---------------------
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
        if row_obs.sum() < 5 or row_mis.sum() == 0:
            continue
        neighbor_order = np.argsort(-corr[p])
        neighbors = [n for n in neighbor_order if corr[p, n] > -np.inf and n != p][:k]
        if len(neighbors) < 3:
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
    # sklearn's IterativeImputer silently DROPS any fully-missing column
    # from its output array (documented behavior, not a bug in sklearn) -
    # at the highest masking/MNAR corners of the grid, a protein that's
    # already ~34% really missing can end up wholly missing after the
    # additional MNAR-weighted mask, so completed.T then has fewer rows
    # than Y.index and the DataFrame reconstruction below used to raise
    # "Shape of passed values is (N, M), indices imply (500, M)". Same
    # seed-with-grand-mean fix already used throughout this file
    # (dreamai_impute, svd_impute, ppca_impute, ...) via the shared
    # _blank_out_unidentifiable_rows helper: seed just enough that sklearn
    # never sees a fully-NaN column to drop, then blank that row back to
    # NaN in the final output.
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


# --- Cell 10: MsImpute (data-driven-rank softImpute) + CF -------
def msimpute_impute(Y, candidate_ranks=(2, 4, 6, 8, 10, 15, 20), holdout_frac=0.10, max_iter=150, tol=1e-5, rng=None):
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
        X_filled, obs, fully_missing_rows = _initial_fill(Y_in)
        for _ in range(max_iter):
            old = X_filled.copy()
            recon = _soft_threshold_svd(X_filled, lam)
            X_filled[~obs] = recon[~obs]
            X_filled[obs] = Y_in.values[obs]
            denom = np.linalg.norm(old)
            if (np.linalg.norm(X_filled - old) / denom if denom > 0 else 0.0) < tol:
                break
        return X_filled, fully_missing_rows

    best_rank, best_err = None, np.inf
    for r in candidate_ranks:
        s_max_cv = np.linalg.svd(_initial_fill(Y_cv)[0], compute_uv=False)[0]
        X_filled, _ = _fit(Y_cv, (r / max(candidate_ranks)) * 0.15 * s_max_cv)
        pred = X_filled[holdout_pos[:, 0], holdout_pos[:, 1]]
        truth = Y.values[holdout_pos[:, 0], holdout_pos[:, 1]]
        err = np.sqrt(np.nanmean((pred - truth) ** 2))
        if err < best_err:
            best_err, best_rank = err, r

    s_max = np.linalg.svd(_initial_fill(Y)[0], compute_uv=False)[0]
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


# --- Cell 11: baseline-substitution / left-censored family (protein only) ---
# Zero, HalfMin, LOD, QRILC ------. MinDet/MinProb already in
# Cell 6. 
def zero_impute(Y):
    return Y.fillna(0.0)


def half_min_impute(Y):
    row_half_min = Y.min(axis=1, skipna=True) / 2.0
    return Y.T.fillna(row_half_min).T


def lod_impute(Y, lod_value=None):
    """Fixed global limit-of-detection substitution. Defaults to this
    matrix's own smallest observed value (notebook 04 used a pre-established
    dataset constant; we don't have that history for our data, so it's
    derived directly here instead)."""
    if lod_value is None:
        lod_value = float(np.nanmin(Y.values))
    return Y.fillna(lod_value)


def qrilc_impute(Y, q_tail=0.30, rng=None):
    """Approximate QRILC (Lazar et al. 2016): per feature, fit a truncated
    normal to the observed left tail, draw missing entries from it."""
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


# --- Cell 12: BPCA + DreamAI/ADMIN -- R/rpy2-backed,---------
def bpca_impute(Y, n_components=10):
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, pandas2ri
    from rpy2.robjects.conversion import localconverter

    # pcaMethods' checkData() rejects a matrix with any fully-missing row
    # (a protein with zero observed values anywhere) - at the highest
    # masking/MNAR corners of the grid, dlbcl_protein's real ~34% baseline
    # missingness plus the artificial mask can push some proteins there,
    # raising "pcaMethods checkData() failed." Same seed-with-grand-mean fix
    # dreamai_impute (right below) already uses, via the shared
    # _blank_out_unidentifiable_rows helper used throughout this file.
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
                    method=("KNN", "MissForest", "ADMIN", "Birnn", "SpectroFM", "RegImpute")):
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
    method_vec = ro.StrVector(list(method))
    seed = int(RNG_SEED) % (2**31 - 1)
    dreamai_result = dreamai_function(r_matrix, k, maxiter_MF, ntree, maxiter_ADMIN, gamma, iter_SpectroFM, method_vec, seed)
    with localconverter(default_converter + pandas2ri.converter):
        X_imputed = ro.conversion.rpy2py(dreamai_result)
    X_imputed = pd.DataFrame(np.asarray(X_imputed), index=X_for_r.index, columns=X_for_r.columns)
    X_imputed.index, X_imputed.columns = Y.index, Y.columns
    return _blank_out_unidentifiable_rows(X_imputed.values, fully_missing_rows, Y.index, Y.columns)


def admin_impute(Y, gamma=50, maxiter_ADMIN=30):
    return dreamai_impute(Y, gamma=gamma, maxiter_ADMIN=maxiter_ADMIN, method=("ADMIN",))


# --- Cell 13: DAE / VAE (PIMMS) -- PyTorch-backed,----------
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


# --- Cell 14: method groups, per-matrix applicability, FAST/MEDIUM/SLOW tiers
rng = np.random.default_rng(RNG_SEED)

GENERAL_METHODS = {  # applies to all 3 matrices
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
    R_BACKED_METHODS["DreamAI"] = lambda Y: dreamai_impute(Y)
    R_BACKED_METHODS["ADMIN"] = lambda Y: admin_impute(Y)
DEEP_LEARNING_METHODS = {}  # all 3 matrices, if available
if DEEP_LEARNING_AVAILABLE:
    DEEP_LEARNING_METHODS["DAE"] = lambda Y: dae_impute(Y, n_epochs=200)
    DEEP_LEARNING_METHODS["VAE"] = lambda Y: vae_impute(Y, n_epochs=200)

PROTEIN_MATRICES = {"dlbcl_protein", "crc_protein"}


def methods_for_matrix(matrix_name):
    methods = {**GENERAL_METHODS, **DEEP_LEARNING_METHODS}
    if matrix_name in PROTEIN_MATRICES:
        methods = {**methods, **LEFT_CENSORED_METHODS, **R_BACKED_METHODS}
    return methods


# Tiers -- about intrinsic method cost.
FAST_NAMES = {"Mean", "Median", "MinDet", "MinProb", "KNN", "SVD", "SoftImpute", "Zero", "HalfMin", "LOD", "QRILC", "LLS", "BPCA"}
MEDIUM_NAMES = {"PPCA", "Regression", "MICE", "CF", "MsImpute"}
SLOW_NAMES = {"RandomForest", "MissForest", "EM_MVN", "DreamAI", "ADMIN", "DAE", "VAE"}
SLOW_GRID_MASKING = {0.10, 0.30, 0.50}
SLOW_GRID_MNAR = {0.00, 0.50, 1.00}


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


# --- Cell 15: pilot scenario per matrix - validate + get real timing -------
# Before committing to the full grid: one representative scenario per
# matrix (10% masking, pure MCAR), every applicable method, to see
# real per-method timing on the data before deciding how big a grid to run
#
#
# Checkpointed the same way Cell 16's full grid already is: each (matrix,
# method) result is written to a CSV as soon as it finishes, so an
# interrupted kernel - or a single slow method (BPCA/DreamAI/DAE/VAE are
# the likely ones) when stopped, resumes from there instead of
# re-running every method that already completed. Keyed on the scenario's
# own parameters too, not just matrix/method, so editing PILOT_ROW later
# and re-running is recognized as a different scenario (the old checkpoint
# is then stale and ignored) rather than silently mixing results from two
# different configs.
#
# Also persists each method's completed pilot matrix to disk (not just its
# scalar metrics) - needed by the evaluation/exploration notebook, which
# computes richer diagnostics
# (Spearman r, error-vs-abundance, a truth/masked/imputed visual
# comparison) from the actual completed matrices, not from scalar metrics
# alone. Because of this, the skip condition requires BOTH the
# metrics checkpoint AND the saved matrix file to be present. 
PILOT_ROW = pd.Series({"Masking_Fraction": 0.10, "MNAR_Fraction": 0.0, "Replicate": 1, "Seed": RNG_SEED})
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
        print("pilot checkpoint is for a different PILOT_ROW config -- ignoring it, starting fresh.")
        pilot_rows = []
    else:
        pilot_rows = _checkpoint_df.to_dict("records")
        print(f"resuming pilot from checkpoint: {len(pilot_rows)} (matrix, method) result(s) already done.")
else:
    pilot_rows = []

# Indexed by (matrix, method) -> row position, so a re-run of a previously
# checkpointed-but-not-matrix-saved method UPDATES its row in place instead
# of appending a duplicate.
_pilot_row_index = {(r["Matrix"], r["Method"]): i for i, r in enumerate(pilot_rows)}

for matrix_name, X in MATRICES.items():
    print(f"\n=== pilot: {matrix_name} {X.shape} ===")
    Y_masked, mask, mechanism = build_masked_scenario(PILOT_ROW, X)
    for method_name, fn in methods_for_matrix(matrix_name).items():
        matrix_path = _pilot_matrix_path(matrix_name, method_name)
        if (matrix_name, method_name) in _pilot_row_index and matrix_path.exists():
            print(f"  {method_name:12s} (skipped -- checkpoint + saved matrix both present)")
            continue
        t0 = time.time()
        try:
            X_hat = fn(Y_masked)
            m = calculate_metrics(X, X_hat, mask)
            m.update({"Method": method_name, "Seconds": time.time() - t0})
            X_hat.rename_axis("Feature_ID").to_csv(matrix_path)  # for the evaluation notebook
        except Exception as exc:
            m = {"Method": method_name, "Seconds": time.time() - t0, "RMSE": np.nan, "Error": str(exc)}
            print(f"  !! {method_name} failed: {type(exc).__name__}: {exc}")
        m.update({"Matrix": matrix_name, **{c: PILOT_ROW[c] for c in _PILOT_SCENARIO_COLS}})
        if (matrix_name, method_name) in _pilot_row_index:
            pilot_rows[_pilot_row_index[(matrix_name, method_name)]] = m
        else:
            pilot_rows.append(m)
            _pilot_row_index[(matrix_name, method_name)] = len(pilot_rows) - 1
        print(f"  {method_name:12s} RMSE={m.get('RMSE', float('nan')):.4f}  ({m['Seconds']:.2f}s)")
        pd.DataFrame(pilot_rows).to_csv(PILOT_CHECKPOINT_PATH, index=False)  # checkpoint after every method

pilot_results = pd.DataFrame(pilot_rows)
pilot_summaries = {
    matrix_name: pilot_results[pilot_results["Matrix"] == matrix_name].set_index("Method")
    for matrix_name in MATRICES
}


# --- Cell 16: experiment design ( masking-fraction x
# MNAR-fraction x replicate factorial) + full benchmark loop with checkpointing
MASKING_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50]
MNAR_FRACTIONS = [0.00, 0.25, 0.50, 0.75, 1.00]
N_REPLICATES = 5  # -> 125 scenarios, same size as notebook 04's grid

FULL_GRID_FOR_ALL_METHODS = False  # True runs every method on every scenario (see notebook 04's own
                                    # markdown on this tradeoff) -- a big runtime increase, off by default

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


def run_full_benchmark(matrix_name, X, checkpoint_path):
    if checkpoint_path.exists():
        all_results = pd.read_csv(checkpoint_path).to_dict("records")
        print(f"{matrix_name}: resuming from checkpoint, {len(all_results)} results already done.")
    else:
        all_results = []
    already_done = {(r["Scenario_ID"], r["Method"]) for r in all_results}

    for _, row in experiment_design.iterrows():
        active = applicable_methods(matrix_name, row, full_grid=FULL_GRID_FOR_ALL_METHODS)
        remaining = {n: f for n, f in active.items() if (row["Scenario_ID"], n) not in already_done}
        if not remaining:
            continue

        Y_masked, mask, mechanism = build_masked_scenario(row, X)
        mcar_only = pd.DataFrame(mechanism.values == "MCAR", index=mask.index, columns=mask.columns)
        mnar_only = pd.DataFrame(mechanism.values == "MNAR", index=mask.index, columns=mask.columns)

        for name, fn in remaining.items():
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
                print(f"    !! {matrix_name}/{name} failed on scenario {row['Scenario_ID']}: {exc}")
            all_results.append(rec)
            already_done.add((row["Scenario_ID"], name))
            # Checkpoint after every method - matches Cell 15's pilot-cell
            # granularity, so a kernel restart mid-scenario loses at most one in-flight method's work
            # instead of the whole scenario's. 
            pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(all_results)


# Run one matrix at a time (uncomment below to run)- each call resumes from its
# own checkpoint if interrupted, so it's safe to run matrices in separate
# sessions rather than all at once.
results = {}
for matrix_name, X in MATRICES.items():
    results[matrix_name] = run_full_benchmark(matrix_name, X, RESULTS_DIR / f"benchmark_checkpoint_{matrix_name}.csv")


# --- Cell 17: aggregate + rank (once Cell 16 has been run) ------------------
all_results = pd.concat(
     [df.assign(Matrix=name) for name, df in results.items()], ignore_index=True
 )
all_results.to_csv(RESULTS_DIR / "benchmark_results_full.csv", index=False)

overall_ranking = (
     all_results.groupby(["Matrix", "Method"])["RMSE"]
     .agg(["mean", "std", "count"]).sort_values(["Matrix", "mean"])
 )
overall_ranking
