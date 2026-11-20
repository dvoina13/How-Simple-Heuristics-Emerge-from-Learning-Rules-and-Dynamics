import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
import pickle
from sklearn.metrics import roc_auc_score

from scipy.signal import fftconvolve
from sklearn.metrics import accuracy_score
from scipy import stats

import argparse

#from tb_logger import TBLogger
import warnings
############ compute CP, noise correlations, and readout weights #################

# helper functions
def _np(a):
    """Tensor/array/list -> numpy array on CPU."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)
 
 
def _as_TN(features):
    """(T, N), (1, T, N), ... -> (T, N)."""
    X = _np(features)
    return X.reshape(-1, X.shape[-1]).astype(float)
 
 
def _zscore_within_stim(X, stim, eps=1e-8):
    """Z-score each column of X within each stimulus condition.
    Returns Z and a dict {stim: per-neuron SD}."""
    Z = np.empty_like(X, dtype=float)
    sds = {}
    for s in np.unique(stim):
        m = stim == s
        sd = X[m].std(axis=0)
        sd = np.where(sd > eps, sd, 1.0)
        Z[m] = (X[m] - X[m].mean(axis=0)) / sd
        sds[s] = sd
    return Z, sds
 

#compute cp function
def compute_cp(weights, bias, features, y, tunings, crit,
               ref_choice=1, neurons=None, include_untuned=False,
               min_per_choice=6, mode="per_stim"):
    """Choice probabilities relative to a FIXED choice (ref_choice), not each
    neuron's preferred stimulus.
 
    CP_k > 0.5  <=>  neuron k fires more on trials where the student chooses ref_choice.
 
    mode="grand"    : z-score each neuron within each stimulus, pool all trials,
                      one AUC per neuron (Britten et al. 1996).
    mode="per_stim" : one AUC per (neuron, stimulus), then nanmean over stimuli.
 
    Returns
    -------
    cp          : (K,) CP per neuron (NaN if too few trials of either choice)
    CP_per_stim : (K, n_stim) per-condition CPs (always computed, for inspection)
    neurons     : (K,) neuron indices, in the same order
    prefs       : (K,) preferred stimulus (0, 1) or -1 for untuned
    choice      : (T,) student's choices
    """
    X = _as_TN(features)
    y = _np(y).ravel().astype(int)
 
    # weight = readout and choice
    w = _np(weights).ravel().astype(float)
    b = _np(bias).ravel()
    b = float(b[0]) if b.size else 0.0
    
    out = X @ w + b
    
    threshold = 0.5 if crit == "MSE" else 0.0
    choice = (out >= threshold).astype(int)
    o = (choice == ref_choice).astype(int)  # 1 = chose the reference stimulus
 
    # which neurons
    stim0, stim1, untuned = (list(map(int, t)) for t in tunings)
    if neurons is None:
        neurons = stim0 + stim1 + (untuned if include_untuned else [])
    neurons = np.asarray(neurons, dtype=int)
    pref_map = {n: 0 for n in stim0} | {n: 1 for n in stim1}
    prefs = np.array([pref_map.get(int(n), -1) for n in neurons])
 
    stims = np.unique(y)
    Z, _ = _zscore_within_stim(X[:, neurons], y)
 
    CP_per_stim = np.full((len(neurons), len(stims)), np.nan)
    for j, s in enumerate(stims):
        m = y == s
        n1, n0 = int(o[m].sum()), int((1 - o[m]).sum())
        if min(n1, n0) < min_per_choice:
            continue
        for i, n in enumerate(neurons):
            CP_per_stim[i, j] = roc_auc_score(o[m], X[m, n])
 
    if mode == "grand":
        cp = np.full(len(neurons), np.nan)
        if min(o.sum(), (1 - o).sum()) >= min_per_choice:
            for i in range(len(neurons)):
                cp[i] = roc_auc_score(o, Z[:, i])
    elif mode == "per_stim":
        with np.errstate(all="ignore"):
            cp = np.nanmean(CP_per_stim, axis=1)
    else:
        raise ValueError("mode must be 'grand' or 'per_stim'")
 
    return cp, CP_per_stim, neurons, prefs, choice
 
 
def compute_noise_corr(features, y, neurons, zclip=None, eps=1e-8):
    """Noise correlation matrix: Pearson correlation of responses after
    z-scoring each neuron within each stimulus condition.
 
    zclip : None (default; right for Gaussian toy data) or a float.  If set,
            values with |z| > zclip are winsorized to ±zclip.  Dropping whole
            trials instead removes ~24% of trials at 100 neurons and biases
            correlations toward 0.
 
    Returns
    -------
    C        : (K, K) noise correlation matrix
    Z        : (T, K) z-scored responses
    sd_pool  : (K,) per-neuron noise SD, averaged over stimulus conditions
    """
    X = _as_TN(features)[:, np.asarray(neurons, dtype=int)]
    stim = _np(y).ravel().astype(int)
 
    Z, sds = _zscore_within_stim(X, stim, eps)
    if zclip is not None:
        Z = np.clip(Z, -zclip, zclip)
 
    C = np.corrcoef(Z, rowvar=False)
    sd_pool = np.mean(np.stack(list(sds.values())), axis=0)
    return C, Z, sd_pool
 
 
def cp_to_rho(cp):
    """Exact inversion of CP = 1/2 + (2/pi) arctan(rho / sqrt(2 - rho^2))
    (Gaussian responses, median split of the decision variable)."""
    return np.sqrt(2.0) * np.sin(np.pi * (np.asarray(cp, float) - 0.5) / 2.0)
 
 
def compute_readout_weights(student_w, student_b, features, y, tunings, crit,
                            ref_choice=1, include_untuned=True,
                            ridge=3.0, min_per_choice=6, mode="per_stim",
                            zclip=None):
    """Infer readout weights from CPs and noise correlations (Haefner et al. 2013):
        rho_k = corr(r_k, d) = (C beta)_k / sqrt(beta^T C beta)
        =>  beta ∝ C^{-1} rho
    CPs are relative to a fixed choice, so no sign flipping is needed.
 
    ridge : the inversion is ill-posed.  Signal in rho lives in the large-eigenvalue
            (shared-noise) directions of C, while CP estimation noise is spread over
            all directions and gets amplified by C^{-1}.  With C a correlation matrix
            (unit diagonal), ridge ~1-10 works well in the toy model; ~0 does not.
    mode  : "per_stim" (default) gives unbiased rho.  "grand" pools z-scored trials
            across stimuli, which shrinks rho by a common factor (~0.75 at ~25% errors):
            fine for the direction of beta, wrong for `explained`.
 
    Returns
    -------
    beta      : (K,) unit-norm weights on z-scored responses (direction only;
                overall scale is not identifiable from CPs)
    w_raw     : (K,) unit-norm weights on raw responses (beta / sd)
    neurons   : (K,) neuron indices used (NaN-CP neurons dropped)
    explained : rho^T C^{-1} rho (unregularized); ~1 if the decision is a linear readout of
                exactly these neurons with no extra decision noise, <1 otherwise
    cp, rho   : CPs and correlations with the decision variable for `neurons`
    """
    cp, _, neurons, _, _ = compute_cp(
        student_w, student_b, features, y, tunings, crit,
        ref_choice=ref_choice, include_untuned=include_untuned,
        min_per_choice=min_per_choice, mode=mode)
 
    ok = np.isfinite(cp)
    if not ok.any():
        warnings.warn(
            "compute_readout_weights: no finite CPs (too few errors per choice); "
            "returning NaN weights.", RuntimeWarning, stacklevel=2)
        neurons_all = np.asarray(neurons)
        nan = np.full(len(neurons_all), np.nan)
        return nan, nan.copy(), neurons_all, np.nan, cp, nan.copy()
        
    neurons, cp = neurons[ok], cp[ok]
    rho = cp_to_rho(cp)
 
    C, _, sd = compute_noise_corr(features, y, neurons, zclip=zclip)
    beta = np.linalg.solve(C + ridge * np.eye(len(rho)), rho)
    # unregularized; biased upward by CP estimation noise (~K * var(rho_hat) / lambda_min(C))
    explained = float(rho @ np.linalg.lstsq(C, rho, rcond=None)[0])
 
    beta = beta / np.linalg.norm(beta)
    w_raw = beta / sd
    w_raw = w_raw / np.linalg.norm(w_raw)
    return beta, w_raw, neurons, explained, cp, rho
 
##############

def compute_d_prime(features, y, eps=1e-8):
    """Per-neuron d' = (m1 - m0) / sqrt((s0^2 + s1^2) / 2).
    Positive = prefers stimulus 1 (same fixed reference as compute_cp with ref_choice=1)."""
    X = _as_TN(features)
    y = _np(y).ravel().astype(int)
    r0, r1 = X[y == 0], X[y == 1]
    den = np.sqrt((r0.var(axis=0) + r1.var(axis=0)) / 2) + eps
    return (r1.mean(axis=0) - r0.mean(axis=0)) / den
 
 
def compute_dprime_cp_correlation(d_prime, cp, neurons, prefs, fold=False,
                                  bins=5, binning="quantile"):
    """Correlate d' with CP, separately for stim-0- and stim-1-preferring neurons.
 
    d_prime : (N,) from compute_d_prime (all neurons)
    cp, neurons, prefs : as returned by compute_cp (cp relative to choice 1)
    fold    : if True, use |d'| and CP relative to the neuron's preferred choice
              (CP_pref = 0.5 + sign(d') * (CP - 0.5)), so both groups share one axis.
              Folding d' alone (old use_abs_dprime) would flip the sign of the
              stim-0 correlation.
    binning : "quantile" (equal counts per bin) or "linear" (equal width), per group.
 
    Returns
    -------
    r      : {pref: (pearson_r, spearman_rho, n)} for pref in (0, 1)
    binned : {pref: dict(d=..., cp=..., cp_sem=..., n=...)} per bin
    """
    from scipy.stats import spearmanr
 
    neurons = np.asarray(neurons, dtype=int)
    prefs = np.asarray(prefs)
    d = np.asarray(d_prime, float)[neurons]          # aligned with cp by neuron index
    c = np.asarray(cp, float).ravel()
    if fold:
        sgn = np.sign(d)
        d, c = np.abs(d), 0.5 + sgn * (c - 0.5)
 
    r, binned = {}, {}
    for p in (0, 1):
        m = (prefs == p) & np.isfinite(d) & np.isfinite(c)
        dp, cp_ = d[m], c[m]
        if m.sum() >= 3 and dp.std() > 0 and cp_.std() > 0:
            r[p] = (np.corrcoef(dp, cp_)[0, 1], spearmanr(dp, cp_)[0], int(m.sum()))
        else:
            r[p] = (np.nan, np.nan, int(m.sum()))
 
        if m.sum() == 0:
            binned[p] = dict(d=[], cp=[], cp_sem=[], n=[])
            continue
        if binning == "quantile":
            edges = np.quantile(dp, np.linspace(0, 1, bins + 1))
        else:
            edges = np.linspace(dp.min(), dp.max(), bins + 1)
            
        idx = np.clip(np.searchsorted(edges, dp, side="right") - 1, 0, bins - 1)
        out = dict(d=[], cp=[], cp_sem=[], n=[])
        for k in range(bins):
            sel = idx == k
            if sel.any():
                out["d"].append(dp[sel].mean())
                out["cp"].append(cp_[sel].mean())
                out["cp_sem"].append(cp_[sel].std(ddof=1) / np.sqrt(sel.sum()) if sel.sum() > 1 else np.nan)
                out["n"].append(int(sel.sum()))
        binned[p] = out
    return r, binned


### find neurons tuned to radial (stim 0); circular (stim 1)
def tuning_stats(features, y, eps=1e-8):
    """Per-unit tuning statistics, vectorized over units.
 
    Parameters
    ----------
    features : (T, N) array/tensor (or (1, T, N) etc.)
    y        : (T,) stimulus labels in {0, 1}; 0 = radial, 1 = circular
 
    Returns
    -------
    dict of (N,) arrays:
      m0, m1   : mean response to stim 0 / stim 1
      t, p     : Welch t statistic (stim1 - stim0) and two-sided p-value
      dprime   : (m1 - m0) / sqrt((var0 + var1) / 2)
      si       : (m1 - m0) / (m1 + m0); NaN where m0 + m1 <= 0
    """
    X = _np(features)
    X = X.reshape(-1, X.shape[-1]).astype(float)
    y = _np(y).ravel().astype(int)
    r0, r1 = X[y == 0], X[y == 1]
 
    m0, m1 = r0.mean(axis=0), r1.mean(axis=0)
    v0, v1 = r0.var(axis=0, ddof=1), r1.var(axis=0, ddof=1)
 
    # Welch's t-test (no equal-variance assumption). Constant units give NaN -> untuned.
    with np.errstate(invalid="ignore", divide="ignore"):
        t, p = stats.ttest_ind(r1, r0, axis=0, equal_var=False)
 
    dprime = (m1 - m0) / (np.sqrt((v0 + v1) / 2) + eps)
 
    denom = m1 + m0
    with np.errstate(invalid="ignore", divide="ignore"):
        si = np.where(denom > eps, (m1 - m0) / denom, np.nan)
 
    return dict(m0=m0, m1=m1, t=t, p=p, dprime=dprime, si=si)
 
 
def find_tuning(features, y, n_neurons=None, method="ttest",
                alpha=0.05, correction=None, min_abs_dprime=0.0,
                dprime_thresh=0.2, si_thresh=0.1,
                seed=None, return_stats=False):
    """Split units into (tuned to stim 0, tuned to stim 1, untuned).
 
    Parameters
    ----------
    features, y   : as in tuning_stats
    n_neurons     : if given, classify a random subset of this many units
                    (None = all units)
    method        : "ttest" | "dprime" | "index"
    alpha         : significance level for "ttest" (two-sided)
    correction    : None | "bonferroni" | "fdr"  (multiple comparisons, "ttest" only)
    min_abs_dprime: extra effect-size requirement for "ttest" (0 = off)
    dprime_thresh : |d'| threshold for "dprime"
    si_thresh     : |SI| threshold for "index"
    seed          : RNG seed for the unit subsample (reproducibility)
    return_stats  : also return the tuning_stats dict and the unit indices examined
 
    Returns
    -------
    neurons_tuned_stim0, neurons_tuned_stim1, neurons_untuned  (lists of unit indices
    into the columns of `features`); plus (stats, units) if return_stats=True.
    """
    st = tuning_stats(features, y)
    n_total = st["t"].shape[0]
 
    # Which units to examine
    rng = np.random.default_rng(seed)
    if n_neurons is None or n_neurons >= n_total:
        units = np.arange(n_total)
    else:
        units = np.sort(rng.choice(n_total, size=n_neurons, replace=False))
 
    # Signed preference score and a boolean "is tuned" mask, per method
    if method == "ttest":
        p = st["p"][units]
        if correction == "bonferroni":
            sig = p < alpha / len(units)
        elif correction == "fdr":
            sig = stats.false_discovery_control(np.nan_to_num(p, nan=1.0)) < alpha
        elif correction is None:
            sig = p < alpha
        else:
            raise ValueError("correction must be None, 'bonferroni' or 'fdr'")
        tuned = sig & (np.abs(st["dprime"][units]) >= min_abs_dprime)
        score = st["t"][units]
    elif method == "dprime":
        score = st["dprime"][units]
        tuned = np.abs(score) >= dprime_thresh
    elif method == "index":
        score = st["si"][units]
        tuned = np.abs(score) >= si_thresh       # NaN compares False -> untuned
    else:
        raise ValueError("method must be 'ttest', 'dprime' or 'index'")
 
    tuned &= np.isfinite(score)
    neurons_tuned_stim0 = units[tuned & (score < 0)].tolist()   # radial
    neurons_tuned_stim1 = units[tuned & (score > 0)].tolist()   # circular
    neurons_untuned = units[~tuned].tolist()
 
    if return_stats:
        return neurons_tuned_stim0, neurons_tuned_stim1, neurons_untuned, st, units
    return neurons_tuned_stim0, neurons_tuned_stim1, neurons_untuned
 

#plot figures
def make_fig(xs, ys, title, xlabl, ylabl):
    fig = plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabl)
    plt.ylabel(ylabl)
    return fig

def make_fig_bar(xs, ys, title, xlabl, ylabl):
    fig = plt.figure()
    plt.bar(xs, ys)
    plt.title(title)
    plt.xlabel(xlabl)
    plt.ylabel(ylabl)
    return fig
    
def plot_weights(teacher_model, student_w, student_b, features, tunings, cp, readout_weights, new_tunings,  r_d_prime_cp, data_d, data_c, name):

    tuned_to_stim0, tuned_to_stim1, untuned = tunings
    t_weights = teacher_model.weight.detach().numpy().flatten()
    s_weights = student_w.detach().numpy().flatten()

    print("s_weights", s_weights)
    n_total = len(t_weights)

    config = {
        "seed": 0,
        "notes": "teacher: BCE, Adam; student: Hebbian learning",
    }

    print("config", config)
    logger = TBLogger(root="runs", run_name=name, config=config)

    fig = plt.figure()
    plt.plot(data_d[0], data_c[0], "b*")
    plt.plot(data_d[1], data_c[1], "ro")
    plt.xlabel("mean d′ in bin")
    plt.ylabel("avg CP")
    logger.log_figure("figures/d_prime_vs_cp", fig, 0)
    plt.show()
    plt.close(fig)
    print("first fig")
    
    fig = plt.figure()
    plt.bar(['stim 0\n neurons', 'stim 1\n neurons'], r_d_prime_cp)
    plt.ylabel("Pearson r(d′, CP)")
    logger.log_figure("figures/d_prime_cp_correl", fig, 0)
    plt.show()
    plt.close(fig)
    print("second fig")
    
    neurons_tuned_to_stim0 = np.where(np.array(new_tunings)==0)[0]; neurons_tuned_to_stim1 = np.where(np.array(new_tunings)==1)[0]; 
    fig = plt.figure()
    plt.bar(['stim 0\n neurons', 'stim 1\n neurons'], [cp[neurons_tuned_to_stim0].mean(), cp[neurons_tuned_to_stim1].mean()])
    plt.ylabel("CP")
    plt.show()
    logger.log_figure("figures/CP_by_neuron_tuning", fig, 0)
    print("third fig")
    logger.flush()
    logger.close()

    fig = plt.figure()
    plt.bar(['stim 0\n neurons', 'stim 1\n neurons'], [readout_weights[neurons_tuned_to_stim0].mean(), readout_weights[neurons_tuned_to_stim1].mean()])
    plt.ylabel("readout_weights")
    plt.show()
    logger.log_figure("figures/readout_weights_by_neuron_tuning", fig, 0)
    print("fourth fig")
    logger.flush()
    logger.close()
    print("Run dir:", logger.run_dir)

    # 5. Visualization    
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(n_total), t_weights)
    plt.title(f"Teacher (Discriminator)\nNeg weights for Stim 2 (red zone)")
    plt.show()
    """
    fig = make_fig_bar(range(n_total), t_weights, "Teacher (Discriminator)\nNeg weights for Stim 2 (red zone)", "weight inds", "teacher weights")
    logger.log_figure("figures/teacher_weights", fig, 0)
    print("fifth fig")
    
    """
    plt.subplot(1, 2, 2)
    plt.bar(range(n_total), s_weights, color='green')
    plt.title("Student (Hebbian Heuristic)\nAll weights positive due to correlation")
    plt.show()
    """
    fig = make_fig_bar(range(n_total), s_weights, "Student (Hebbian Heuristic)\nAll weights positive due to correlation", "weight inds", "student weights")
    logger.log_figure("figures/student_weights", fig, 0)
    print("sixth fig")
    
    print(f"Teacher Bias: {teacher_model.bias.item():.2f}")
    print(f"Student Bias: {student_b.item():.2f}")
        
    fig = plt.figure()
    _ = plt.hist(t_weights)
    plt.title("teacher weights")
    plt.show()
    logger.log_figure("figures/teacher_weights_hist", fig, 0)
    print("seventh fig")

    if not np.isnan(s_weights).any():
        fig = plt.figure()
        _ = plt.hist(s_weights)
        plt.title("student weights")
        plt.show()
        logger.log_figure("figures/student_weights_hist", fig, 0)
    print("eighth fig")
    
    features = features.squeeze()
    fig = plt.figure()
    plt.plot(features[:,tuned_to_stim0].mean(0), t_weights[tuned_to_stim0], "*")
    plt.plot(features[:,tuned_to_stim1].mean(0), t_weights[tuned_to_stim1], "*")
    plt.xlabel("x")
    plt.ylabel("teacher weights")
    plt.show()
    logger.log_figure("figures/teacher_weights_by_neuron_tuning", fig, 0)
    print("ninth fig")
    
    fig = plt.figure()
    plt.plot(features[:,tuned_to_stim0].mean(0), s_weights[tuned_to_stim0], "*")
    plt.plot(features[:,tuned_to_stim1].mean(0), s_weights[tuned_to_stim1], "*")
    plt.xlabel("x")
    plt.ylabel("student weights")
    plt.show()
    logger.log_figure("figures/student_weights_by_neuron_tuning", fig, 0)
    print("tenth fig")
    
    logger.flush()
    logger.close()


#inactivation bias (lesioning experiment)
def _choices(X, w, b, threshold):
    """Student's choices (1 if logit >= threshold, else 0) and logits."""
    logits = X @ w + b
    return (logits >= threshold).astype(int), logits
 
 
def find_bias_simple(p, student_w, student_b, features, y, crit,
                     same_units=False, silence_value=0.0,
                     auto_flip=False, seed=None, verbose=True):
    """Inactivation experiment: silence a fraction p of units, measure accuracy and
    choice fractions of the (fixed) student readout.
 
    Parameters
    ----------
    p             : fraction of units to silence (0 <= p <= 1)
    student_w     : (N,) readout weights;  student_b : bias (scalar or (1,))
    features      : (T, N) or (1, T, N) responses
    y             : (T,) labels in {0, 1}
    crit          : "BCE" (threshold 0 on logits) or "MSE" (threshold 0.5)
    same_units    : False -> a new random set of units per trial (dropout-like);
                    True  -> the same units on every trial (like a real inactivation)
    silence_value : value silenced units are set to. 0.0 = firing goes to zero
                    (removes their baseline too, so the bias depends on the baseline);
                    "mean" = set to each unit's mean over all trials (removes only
                    the stimulus-driven and noise parts).
    auto_flip     : if True, and the intact student is below chance with the standard
                    mapping (logit >= threshold -> class 1), use the flipped mapping.
                    At exactly chance the standard mapping is kept.
    seed          : RNG seed.
 
    Returns
    -------
    acc    : accuracy after inactivation
    frac0  : fraction of trials on which the student chooses class 0
    frac1  : fraction of trials on which the student chooses class 1
    (same order as before: acc, P(choose 0), P(choose 1))
    """

    rng = np.random.default_rng(seed)
 
    X = _np(features)
    X = X.reshape(-1, X.shape[-1])                       # (T, N)
    y = _np(y).ravel().astype(int)
    w = _np(student_w).ravel()
    b = _np(student_b).ravel()
    b = float(b[0]) if b.size else 0.0
 
    if crit == "BCE":
        threshold = 0.0
    elif crit == "MSE":
        threshold = 0.5
    else:
        raise ValueError("crit must be 'BCE' or 'MSE'")
 
    T, N = X.shape
    k = int(round(p * N))

    # --- intact student: fix the label mapping once --------------------------------
    choice_intact, _ = _choices(X, w, b, threshold)
    acc_intact = np.mean(choice_intact == y)
    flip = auto_flip and acc_intact < 0.5               # ties (== 0.5) keep standard mapping
 
    # --- silence units --------------------------------------------------------------
    if same_units:
        silenced = np.zeros((T, N), dtype=bool)
        silenced[:, rng.choice(N, size=k, replace=False)] = True
    else:
        # k distinct random units per trial, vectorized
        order = np.argsort(rng.random((T, N)), axis=1)
        silenced = np.zeros((T, N), dtype=bool)
        np.put_along_axis(silenced, order[:, :k], True, axis=1)
 
    X_sil = X.copy()
    if isinstance(silence_value, str) and silence_value == "mean":
        fill = np.broadcast_to(X.mean(axis=0), X.shape)
        X_sil[silenced] = fill[silenced]
    else:
        X_sil[silenced] = float(silence_value)
 
    # --- inactivated student ---------------------------------------------------------
    choice, _ = _choices(X_sil, w, b, threshold)
    if flip:
        choice = 1 - choice
 
    acc = float(np.mean(choice == y))
    frac1 = float(choice.mean())
    frac0 = 1.0 - frac1

    if verbose:
        print(f"p={p:.2f}: acc={acc:.3f} (intact {max(acc_intact, 1 - acc_intact) if flip else acc_intact:.3f}), "
              f"P(choose 0)={frac0:.3f}, P(choose 1)={frac1:.3f}"
              + ("  [flipped mapping]" if flip else ""))
    return acc, frac0, frac1
 

#test model
def _threshold(crit):
    """Decision threshold on the student's output (w @ x + b)."""
    if crit == "BCE":
        return 0.0
    if crit == "MSE":
        return 0.5
    raise ValueError("crit must be 'BCE' or 'MSE'")
 
 
def orient_weights(weights, bias, X, y, crit="BCE"):
    """Fix the sign of the readout once, so every downstream function can use the
    standard mapping (output >= threshold -> class 1) without flipping.
 
    Needed for unsupervised / Hebbian rules (e.g. Oja), whose weight vector is only
    defined up to sign.  If the student is below chance with the standard mapping,
    return weights whose choices are exactly the flipped ones:
        BCE: (w, b) -> (-w, -b)        since  -(w x + b) >= 0   <=>  w x + b <= 0
        MSE: (w, b) -> (-w, 1 - b)     since  1 - (w x + b) >= 0.5 <=> w x + b <= 0.5
    (identical except for trials exactly at threshold).
 
    Returns w, b (numpy), flipped (bool).
    """
    X = _np(X).reshape(-1, _np(X).shape[-1])
    y = _np(y).ravel().astype(int)
    w = _np(weights).ravel()
    b = _np(bias).ravel()
    b = float(b[0]) if b.size else 0.0
 
    thr = _threshold(crit)
    acc = np.mean(((X @ w + b) >= thr).astype(int) == y)
    if acc >= 0.5:                      # at exactly chance, keep the standard mapping
        return w, b, False
    return -w, (-b if crit == "BCE" else 1.0 - b), True
 
def test_learning_rule(X_test, y_test, student_w, student_b, crit="BCE",
                       auto_flip=False, as_fraction=False, verbose=True):
    """Test accuracy and choice counts of a linear student.
 
    Choice = 1 if w @ x + b >= threshold (0 for BCE, 0.5 for MSE), else 0.
    The old version used threshold = -b on w @ x, which equals the BCE rule
    but is wrong for MSE (should be 0.5 - b).
 
    auto_flip : if the student is below chance, evaluate the flipped readout
                (see orient_weights).  Better: call orient_student once after
                training and pass the oriented weights everywhere
                (compute_cp, find_bias_simple, ...), with auto_flip=False.
 
    Returns
    -------
    acc, n0, n1 : accuracy, number (or fraction) of trials choosing class 0 / class 1
                  (same order as before)
    """
    X = _np(X_test)
    X = X.reshape(-1, X.shape[-1])                       # (T, N)
    y = _np(y_test).ravel().astype(int)
 
    if auto_flip:
        w, b, flipped = orient_weights(student_w, student_b, X, y, crit)
    else:
        w = _np(student_w).ravel()
        b = _np(student_b).ravel()
        b = float(b[0]) if b.size else 0.0
        flipped = False
 
    choice = ((X @ w + b) >= _threshold(crit)).astype(int)
    acc = float(np.mean(choice == y))
    n1 = int(choice.sum())
    n0 = len(choice) - n1
 
    if verbose:
        print(f"accuracy = {acc:.3f}, choose 0: {n0}, choose 1: {n1}"
              + ("  [flipped readout]" if flipped else ""))
    if as_fraction:
        return acc, n0 / len(choice), n1 / len(choice)
    return acc, n0, n1
 
