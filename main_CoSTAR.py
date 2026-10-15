# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""CoSTAR — Fixed / Adaptive-CV / Triplet / NCA."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import dct
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.svm import SVC


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED, K, H, N, NE = 2027, 24, 2, 241, 64
FT, FR, FS, FC, FE = 8, 16, 4, 4, 4
SIGMA, TURNING_SCALE = .5, .20

OUTER, REPEATS, INNER = 5, 5, 3
C_GRID = (10., 100.)
GAMMA_GRID = (.125, .25, .5, 1., 2., 4.)

# Adaptive-CV.
FACTORS = (.5, .75, 1., 1.5, 2.)
CV_PENALTY, CV_PASSES = .0001, 2

# Triplet baseline.
TRIPLET_MARGIN, TRIPLET_TAU, TRIPLET_BETA = .1, .2, .001
TRIPLET_ANCHORS_PER_CLASS, TRIPLETS_PER_ANCHOR, HARD_NEG_K = 40, 4, 10

# NCA — hyperparameters selected by inner CV.
NCA_TAU_GRID = (.05, .075, .10, .15, .30)
NCA_BETA_GRID = (.01, .03, .05, .10)
NCA_ANCHORS_PER_CLASS = 32
NCA_MAXITER = 100

# Shared positive weight constraints.
WEIGHT_BOUNDS = (.5, 2.)
WEIGHT_MAXITER = 200

T, E = np.linspace(0, 1, N), np.linspace(0, 1, NE)

CORE = dict(time=1., arc_q=4., events_arc=4/3, legacy_arc=2/3, speed_t=.25, turn_arc=2.)
EXTRA = dict(turn_time=2., speed_arc=.25, mds_delta=4/3, oldpos=.5)
BASE = {**CORE, **EXTRA}

GROUPS = {
    "core_orientation": ("time", "arc_q"),
    "core_events": ("events_arc", "legacy_arc"),
    "core_dynamics": ("speed_t", "turn_arc"),
    "extra_turn": ("turn_time",),
    "extra_speed": ("speed_arc",),
    "extra_geometry": ("mds_delta", "oldpos")
}

GROUP_NAMES = tuple(GROUPS)
GROUP_BASE_SUMS = np.array([sum(BASE[k] for k in GROUPS[g]) for g in GROUP_NAMES])
ONES = (1.,)*len(GROUPS)

D_CORE = 2*H*(FT+FS+FE) + FE + 1 + FR + FC
D_EXTRA = FC + FR + 8*H + 4
DIM = D_CORE + D_EXTRA

FIX = f"R5-Fixed-{DIM}D"
REG_CV = f"R5-Adaptive-CV-{DIM}D"
REG_TRIPLET = f"R5-Adaptive-Triplet-{DIM}D"
REG_NCA = f"R5-Adaptive-NCA-{DIM}D"

MODELS = (FIX, REG_CV, REG_TRIPLET, REG_NCA)
COLORS = ("lightgray", "steelblue", "darkorange", "crimson")

PREFIX = f"costar_r5_nca_{DIM}D"
FIG, OUT = Path("figures"), Path("results")


# ============================================================
# 1. R5 SYMBOLIC FEATURE EXTRACTION
# ============================================================

def quantize(theta):
    return np.floor((theta+np.pi/K)/(2*np.pi/K)).astype(np.int16) % K


def harmonics(theta):
    return np.stack([f(h*theta)*np.sqrt(2./K) for h in range(1, H+1) for f in (np.cos, np.sin)], axis=1)


def interp_rank(x):

    x = np.asarray(x, float)

    if len(x) == 0: return np.zeros((NE,)+x.shape[1:])
    if len(x) == 1: return np.repeat(x, NE, axis=0)

    xp = np.linspace(0, 1, len(x))

    if x.ndim == 1:
        return np.interp(E, xp, x)

    return np.column_stack([np.interp(E, xp, x[:,j]) for j in range(x.shape[1])])


def event_features(ids):

    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]

    symbols = ids[starts]
    lengths = (ends-starts)/len(ids)
    delta = (symbols[1:].astype(int)-symbols[:-1].astype(int)+K//2) % K-K//2

    def pack(x):
        return dct(interp_rank(x), axis=0, type=2, norm="ortho")[:FE].ravel()

    events = pack(harmonics(2*np.pi*delta/K)) if len(delta) else np.zeros(2*H*FE)
    durations = pack(np.log(np.maximum(len(lengths)*lengths, 1e-12)))
    density = np.array([np.log1p(len(delta))/np.log1p(N-1)])

    return events, np.r_[durations, density], starts, symbols


def encode(case):

    v = np.nan_to_num(np.asarray(case, float)[:2], nan=0., posinf=0., neginf=0.)

    if v.shape[0] != 2 or v.shape[1] < 2:
        raise ValueError("Expected two-channel velocity trajectory")

    n = v.shape[1]
    speed = np.hypot(v[0], v[1])
    positive = speed[speed > 1e-10]

    cutoff = max(.03*np.median(positive), 1e-10) if len(positive) else 1e-10
    theta = np.arctan2(v[1], v[0])
    valid = speed > cutoff

    if valid.any():
        prev = np.where(valid, np.arange(n), -1)
        np.maximum.accumulate(prev, out=prev)
        prev[prev < 0] = np.flatnonzero(valid)[0]
        theta = theta[prev]
    else:
        theta = np.zeros(n)

    # Time parameterization.
    ix = np.minimum((T*n).astype(int), n-1)
    theta_t = theta[ix]
    ids_t = quantize(theta_t)

    ft = dct(harmonics(2*np.pi*ids_t/K), axis=0, type=2, norm="ortho")[:FT].ravel()

    rel = speed[ix]/max(float(speed[ix].mean()), 1e-8)
    speed_t = dct(np.linspace(0,1,4)[np.digitize(rel, (.5,1.,2.))], type=2, norm="ortho")[:FR]

    # Arc-length parameterization.
    s = np.r_[0., np.cumsum(.5*(speed[:-1]+speed[1:]))]

    if s[-1] > 1e-12:
        keep = np.r_[True, np.diff(s) > 1e-12]
        theta_arc = np.interp(T*s[-1], s[keep], np.unwrap(theta)[keep])
        arc_speed = np.interp(T*s[-1], s[keep], speed[keep])
    else:
        theta_arc, arc_speed = theta_t, speed[ix]

    ids_arc = quantize(theta_arc)
    Za = harmonics(2*np.pi*ids_arc/K)

    arc_q = dct(Za, axis=0, type=2, norm="ortho")[:FS].ravel()

    smooth_arc = gaussian_filter1d(theta_arc, sigma=SIGMA, mode="nearest")
    turning_arc = np.tanh(np.gradient(smooth_arc, T)/(TURNING_SCALE*(N-1)))
    turn_arc = dct(np.linspace(-1,1,5)[np.digitize(turning_arc, (-.6,-.2,.2,.6))], type=2, norm="ortho")[:FC]

    events_arc, legacy_arc, starts, symbols = event_features(ids_arc)

    # Supplementary R5 features.
    turn_t_sig = np.tanh(np.gradient(gaussian_filter1d(np.unwrap(theta_t), sigma=SIGMA, mode="nearest"), T)/(TURNING_SCALE*(N-1)))
    turn_time = dct(np.linspace(-1,1,5)[np.digitize(turn_t_sig, (-.6,-.2,.2,.6))], type=2, norm="ortho")[:FC]

    speed_arc_rel = arc_speed/max(float(speed[ix].mean()), 1e-8)
    speed_arc = dct(np.linspace(0,1,4)[np.digitize(speed_arc_rel, (.5,1.,2.))], type=2, norm="ortho")[:FR]

    M = harmonics(2*np.pi*np.arange(K)/K)
    delta = M[symbols[1:]]-M[symbols[:-1]]

    mds_delta = dct(interp_rank(delta), axis=0, type=2, norm="ortho")[:4].ravel() if len(delta) else np.zeros(8*H)

    positions = starts[1:]/(N-1)
    oldpos = dct(interp_rank(positions)-E, type=2, norm="ortho")[:4] if len(positions) else np.zeros(4)

    return dict(
        time=ft, arc_q=arc_q, events_arc=events_arc, legacy_arc=legacy_arc,
        speed_t=speed_t, turn_arc=turn_arc, turn_time=turn_time,
        speed_arc=speed_arc, mds_delta=mds_delta, oldpos=oldpos
    )


def extract(X):

    blocks = [encode(x) for x in X]
    features = {k: np.stack([b[k] for b in blocks]).astype(np.float32) for k in BASE}

    assert sum(features[k].shape[1] for k in CORE) == D_CORE
    assert sum(features[k].shape[1] for k in EXTRA) == D_EXTRA

    return features


# ============================================================
# 2. TRAINING-FOLD DISTANCE NORMALIZATION
# ============================================================

def group_basis(D, it, iv):

    basis = {}

    for group, keys in GROUPS.items():

        A_sum, B_sum = None, None

        for k in keys:

            A = np.asarray(D[k][np.ix_(it,it)], dtype=np.float64)
            B = np.asarray(D[k][np.ix_(iv,it)], dtype=np.float64)

            scale = max(float(A.sum())/(len(it)*(len(it)-1)), 1e-12)
            A, B = BASE[k]*A/scale, BASE[k]*B/scale

            if A_sum is None:
                A_sum, B_sum = A, B
            else:
                A_sum += A
                B_sum += B

        basis[group] = (A_sum, B_sum)

    return basis


def mix(basis, factors):

    lam = np.asarray(factors, float)
    den = float(lam @ GROUP_BASE_SUMS)

    A = sum(lam[j]*basis[g][0] for j,g in enumerate(GROUP_NAMES))/den
    B = sum(lam[j]*basis[g][1] for j,g in enumerate(GROUP_NAMES))/den

    return A, B


def weight_bounds():
    return [(np.log(WEIGHT_BOUNDS[0]), np.log(WEIGHT_BOUNDS[1]))]*len(GROUP_NAMES)


# ============================================================
# 3. CONTRASTIVE TRIPLET BASELINE
# ============================================================

def make_triplets(reference_D, labels, seed):

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))

    triples = []

    for c in np.unique(labels):

        members = indices[labels == c]
        outsiders = indices[labels != c]

        if len(members) < 2 or not len(outsiders):
            continue

        anchors = rng.choice(
            members,
            size=TRIPLET_ANCHORS_PER_CLASS,
            replace=len(members) < TRIPLET_ANCHORS_PER_CLASS
        )

        for i in anchors:

            positives = members[members != i]
            k = min(HARD_NEG_K, len(outsiders))
            hard = outsiders[np.argpartition(reference_D[i, outsiders], k-1)[:k]]

            for _ in range(TRIPLETS_PER_ANCHOR):

                p = int(rng.choice(positives))
                neg = int(rng.choice(hard if rng.random() < .75 else outsiders))
                triples.append((int(i), p, neg))

    if not triples:
        raise ValueError("Insufficient labeled data to construct triplets.")

    return np.asarray(triples, dtype=np.intp)


def learn_triplet(basis, labels, seed):

    reference_D, _ = mix(basis, ONES)
    triplets = make_triplets(reference_D, labels, seed)

    i, p, n = triplets.T

    X = np.column_stack([
        basis[g][0][i,p]-basis[g][0][i,n]
        for g in GROUP_NAMES
    ])

    b = GROUP_BASE_SUMS

    def loss_grad(theta):

        lam = np.exp(theta)
        den = float(lam @ b)

        delta = (X @ lam)/den
        z = (TRIPLET_MARGIN+delta)/TRIPLET_TAU

        loss = np.logaddexp(0., z).mean()+TRIPLET_BETA*(theta @ theta)
        grad = np.mean(expit(z)[:,None]*(X-delta[:,None]*b[None,:])*(lam[None,:]/(den*TRIPLET_TAU)), axis=0)+2*TRIPLET_BETA*theta

        return float(loss), grad

    result = minimize(
        loss_grad, np.zeros(len(GROUP_NAMES)), jac=True,
        method="L-BFGS-B", bounds=weight_bounds(),
        options={"maxiter": WEIGHT_MAXITER, "ftol": 1e-12}
    )

    factors = tuple(map(float, np.exp(result.x)))

    info = {
        "loss": float(result.fun),
        "triplets": int(len(triplets)),
        "success": bool(result.success),
        "iterations": int(result.nit)
    }

    return factors, info


# ============================================================
# 4. NCA — NEIGHBORHOOD COMPONENTS ANALYSIS
# ============================================================

def nca_training_data(basis, labels, seed):

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    # Class-balanced subset of anchors.
    anchors = []

    for c in np.unique(labels):

        members = np.flatnonzero(labels == c)

        if len(members) < 2:
            continue

        k = min(len(members), NCA_ANCHORS_PER_CLASS)
        anchors.extend(rng.choice(members, size=k, replace=False))

    anchors = np.asarray(anchors, dtype=np.intp)

    if len(anchors) == 0:
        raise ValueError("NCA requires at least two samples per represented class.")

    # All training samples remain candidate neighbors.
    A = np.stack([basis[g][0][anchors,:] for g in GROUP_NAMES], axis=0)

    same = labels[anchors,None] == labels[None,:]
    same[np.arange(len(anchors)), anchors] = False

    if not np.all(same.any(axis=1)):
        raise ValueError("Some NCA anchors have no positive neighbors.")

    return anchors, A, same


def learn_nca(basis, labels, seed, tau, beta):

    anchors, A, same = nca_training_data(basis, labels, seed)

    b = np.asarray(GROUP_BASE_SUMS, dtype=float)
    n_anchors = len(anchors)

    def loss_grad(theta):

        lam = np.exp(theta)
        den = float(lam @ b)

        # Fused squared distance from each anchor to each training sample.
        D = np.einsum("g,gij->ij", lam, A, optimize=True)/den

        logits = -D/tau

        # Exclude self-neighbors from the denominator.
        logits[np.arange(n_anchors), anchors] = -np.inf

        positive_logits = np.where(same, logits, -np.inf)

        log_all = logsumexp(logits, axis=1)
        log_pos = logsumexp(positive_logits, axis=1)

        # Negative log probability of a same-class neighbor.
        nca_loss = float(np.mean(log_all-log_pos))
        penalty = float(beta*(theta @ theta))

        # Neighbor probabilities across all candidates and positives.
        P = np.exp(logits-log_all[:,None])
        Q = np.exp(positive_logits-log_pos[:,None])

        # dL/dD = (Q-P)/(M*tau).
        dL_dD = (Q-P)/(n_anchors*tau)

        # Analytical derivatives through distance normalization.
        s = np.einsum("ij,gij->g", dL_dD, A, optimize=True)
        correction = float(np.sum(dL_dD*D))

        grad = (lam/den)*(s-b*correction)+2*beta*theta

        return nca_loss+penalty, grad

    result = minimize(
        loss_grad, np.zeros(len(GROUP_NAMES)), jac=True,
        method="L-BFGS-B", bounds=weight_bounds(),
        options={"maxiter": NCA_MAXITER, "ftol": 1e-10, "gtol": 1e-8}
    )

    factors = tuple(map(float, np.exp(result.x)))
    at_bounds = np.isclose(np.exp(result.x), WEIGHT_BOUNDS[0], atol=1e-4) | np.isclose(np.exp(result.x), WEIGHT_BOUNDS[1], atol=1e-4)

    info = {
        "loss": float(result.fun),
        "tau": float(tau),
        "beta": float(beta),
        "anchors": int(len(anchors)),
        "candidate_neighbors": int(len(labels)),
        "n_at_bounds": int(at_bounds.sum()),
        "success": bool(result.success),
        "iterations": int(result.nit)
    }

    return factors, info


# ============================================================
# 5. SVM EVALUATION ON INNER FOLDS
# ============================================================

def svm_scores(folds, y, factors_by_fold):

    scores = {(C,gamma): [] for C in C_GRID for gamma in GAMMA_GRID}

    for (ia, ib, basis), factors in zip(folds, factors_by_fold):

        A, B = mix(basis, factors)

        for gamma in GAMMA_GRID:

            Ktr, Kva = np.exp(-gamma*A), np.exp(-gamma*B)

            for C in C_GRID:

                clf = SVC(C=C, kernel="precomputed").fit(Ktr, y[ia])
                pred = clf.predict(Kva)

                scores[C,gamma].append(
                    f1_score(y[ib], pred, average="macro", zero_division=0)
                )

    return {k: float(np.mean(v)) for k,v in scores.items()}


def best_svm(scores):
    C, gamma = max(scores, key=lambda k: scores[k])
    return C, gamma, float(scores[C,gamma])


# ============================================================
# 6. INNER CV — FOUR METHODS
# ============================================================

def inner_tune(D, y, it, seed):

    cv = StratifiedKFold(n_splits=INNER, shuffle=True, random_state=seed)
    folds = []

    for a0, b0 in cv.split(np.zeros(len(it)), y[it]):
        ia, ib = it[a0], it[b0]
        folds.append((ia, ib, group_basis(D, ia, ib)))

    # --------------------------------------------------------
    # 6.1 Fixed and adaptive-CV — shared cache.
    # --------------------------------------------------------

    cache = {}

    def evaluate(factors):

        if factors not in cache:
            cache[factors] = svm_scores(folds, y, [factors]*len(folds))

        return cache[factors]

    def best_for(factors):

        C, gamma, score = best_svm(evaluate(factors))
        return factors, C, gamma, score

    def objective(key):

        factors, C, gamma, score = key
        return score-CV_PENALTY*sum(np.log(v)**2 for v in factors)

    fixed = best_for(ONES)
    current = fixed

    for _ in range(CV_PASSES):

        previous = current

        for j in range(len(GROUP_NAMES)):

            candidates = []

            for value in FACTORS:

                factors = list(current[0])
                factors[j] = value

                candidates.append(best_for(tuple(factors)))

            challenger = max(candidates, key=objective)

            if objective(challenger) > objective(current)+1e-12:
                current = challenger

        if current == previous:
            break

    # --------------------------------------------------------
    # 6.2 Triplet learning.
    # --------------------------------------------------------

    triplet_factors = []

    for f, (ia, ib, basis) in enumerate(folds):

        factors, _ = learn_triplet(
            basis, y[ia], seed=seed+10000+f
        )

        triplet_factors.append(factors)

    triplet_scores = svm_scores(folds, y, triplet_factors)
    Ct, gt, st = best_svm(triplet_scores)

    # --------------------------------------------------------
    # 6.3 NCA — inner selection of tau and beta.
    # --------------------------------------------------------

    best_nca = None
    nca_candidates = []

    for tau in NCA_TAU_GRID:
        for beta in NCA_BETA_GRID:

            fold_factors = []

            for f, (ia, ib, basis) in enumerate(folds):

                factors, _ = learn_nca(
                    basis, y[ia], seed=seed+20000+f,
                    tau=tau, beta=beta
                )

                fold_factors.append(factors)

            candidate_scores = svm_scores(folds, y, fold_factors)
            Cn, gn, sn = best_svm(candidate_scores)

            candidate = (float(tau), float(beta), Cn, gn, sn)
            nca_candidates.append(candidate)

            # Prefer the first candidate if scores tie.
            # NCA_BETA_GRID lists stronger regularization first.
            if best_nca is None or sn > best_nca[-1]+1e-12:
                best_nca = candidate

    # --------------------------------------------------------
    # 6.4 Return selections.
    # --------------------------------------------------------

    selected = {
        FIX: fixed,
        REG_CV: current,
        REG_TRIPLET: (Ct, gt, st),
        REG_NCA: best_nca
    }

    print(
        f"Inner | CV patterns={len(cache)} | "
        f"CV={np.round(current[0],3)} | "
        f"Triplet={np.round(np.mean(triplet_factors,axis=0),3)} | "
        f"NCA=(tau={best_nca[0]}, beta={best_nca[1]})",
        flush=True
    )

    return selected, nca_candidates


# ============================================================
# 7. OUTER-FOLD TRAINING AND PREDICTION
# ============================================================

def fit_predict(D, y, it, iv, model, key, seed):

    basis = group_basis(D, it, iv)

    if model in (FIX, REG_CV):

        factors, C, gamma, inner_score = key
        info = {}

    elif model == REG_TRIPLET:

        C, gamma, inner_score = key
        factors, info = learn_triplet(basis, y[it], seed)

    elif model == REG_NCA:

        tau, beta, C, gamma, inner_score = key
        factors, info = learn_nca(basis, y[it], seed, tau, beta)

    else:
        raise ValueError(f"Unknown model: {model}")

    A, B = mix(basis, factors)

    clf = SVC(C=C, kernel="precomputed").fit(np.exp(-gamma*A), y[it])
    pred = clf.predict(np.exp(-gamma*B))

    return pred, factors, C, gamma, inner_score, info


# ============================================================
# 8. FIGURES — PERFORMANCE
# ============================================================

def plot_error_figures(report):

    names = list(report)
    errors = np.asarray([report[k]["errors"] for k in names], float)
    means, stds = errors.mean(axis=1), errors.std(axis=1, ddof=1)

    # Figure 01 — Mean classification errors.
    fig, ax = plt.subplots(figsize=(12, 4.8))

    ax.barh(names, means, xerr=stds, capsize=3, color=COLORS)
    ax.invert_yaxis()

    ax.set(xlabel="Errors per 1422 trajectories", title="R5 — Comparison of metric-learning strategies")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)

    for i, (m, sd) in enumerate(zip(means, stds)):
        ax.text(m+sd+.1, i, f"{m:.2f} ± {sd:.2f}", fontsize=9, va="center")

    ax.set_xlim(right=max(means+stds)+1.5)

    fig.tight_layout()
    fig.savefig(FIG/f"{PREFIX}_01_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 02 — Errors by CV repeat.
    fig, ax = plt.subplots(figsize=(11, 4.5))

    for name, values, color in zip(names, errors, COLORS):
        ax.plot(range(1,REPEATS+1), values, ".-", ms=3.5, lw=1.3, color=color, label=name)

    ax.set(xlabel="CV repeat", ylabel="Classification errors", title="Paired repeated nested cross-validation")
    ax.set_xticks(range(1,REPEATS+1))
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG/f"{PREFIX}_02_repeats.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 9. FIGURES — CLASS ERRORS AND CORRECTIONS
# ============================================================

def plot_red_errors(y, preds, baseline, model, suffix):

    y = np.asarray(y)
    p0, p1 = np.asarray(preds[baseline]), np.asarray(preds[model])
    classes = sorted(np.unique(y), key=lambda z: int(z))

    e0 = np.array([(p0[:,y==c] != c).sum() for c in classes])
    e1 = np.array([(p1[:,y==c] != c).sum() for c in classes])

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(classes))

    ax.bar(x-.18, e0, .36, color="lightgray", label=baseline)
    ax.bar(x+.18, e1, .36, color="crimson", label=model)

    ax.set_xticks(x, classes)
    ax.set(xlabel="True class", ylabel="Errors across five repeats", title="Class-specific classification errors")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG/f"{PREFIX}_{suffix}_class_errors_red.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    corrected = (p0 != y[None,:]) & (p1 == y[None,:])
    introduced = (p0 == y[None,:]) & (p1 != y[None,:])

    wins, losses = corrected.sum(axis=1), introduced.sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(1,REPEATS+1)

    ax.bar(x-.18, wins, .36, color="seagreen", label="Corrected errors")
    ax.bar(x+.18, losses, .36, color="crimson", label="New errors")

    ax.set_xticks(x)
    ax.set(xlabel="CV repeat", ylabel="Number of decisions", title="Corrected and introduced errors")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG/f"{PREFIX}_{suffix}_corrections_red.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "corrected": int(wins.sum()),
        "introduced": int(losses.sum()),
        "corrected_by_repeat": wins.tolist(),
        "introduced_by_repeat": losses.tolist()
    }


# ============================================================
# 10. FIGURES — LEARNED WEIGHT DISTRIBUTIONS
# ============================================================

def plot_factors(parameters, model, suffix):

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for j, (ax, group) in enumerate(zip(axes.flat, GROUP_NAMES)):

        values = np.asarray([v["factors"][j] for v in parameters[model]])

        if model == REG_CV:

            counts = [int(np.isclose(values,f).sum()) for f in FACTORS]

            ax.bar(range(len(FACTORS)), counts, color="crimson")
            ax.set_xticks(range(len(FACTORS)), [str(f) for f in FACTORS])

        else:

            edges = np.linspace(*WEIGHT_BOUNDS, 13)

            ax.hist(values, bins=edges, color="crimson", edgecolor="white")
            ax.axvline(1., color="black", ls="--", lw=1)
            ax.set_xlim(*WEIGHT_BOUNDS)

        ax.set(xlabel="Selected multiplier", ylabel=f"Outer folds / {OUTER*REPEATS}", title=group.replace("_"," ").title())
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG/f"{PREFIX}_{suffix}_selected_factors.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 11. FIGURE — NCA HYPERPARAMETERS
# ============================================================

def plot_nca_hyperparameters(parameters):

    entries = parameters[REG_NCA]
    taus = [v["training_info"]["tau"] for v in entries]
    betas = [v["training_info"]["beta"] for v in entries]

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))

    for ax, values, grid, title in (
        (axs[0], taus, NCA_TAU_GRID, "NCA temperature"),
        (axs[1], betas, NCA_BETA_GRID, "NCA regularization")
    ):

        counts = [sum(np.isclose(v,g) for v in values) for g in grid]

        ax.bar(range(len(grid)), counts, color="crimson")
        ax.set_xticks(range(len(grid)), [str(g) for g in grid])
        ax.set(xlabel="Selected value", ylabel="Outer folds / 25", title=title)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)

    fig.tight_layout()
    fig.savefig(FIG/f"{PREFIX}_nca_hyperparameters.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 12. MAIN
# ============================================================

def main():

    from aeon.datasets import load_classification

    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    X, y = load_classification("CharacterTrajectories", split="train", load_equal_length=False)
    X, y = list(X), np.asarray(y, dtype=str)

    n = len(y)

    assert n == 1422 and len(np.unique(y)) == 20

    print(f"Train={n} | Dim={DIM} | Outer={REPEATS}x{OUTER} | Inner={INNER}")
    print(f"Models={MODELS}")
    print("Extracting R5 symbolic features...", flush=True)

    features = extract(X)

    D = {
        k: squareform(pdist(v, "sqeuclidean")).astype(np.float32)
        for k,v in features.items()
    }

    preds = {name: np.empty((REPEATS,n), dtype=y.dtype) for name in MODELS}
    params = {name: [] for name in MODELS}
    nca_search = []
    seen = np.zeros((REPEATS,n), dtype=np.uint8)

    cv = RepeatedStratifiedKFold(n_splits=OUTER, n_repeats=REPEATS, random_state=SEED)

    print("\n"+"="*95)
    print("REPEATED NESTED TRAIN-ONLY CROSS-VALIDATION")
    print("="*95)

    for step, (it,iv) in enumerate(cv.split(np.zeros(n), y)):

        r, fold = divmod(step, OUTER)
        seen[r,iv] += 1

        selected, candidates = inner_tune(D, y, it, seed=1000+step)
        nca_search.append([list(v) for v in candidates])

        row = []

        for name in MODELS:

            pred, factors, C, gamma, inner_score, info = fit_predict(
                D, y, it, iv, name, selected[name], seed=SEED+50000+step
            )

            preds[name][r,iv] = pred

            denominator = float(np.asarray(factors) @ GROUP_BASE_SUMS)

            effective_weights = {
                k: float(BASE[k]*factors[j])
                for j,g in enumerate(GROUP_NAMES)
                for k in GROUPS[g]
            }

            normalized_weights = {
                k: value/denominator
                for k,value in effective_weights.items()
            }

            params[name].append({
                "repeat": int(r+1),
                "fold": int(fold+1),
                "factors": list(factors),
                "weights": effective_weights,
                "normalized_weights": normalized_weights,
                "C": float(C),
                "gamma": float(gamma),
                "inner_macro_f1": float(inner_score),
                "training_info": info
            })

            row.append(f"{name}: {int((pred != y[iv]).sum())}")

        print(
            f"Repeat {r+1}/{REPEATS} | Fold {fold+1}/{OUTER} | "
            + " | ".join(row),
            flush=True
        )

    assert np.all(seen == 1), "Incomplete out-of-fold predictions."

    # --------------------------------------------------------
    # Aggregate results.
    # --------------------------------------------------------

    report = {}

    print("\n"+"="*95)
    print("AGGREGATED TRAIN-ONLY CV")
    print("="*95)

    for name in MODELS:

        errors = [int((preds[name][r] != y).sum()) for r in range(REPEATS)]
        f1s = [f1_score(y, preds[name][r], average="macro", zero_division=0) for r in range(REPEATS)]

        report[name] = {
            "errors": errors,
            "mean_errors": float(np.mean(errors)),
            "sd_errors": float(np.std(errors, ddof=1)),
            "accuracy": float(1-np.mean(errors)/n),
            "macro_f1": float(np.mean(f1s))
        }

        print(
            f"{name:34s} | Errors={errors} | "
            f"Mean={np.mean(errors):.2f} ± {np.std(errors,ddof=1):.2f} | "
            f"Acc={100*report[name]['accuracy']:.4f}% | "
            f"F1={report[name]['macro_f1']:.5f}"
        )

    # --------------------------------------------------------
    # NCA diagnostics.
    # --------------------------------------------------------

    nca_factors = np.asarray([v["factors"] for v in params[REG_NCA]])
    nca_bounds = np.isclose(nca_factors, WEIGHT_BOUNDS[0], atol=1e-4) | np.isclose(nca_factors, WEIGHT_BOUNDS[1], atol=1e-4)

    print("\nNCA FACTOR DIAGNOSTICS")
    print(f"Mean factors: {np.round(nca_factors.mean(axis=0),3)}")
    print(f"SD factors:   {np.round(nca_factors.std(axis=0,ddof=1),3)}")
    print(f"At bounds:    {nca_bounds.sum(axis=0).tolist()} / {OUTER*REPEATS}")

    # --------------------------------------------------------
    # Figures.
    # --------------------------------------------------------

    plot_error_figures(report)

    corrections = {}

    for name, suffix in (
        (REG_CV, "adaptive_cv"),
        (REG_TRIPLET, "triplet"),
        (REG_NCA, "nca")
    ):

        corrections[name] = plot_red_errors(y, preds, FIX, name, suffix)
        plot_factors(params, name, suffix)

    plot_nca_hyperparameters(params)

    # --------------------------------------------------------
    # Save OOF predictions.
    # --------------------------------------------------------

    np.savez_compressed(
        OUT/f"{PREFIX}_oof.npz",
        y=y,
        pred_fixed=preds[FIX],
        pred_cv=preds[REG_CV],
        pred_triplet=preds[REG_TRIPLET],
        pred_nca=preds[REG_NCA]
    )

    # --------------------------------------------------------
    # Save complete experiment.
    # --------------------------------------------------------

    output = {
        "dataset": "CharacterTrajectories",
        "official_test_used": False,
        "seed": SEED,
        "dimension": DIM,
        "protocol": f"outer {REPEATS}x{OUTER}, inner {INNER}",
        "feature_parameters": {"K": K, "H": H, "N": N, "NE": NE, "FT": FT, "FR": FR, "FS": FS, "FC": FC, "FE": FE},
        "groups": GROUPS,
        "base_weights": BASE,
        "C_grid": C_GRID,
        "gamma_grid": GAMMA_GRID,
        "adaptive_cv": {"factors": FACTORS, "penalty": CV_PENALTY, "passes": CV_PASSES},
        "triplet": {
            "margin": TRIPLET_MARGIN, "tau": TRIPLET_TAU, "beta": TRIPLET_BETA,
            "anchors_per_class": TRIPLET_ANCHORS_PER_CLASS,
            "triplets_per_anchor": TRIPLETS_PER_ANCHOR, "hard_neg_k": HARD_NEG_K
        },
        "nca": {
            "tau_grid": NCA_TAU_GRID,
            "beta_grid": NCA_BETA_GRID,
            "anchors_per_class": NCA_ANCHORS_PER_CLASS,
            "maxiter": NCA_MAXITER
        },
        "weight_bounds": WEIGHT_BOUNDS,
        "report": report,
        "parameters": params,
        "nca_inner_search": nca_search,
        "nca_bounds_per_group": nca_bounds.sum(axis=0).tolist(),
        "corrections": corrections
    }

    (OUT/f"{PREFIX}_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n"+"="*95)
    print("EXPERIMENT FINISHED")
    print("="*95)

    for name in MODELS:
        print(f"{name} | Acc={100*report[name]['accuracy']:.4f}% | Errors={report[name]['errors']}")

    print(f"\nResults: {OUT/f'{PREFIX}_results.json'}")
    print(f"Predictions: {OUT/f'{PREFIX}_oof.npz'}")
    print(f"Figures: {FIG}/{PREFIX}_*.png")


if __name__ == "__main__":
    main()
