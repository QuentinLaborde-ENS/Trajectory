# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""CoSTAR — Fixed, adaptive CV and contrastive metric learning."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import dct
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.special import expit

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

# Adaptive-CV configuration.
FACTORS = (.5, .75, 1., 1.5, 2.)
CV_PENALTY = .0001
PASSES = 2

# Adaptive-Contrastive configuration.
TRIPLET_MARGIN = .1
TRIPLET_TAU = .2
TRIPLET_BETA = .001

TRIPLET_ANCHORS_PER_CLASS = 40
TRIPLETS_PER_ANCHOR = 4
HARD_NEG_K = 10
TRIPLET_BOUNDS = (.5, 2.)

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
ONES = (1.,)*len(GROUP_NAMES)

D_CORE = 2*H*(FT+FS+FE) + FE + 1 + FR + FC
D_EXTRA = FC + FR + 8*H + 4
DIM = D_CORE + D_EXTRA

FIX = f"R5-Fixed-{DIM}D"
REG_CV = f"R5-Adaptive-CV-{DIM}D"
REG_TRIPLET = f"R5-Adaptive-Contrastive-{DIM}D"

MODELS = (FIX, REG_CV, REG_TRIPLET)

PREFIX = f"costar_r5_contrastive_{DIM}D"
FIG, OUT = Path("figures"), Path("results")


# ============================================================
# 1. SYMBOLIC FEATURE EXTRACTION — R5
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

    if x.ndim == 1: return np.interp(E, xp, x)

    return np.column_stack([np.interp(E, xp, x[:,j]) for j in range(x.shape[1])])


def event_features(ids):

    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]

    symbols, lengths = ids[starts], (ends-starts)/len(ids)
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
    theta, valid = np.arctan2(v[1], v[0]), speed > cutoff

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

    # Additional R5 components.
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
# 2. GROUPED DISTANCE NORMALIZATION
# ============================================================

def group_basis(D, it, iv):

    basis = {}

    for g, keys in GROUPS.items():

        A_sum, B_sum = None, None

        for k in keys:

            A = D[k][np.ix_(it,it)].astype(np.float64)
            B = D[k][np.ix_(iv,it)].astype(np.float64)

            # Scale estimated exclusively on training observations.
            scale = max(float(A.sum())/(len(it)*(len(it)-1)), 1e-12)

            A, B = BASE[k]*(A/scale), BASE[k]*(B/scale)

            if A_sum is None:
                A_sum, B_sum = A, B
            else:
                A_sum += A
                B_sum += B

        basis[g] = (A_sum, B_sum)

    return basis


def mix(basis, factors):

    lam = np.asarray(factors, dtype=float)
    den = float(lam @ GROUP_BASE_SUMS)

    A = sum(lam[j]*basis[g][0] for j,g in enumerate(GROUP_NAMES))/den
    B = sum(lam[j]*basis[g][1] for j,g in enumerate(GROUP_NAMES))/den

    return A, B


# ============================================================
# 3. TRAINING-ONLY CONTRASTIVE TRIPLET MINING
# ============================================================

def make_triplets(reference_D, labels, seed):

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    triples = []
    n = len(labels)
    all_idx = np.arange(n)

    for c in np.unique(labels):

        members = all_idx[labels == c]
        outsiders = all_idx[labels != c]

        if len(members) < 2 or not len(outsiders):
            continue

        anchors = rng.choice(
            members,
            size=TRIPLET_ANCHORS_PER_CLASS,
            replace=len(members) < TRIPLET_ANCHORS_PER_CLASS
        )

        for i in anchors:

            positives = members[members != i]

            # Hard negatives are mined using the fixed reference distance.
            nh = min(HARD_NEG_K, len(outsiders))
            hard = outsiders[np.argpartition(reference_D[i, outsiders], nh-1)[:nh]]

            for _ in range(TRIPLETS_PER_ANCHOR):

                p = int(rng.choice(positives))
                neg_pool = hard if rng.random() < .75 else outsiders
                nidx = int(rng.choice(neg_pool))

                triples.append((int(i), p, nidx))

    if not triples:
        raise ValueError("Cannot mine triplets: insufficient class membership.")

    return np.asarray(triples, dtype=np.intp)


# ============================================================
# 4. CONTRASTIVE METRIC LEARNING
# ============================================================

def learn_contrastive(basis, labels, seed):

    # Reference distances are built solely from this training fold.
    reference_D, _ = mix(basis, ONES)
    triplets = make_triplets(reference_D, labels, seed)

    i, p, n = triplets.T

    # For each triplet and group:
    # squared distance(anchor, positive) - squared distance(anchor, negative).
    diffs = np.column_stack([
        basis[g][0][i,p] - basis[g][0][i,n]
        for g in GROUP_NAMES
    ])

    b = GROUP_BASE_SUMS

    def loss_grad(theta):

        # Log-parameterization guarantees positive weights.
        lam = np.exp(theta)
        den = lam @ b

        delta = (diffs @ lam)/den
        z = (TRIPLET_MARGIN + delta)/TRIPLET_TAU

        # Stable softplus triplet loss with log-weight regularization.
        loss = float(np.logaddexp(0., z).mean() + TRIPLET_BETA*(theta @ theta))

        # Analytical gradient with respect to log(lambda).
        dz_dtheta = (diffs - delta[:,None]*b[None,:])*(lam[None,:]/(den*TRIPLET_TAU))
        grad = np.mean(expit(z)[:,None]*dz_dtheta, axis=0) + 2*TRIPLET_BETA*theta

        return loss, grad

    bounds = [(np.log(TRIPLET_BOUNDS[0]), np.log(TRIPLET_BOUNDS[1]))]*len(GROUP_NAMES)

    result = minimize(
        loss_grad,
        np.zeros(len(GROUP_NAMES)),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-12}
    )

    factors = tuple(map(float, np.exp(result.x)))

    info = {
        "triplets": int(len(triplets)),
        "loss": float(result.fun),
        "success": bool(result.success)
    }

    return factors, info


# ============================================================
# 5. INNER CV — THREE METHODS
# ============================================================

def inner_tune(D, y, it, seed):

    cv = StratifiedKFold(n_splits=INNER, shuffle=True, random_state=seed)

    folds = []
    contrastive_factors = []

    for fold_index, (a0,b0) in enumerate(cv.split(np.zeros(len(it)), y[it])):

        ia, ib = it[a0], it[b0]
        basis = group_basis(D, ia, ib)

        # Train contrastive weights only on this inner-training fold.
        factors, _ = learn_contrastive(basis, y[ia], seed=seed+10000+fold_index)

        folds.append((ia, ib, basis))
        contrastive_factors.append(factors)

    # Shared cache for Fixed and Adaptive-CV.
    cache = {}

    def evaluate(factors):

        if factors not in cache:

            scores = {(C,gamma): [] for C in C_GRID for gamma in GAMMA_GRID}

            for ia, ib, basis in folds:

                A, B = mix(basis, factors)

                for gamma in GAMMA_GRID:

                    Ktr, Kva = np.exp(-gamma*A), np.exp(-gamma*B)

                    for C in C_GRID:

                        clf = SVC(C=C, kernel="precomputed").fit(Ktr, y[ia])
                        pred = clf.predict(Kva)
                        scores[C,gamma].append(f1_score(y[ib], pred, average="macro", zero_division=0))

            cache[factors] = {k: float(np.mean(v)) for k,v in scores.items()}

        return cache[factors]

    def best_for(factors):

        scores = evaluate(factors)
        C, gamma = max(scores, key=lambda k: scores[k])

        return factors, C, gamma

    def objective(key):

        f, C, gamma = key
        return evaluate(f)[C,gamma] - CV_PENALTY*sum(np.log(x)**2 for x in f)

    # --------------------------------------------------------
    # 5.1 Fixed baseline
    # --------------------------------------------------------

    fixed = best_for(ONES)

    # --------------------------------------------------------
    # 5.2 Adaptive-CV — coordinate descent
    # --------------------------------------------------------

    current = fixed

    for _ in range(PASSES):

        previous = current

        for j in range(len(GROUP_NAMES)):

            candidates = []

            for value in FACTORS:

                f = list(current[0])
                f[j] = value

                candidates.append(best_for(tuple(f)))

            candidate = max(candidates, key=objective)

            if objective(candidate) > objective(current)+1e-12:
                current = candidate

        if current == previous:
            break

    # --------------------------------------------------------
    # 5.3 Adaptive-Contrastive — inner SVM selection
    # --------------------------------------------------------

    # Weights are learned independently on each inner-training fold.
    contrast_scores = {(C,gamma): [] for C in C_GRID for gamma in GAMMA_GRID}

    for (ia, ib, basis), factors in zip(folds, contrastive_factors):

        A, B = mix(basis, factors)

        for gamma in GAMMA_GRID:

            Ktr, Kva = np.exp(-gamma*A), np.exp(-gamma*B)

            for C in C_GRID:

                clf = SVC(C=C, kernel="precomputed").fit(Ktr, y[ia])
                pred = clf.predict(Kva)
                contrast_scores[C,gamma].append(f1_score(y[ib], pred, average="macro", zero_division=0))

    means_contrast = {k: float(np.mean(v)) for k,v in contrast_scores.items()}
    Cc, gammac = max(means_contrast, key=lambda k: means_contrast[k])

    selected = {
        FIX: fixed,
        REG_CV: current,
        REG_TRIPLET: (Cc, gammac)
    }

    scores = {
        FIX: evaluate(fixed[0])[fixed[1:]],
        REG_CV: evaluate(current[0])[current[1:]],
        REG_TRIPLET: means_contrast[Cc,gammac]
    }

    print(
        f"Inner | CV patterns={len(cache)} | CV factors={current[0]} | "
        f"Triplet factor mean={np.mean(contrastive_factors, axis=0).round(3)}",
        flush=True
    )

    return selected, scores


# ============================================================
# 6. FINAL OUTER-FOLD TRAINING
# ============================================================

def fit_predict(D, y, it, iv, model, key, seed):

    basis = group_basis(D, it, iv)

    if model == REG_TRIPLET:

        C, gamma = key

        # Relearn metric on the complete outer-training partition.
        factors, info = learn_contrastive(basis, y[it], seed)

    else:

        factors, C, gamma = key
        info = {}

    A, B = mix(basis, factors)

    clf = SVC(C=C, kernel="precomputed").fit(np.exp(-gamma*A), y[it])
    pred = clf.predict(np.exp(-gamma*B))

    return pred, factors, C, gamma, info


# ============================================================
# 7. FIGURES 01–02 — MODEL COMPARISON
# ============================================================

def plot_error_figures(report):

    names = list(report)
    e = np.array([report[k]["errors"] for k in names], float)

    mu, sd = e.mean(axis=1), e.std(axis=1, ddof=1)
    colors = ("lightgray", "steelblue", "crimson")

    # Figure 01 — Mean errors ± standard deviation.
    fig, ax = plt.subplots(figsize=(11, 4.2))

    ax.barh(names, mu, xerr=sd, capsize=3, color=colors)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(xlabel="Errors per 1422 trajectories", title="R5 — fixed, CV and contrastive weighting")
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)

    for k, (m,s) in enumerate(zip(mu,sd)):
        ax.text(m+s+.1, k, f"{m:.2f} ± {s:.2f}", va="center", fontsize=9)

    ax.set_xlim(right=max(mu+sd)+1.5)
    fig.tight_layout()
    fig.savefig(FIG/f"{PREFIX}_01_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 02 — Errors across repeats.
    fig, ax = plt.subplots(figsize=(11, 4.2))

    for name, values, color in zip(names, e, colors):
        ax.plot(range(1, REPEATS+1), values, ".-", lw=1.3, ms=4, color=color, label=name)

    ax.set(xlabel="CV repeat", ylabel="Classification errors", title="Paired repeated nested CV")
    ax.set_xticks(range(1, REPEATS+1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG/f"{PREFIX}_02_repeats.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. RED ERROR FIGURES
# ============================================================

def plot_red_errors(y, preds, baseline, model, suffix):

    y = np.asarray(y)
    p0, p1 = np.asarray(preds[baseline]), np.asarray(preds[model])
    classes = sorted(np.unique(y), key=lambda z: int(z))

    e0 = np.array([(p0[:,y==c] != c).sum() for c in classes])
    e1 = np.array([(p1[:,y==c] != c).sum() for c in classes])

    # Errors by class.
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(classes))

    ax.bar(x-.18, e0, .36, color="lightgray", label=baseline)
    ax.bar(x+.18, e1, .36, color="crimson", label=model)

    ax.set_xticks(x, classes)
    ax.set(xlabel="True class", ylabel="Errors across five repeats", title="Class-specific classification errors")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG/f"{PREFIX}_{suffix}_class_errors_red.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Corrected versus introduced errors.
    corrected = (p0 != y[None,:]) & (p1 == y[None,:])
    introduced = (p0 == y[None,:]) & (p1 != y[None,:])

    wins, losses = corrected.sum(axis=1), introduced.sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(1, REPEATS+1)

    ax.bar(x-.18, wins, .36, color="seagreen", label="Corrected errors")
    ax.bar(x+.18, losses, .36, color="crimson", label="New errors")

    ax.set_xticks(x)
    ax.set(xlabel="CV repeat", ylabel="Number of decisions", title="Corrected and introduced errors")
    ax.spines[["top", "right"]].set_visible(False)
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
# 9. WEIGHT DISTRIBUTIONS
# ============================================================

def plot_factors(parameters, model, suffix):

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for j, (ax,g) in enumerate(zip(axes.flat, GROUP_NAMES)):

        values = np.asarray([v["factors"][j] for v in parameters[model]])

        if model == REG_CV:

            counts = [int(np.isclose(values,f).sum()) for f in FACTORS]
            ax.bar(np.arange(len(FACTORS)), counts, color="crimson")
            ax.set_xticks(np.arange(len(FACTORS)), [str(v) for v in FACTORS])

        else:

            ax.hist(values, bins=np.linspace(TRIPLET_BOUNDS[0], TRIPLET_BOUNDS[1], 13), color="crimson", edgecolor="white")
            ax.set_xlim(*TRIPLET_BOUNDS)
            ax.axvline(1., color="black", lw=1, ls="--")

        ax.set(xlabel="Selected multiplier", ylabel=f"Outer folds / {OUTER*REPEATS}", title=g.replace("_"," ").title())
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG/f"{PREFIX}_{suffix}_selected_factors.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 10. MAIN — REPEATED NESTED CROSS-VALIDATION
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
    print(f"Models={MODELS}", flush=True)

    features = extract(X)

    D = {
        k: squareform(pdist(v, "sqeuclidean")).astype(np.float32)
        for k,v in features.items()
    }

    preds = {name: np.empty((REPEATS,n), dtype=y.dtype) for name in MODELS}
    params = {name: [] for name in MODELS}
    seen = np.zeros((REPEATS,n), dtype=np.uint8)

    cv = RepeatedStratifiedKFold(n_splits=OUTER, n_repeats=REPEATS, random_state=SEED)

    print("\n"+"="*90)
    print("REPEATED NESTED TRAIN-ONLY CV")
    print("="*90)

    for step, (it,iv) in enumerate(cv.split(np.zeros(n), y)):

        r, fold = divmod(step, OUTER)
        seen[r,iv] += 1

        selected, scores = inner_tune(D, y, it, seed=1000+step)
        row = []

        for name in MODELS:

            pred, factors, C, gamma, info = fit_predict(
                D, y, it, iv, name, selected[name], seed=SEED+50000+step
            )

            preds[name][r,iv] = pred

            weights = {
                k: float(BASE[k]*factors[j])
                for j,g in enumerate(GROUP_NAMES)
                for k in GROUPS[g]
            }

            params[name].append({
                "factors": list(factors),
                "weights": weights,
                "C": float(C),
                "gamma": float(gamma),
                "inner_macro_f1": float(scores[name]),
                "triplet_info": info
            })

            row.append(f"{name}: {int((pred != y[iv]).sum())}")

        print(
            f"Repeat {r+1}/{REPEATS} | Fold {fold+1}/{OUTER} | "
            + " | ".join(row), flush=True
        )

    assert np.all(seen == 1), "Incomplete out-of-fold predictions."

    # --------------------------------------------------------
    # Aggregate results.
    # --------------------------------------------------------

    report = {}

    print("\n"+"="*90)
    print("AGGREGATED TRAIN-ONLY CV")
    print("="*90)

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
            f"{name} | Errors={errors} | "
            f"Acc={100*report[name]['accuracy']:.4f}% | "
            f"F1={report[name]['macro_f1']:.5f}"
        )

    # --------------------------------------------------------
    # Figures.
    # --------------------------------------------------------

    plot_error_figures(report)

    corrections = {
        REG_CV: plot_red_errors(y, preds, FIX, REG_CV, "adaptive_cv"),
        REG_TRIPLET: plot_red_errors(y, preds, FIX, REG_TRIPLET, "contrastive")
    }

    plot_factors(params, REG_CV, "adaptive_cv")
    plot_factors(params, REG_TRIPLET, "contrastive")

    # --------------------------------------------------------
    # Save predictions.
    # --------------------------------------------------------

    np.savez_compressed(
        OUT/f"{PREFIX}_oof.npz",
        y=y,
        pred_fixed=preds[FIX],
        pred_adaptive_cv=preds[REG_CV],
        pred_contrastive=preds[REG_TRIPLET]
    )

    # --------------------------------------------------------
    # Save complete report.
    # --------------------------------------------------------

    output = {
        "dataset": "CharacterTrajectories",
        "official_test_used": False,
        "dimension": DIM,
        "protocol": f"outer {REPEATS}x{OUTER}, inner {INNER}",
        "seed": SEED,
        "model_order": MODELS,
        "groups": GROUPS,
        "base_weights": BASE,
        "C_grid": C_GRID,
        "gamma_grid": GAMMA_GRID,
        "CV_factors": FACTORS,
        "CV_penalty": CV_PENALTY,
        "CV_passes": PASSES,
        "triplet": {
            "margin": TRIPLET_MARGIN,
            "tau": TRIPLET_TAU,
            "beta": TRIPLET_BETA,
            "bounds": TRIPLET_BOUNDS,
            "anchors_per_class": TRIPLET_ANCHORS_PER_CLASS,
            "triplets_per_anchor": TRIPLETS_PER_ANCHOR,
            "hard_neg_k": HARD_NEG_K
        },
        "report": report,
        "parameters": params,
        "corrections": corrections
    }

    (OUT/f"{PREFIX}_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nResults: {OUT/f'{PREFIX}_results.json'}")
    print(f"Figures: {FIG}/{PREFIX}_*.png")


if __name__ == "__main__":
    main()
