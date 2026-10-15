# -*- coding: utf-8 -*-


import json
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import dct
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC


# ============================================================
# HASTE-AL-CS — R5 WEIGHTING CONFIRMATION
# ============================================================

SEED, K, H, N, NE = 2027, 24, 2, 241, 64
FT, FR, FS, FC, FE = 8, 16, 4, 4, 4
SIGMA, TURNING_SCALE = .5, .20

OUTER, REPEATS, INNER = 5, 5, 3
C_GRID = (10., 100.)
GAMMA_GRID = (.125, .25, .5, 1., 2., 4.)

FACTORS = (.5, 1., 1.5)
PENALTY = .0005
PENALTY_SWEEP = (0., .0001, .00025, .0005, .001, .002)

T = np.linspace(0, 1, N)
E = np.linspace(0, 1, NE)

FIG, OUT = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# Historical blocks kept unchanged.
HIST = dict(time=1., speed_c=.25, arc_mix=4., curv_arc_c=2., events_arc=4/3, legacy_arc=2/3)

# S3 symbolic core.
CORE = dict(time=1., arc_q=4., events_arc=4/3, legacy_arc=2/3, speed_t=.25, turn_arc=2.)

# Four complementary symbolic blocks in R5.
EXTRA = dict(turn_time=2., speed_arc=.25, mds_delta=4/3, oldpos=.5)
EXTRA_KEYS = tuple(EXTRA)

REF = "R5-Reference-129D"
FIX = "R5-Fixed-129D"
REG = "R5-3Group-Reg-129D"
NAMES = [REF, FIX, REG]

REF_PATTERNS = [(.5,)*4, (1.5,)*4]
FIX_PATTERNS = [(1.,)*4]
GROUP3_PATTERNS = [(a, b, c, c) for a, b, c in product(FACTORS, repeat=3)]

# Shared candidate calculations.
ALL_PATTERNS = list(dict.fromkeys(REF_PATTERNS + FIX_PATTERNS + GROUP3_PATTERNS))


# ============================================================
# 1. SYMBOLIC FEATURE EXTRACTION
# ============================================================

def quantize(theta):
    return np.floor((theta + np.pi/K)/(2*np.pi/K)).astype(np.int16) % K


def harmonics(theta):
    return np.stack([f(h*theta)*np.sqrt(2./K) for h in range(1, H+1) for f in (np.cos, np.sin)], axis=1)


def interp_rank(x):
    x = np.asarray(x, float)

    if len(x) == 0:
        return np.zeros((NE,) + x.shape[1:])

    if len(x) == 1:
        return np.repeat(x, NE, axis=0)

    xp = np.linspace(0, 1, len(x))

    if x.ndim == 1:
        return np.interp(E, xp, x)

    return np.column_stack([np.interp(E, xp, x[:, j]) for j in range(x.shape[1])])


def event_features(ids):
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]

    symbols = ids[starts]
    lengths = (ends-starts)/len(ids)
    delta = (symbols[1:].astype(int)-symbols[:-1].astype(int)+K//2) % K-K//2

    def pack(x):
        return dct(interp_rank(x), axis=0, type=2, norm="ortho")[:FE].ravel()

    events = pack(harmonics(2*np.pi*delta/K)) if len(delta) else np.zeros(16)
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
    speed_c = dct(rel, type=2, norm="ortho")[:FR]
    speed_t = dct(np.linspace(0, 1, 4)[np.digitize(rel, (.5, 1., 2.))], type=2, norm="ortho")[:FR]

    # Arc-length parameterization.
    s = np.r_[0., np.cumsum(.5*(speed[:-1]+speed[1:]))]

    if s[-1] > 1e-12:
        keep = np.r_[True, np.diff(s) > 1e-12]

        theta_arc = np.interp(T*s[-1], s[keep], np.unwrap(theta)[keep])
        arc_speed = np.interp(T*s[-1], s[keep], speed[keep])
    else:
        theta_arc = theta_t
        arc_speed = speed[ix]

    ids_arc = quantize(theta_arc)
    Za = harmonics(2*np.pi*ids_arc/K)

    arc_mix = dct(.5*Za+.5*harmonics(theta_arc), axis=0, type=2, norm="ortho")[:FS].ravel()
    arc_q = dct(Za, axis=0, type=2, norm="ortho")[:FS].ravel()

    smooth_arc = gaussian_filter1d(theta_arc, sigma=SIGMA, mode="nearest")
    turning_arc = np.tanh(np.gradient(smooth_arc, T)/(TURNING_SCALE*(N-1)))

    curv_arc_c = dct(turning_arc, type=2, norm="ortho")[:FC]
    turn_arc = dct(np.linspace(-1, 1, 5)[np.digitize(turning_arc, (-.6, -.2, .2, .6))], type=2, norm="ortho")[:FC]

    events_arc, legacy_arc, starts, symbols = event_features(ids_arc)

    # Additional R5 symbolic branches.
    turn_t_sig = np.tanh(
        np.gradient(
            gaussian_filter1d(np.unwrap(theta_t), sigma=SIGMA, mode="nearest"), T
        )/(TURNING_SCALE*(N-1))
    )

    turn_time = dct(np.linspace(-1, 1, 5)[np.digitize(turn_t_sig, (-.6, -.2, .2, .6))], type=2, norm="ortho")[:FC]

    speed_arc_rel = arc_speed/max(float(speed[ix].mean()), 1e-8)
    speed_arc = dct(np.linspace(0, 1, 4)[np.digitize(speed_arc_rel, (.5, 1., 2.))], type=2, norm="ortho")[:FR]

    # MDS4-equivalent harmonic embedding.
    M = harmonics(2*np.pi*np.arange(K)/K)
    delta = M[symbols[1:]]-M[symbols[:-1]]

    mds_delta = dct(interp_rank(delta), axis=0, type=2, norm="ortho")[:4].ravel() if len(delta) else np.zeros(16)

    positions = starts[1:]/(N-1)
    oldpos = dct(interp_rank(positions)-E, type=2, norm="ortho")[:4] if len(positions) else np.zeros(4)

    return dict(
        time=ft, speed_c=speed_c, arc_mix=arc_mix, curv_arc_c=curv_arc_c,
        events_arc=events_arc, legacy_arc=legacy_arc, arc_q=arc_q,
        speed_t=speed_t, turn_arc=turn_arc, turn_time=turn_time,
        speed_arc=speed_arc, mds_delta=mds_delta, oldpos=oldpos
    )


def extract(X):
    blocks = [encode(x) for x in X]
    features = {k: np.stack([b[k] for b in blocks]).astype(np.float32) for k in blocks[0]}

    assert sum(features[k].shape[1] for k in HIST) == 89
    assert sum(features[k].shape[1] for k in CORE) == 89
    assert sum(features[k].shape[1] for k in CORE) + sum(features[k].shape[1] for k in EXTRA) == 129

    return features


# ============================================================
# 2. DISTANCE NORMALIZATION
# ============================================================

def pair_basis(D, it, iv):
    basis = {}

    for key, matrix in D.items():
        A = matrix[np.ix_(it, it)]
        scale = max(float(A.sum())/(len(it)*(len(it)-1)), 1e-12)

        basis[key] = (
            np.asarray(A, dtype=np.float64)/scale,
            np.asarray(matrix[np.ix_(iv, it)], dtype=np.float64)/scale
        )

    return basis


def mix(basis, weights):
    total = sum(weights.values())
    return tuple(sum(w*basis[k][j] for k, w in weights.items())/total for j in (0, 1))


def r5_weights(factors):
    return {**CORE, **{k: EXTRA[k]*f for k, f in zip(EXTRA_KEYS, factors)}}


def penalty_cost(factors, penalty):
    return penalty*sum(np.log(v)**2 for v in factors[:3])


# ============================================================
# 3. INNER CV — WEIGHTS, C AND GAMMA
# ============================================================

def inner_tune(D, y, it, seed):
    cv = StratifiedKFold(n_splits=INNER, shuffle=True, random_state=seed)

    score = {
        (p, C, g): []
        for p in ALL_PATTERNS
        for C in C_GRID
        for g in GAMMA_GRID
    }

    for ia0, ib0 in cv.split(np.zeros(len(it)), y[it]):
        ia, ib = it[ia0], it[ib0]
        basis = pair_basis(D, ia, ib)

        for factors in ALL_PATTERNS:
            A, B = mix(basis, r5_weights(factors))

            for gamma in GAMMA_GRID:
                Ktr, Kva = np.exp(-gamma*A), np.exp(-gamma*B)

                for C in C_GRID:
                    clf = SVC(C=C, kernel="precomputed").fit(Ktr, y[ia])
                    yp = clf.predict(Kva)

                    score[factors, C, gamma].append(
                        f1_score(y[ib], yp, average="macro", zero_division=0)
                    )

    means = {k: float(np.mean(v)) for k, v in score.items()}

    def winner(patterns, penalty=0.):
        keys = [(p, C, g) for p in patterns for C in C_GRID for g in GAMMA_GRID]
        return max(keys, key=lambda k: means[k]-penalty_cost(k[0], penalty))

    selected = {
        REF: winner(REF_PATTERNS),
        FIX: winner(FIX_PATTERNS),
        REG: winner(GROUP3_PATTERNS, PENALTY)
    }

    sensitivity = {}

    for pen in PENALTY_SWEEP:
        key = winner(GROUP3_PATTERNS, pen)
        factors, C, gamma = key

        sensitivity[str(pen)] = dict(
            factors=list(factors), C=float(C), gamma=float(gamma),
            inner_f1=float(means[key]),
            penalized_objective=float(means[key]-penalty_cost(factors, pen))
        )

    return selected, means, sensitivity


# ============================================================
# 4. SVM MARGINS
# ============================================================

def signed_ovo_true_margin(clf, Kva, yt):
    # Positive OVO score favors the lower-indexed class.
    ovo = clf.decision_function(Kva)
    classes = clf.classes_
    nclass = len(classes)
    ci = {label: j for j, label in enumerate(classes)}

    positions = {}
    h = 0

    for a in range(nclass):
        for b in range(a+1, nclass):
            positions[a, b] = h
            h += 1

    signed = np.empty((len(yt), nclass), float)

    for i, label in enumerate(yt):
        a = ci[label]

        for b in range(nclass):
            if a == b:
                signed[i, b] = np.inf
            elif a < b:
                signed[i, b] = ovo[i, positions[a, b]]
            else:
                signed[i, b] = -ovo[i, positions[b, a]]

    worst = signed.min(axis=1)

    # One-vs-rest aggregation of SVC one-vs-one scores.
    clf.decision_function_shape = "ovr"
    ovr = clf.decision_function(Kva)

    ix = np.array([ci[z] for z in yt])
    tscore = ovr[np.arange(len(yt)), ix].copy()

    ovr[np.arange(len(yt)), ix] = -np.inf
    ovr_gap = tscore-ovr.max(axis=1)

    return worst, ovr_gap


def fit_predict(D, y, it, iv, key):
    factors, C, gamma = key
    basis = pair_basis(D, it, iv)
    A, B = mix(basis, r5_weights(factors))

    clf = SVC(
        C=C, kernel="precomputed", decision_function_shape="ovo"
    ).fit(np.exp(-gamma*A), y[it])

    Kva = np.exp(-gamma*B)
    yp = clf.predict(Kva)

    worst, ovr_gap = signed_ovo_true_margin(clf, Kva, y[iv])

    return yp, B, worst, ovr_gap


# ============================================================
# 5. FIGURE HELPERS
# ============================================================

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG/f"haste_weightcheck_{name}.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def trajectory(ax, case, title):
    xy = np.cumsum(np.nan_to_num(np.asarray(case, float)[:2]).T, axis=0)
    xy -= xy[0]
    xy /= max(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 1e-12)

    ax.plot(xy[:, 0], xy[:, 1], lw=1.2)
    ax.scatter(*xy[0], s=12)
    ax.scatter(*xy[-1], marker="x", s=24)

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def mds_oos(A, B):
    row = A.mean(axis=1)
    grand = A.mean()

    G = -.5*(A-row[:, None]-row[None, :]+grand)
    values, U = eigh(G, subset_by_index=(len(A)-2, len(A)-1), check_finite=False)

    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 1e-12)
    U = U[:, order]

    train = U*np.sqrt(values)
    val = (-.5*(B-B.mean(axis=1, keepdims=True)-row[None, :]+grand))@U/np.sqrt(values)

    return train, val


# ============================================================
# 6. ALL DIAGNOSTIC FIGURES
# ============================================================

def figures(X, y, D, pred, report, params, sensitivity, svm_ovo, svm_ovr, neigh, ratios, first):
    classes = np.array(sorted(np.unique(y), key=int))
    errors = np.array([report[k]["errors"] for k in NAMES])
    avg = errors.mean(axis=1)
    std = errors.std(axis=1, ddof=1)

    # 01 — Main ablation.
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.barh(NAMES, avg, xerr=std, capsize=3)
    ax.invert_yaxis()
    ax.set(xlabel="Errors / 1422 trajectories", title="Predefined R5 weighting comparison — train-only nested CV")
    save(fig, "01_ablation")

    # 02 — Repeated CV.
    fig, ax = plt.subplots(figsize=(11, 4))

    for k, e in zip(NAMES, errors):
        ax.plot(range(1, REPEATS+1), e, ".-", label=k)

    ax.set_xticks(range(1, REPEATS+1))
    ax.set(xlabel="Outer CV repeat", ylabel="Errors", title="Paired repeated CV")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "02_repeats")

    # 03 — Confusion matrices.
    fig, axs = plt.subplots(1, len(NAMES), figsize=(14, 4))

    for ax, k in zip(axs, NAMES):
        cm = confusion_matrix(np.tile(y, REPEATS), pred[k].ravel(), labels=classes)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6)
        ax.set_yticks(range(len(classes)), classes, fontsize=6)
        ax.set_title(k, fontsize=9)

    save(fig, "03_confusions")

    # 04 — Class errors.
    fig, ax = plt.subplots(figsize=(12, 4))
    xx = np.arange(len(classes))
    width = .24

    for j, k in enumerate(NAMES):
        counts = [int((pred[k][:, y == c] != c).sum()) for c in classes]
        ax.bar(xx+(j-1)*width, counts, width, label=k)

    ax.set_xticks(xx, classes)
    ax.set(xlabel="True class", ylabel="Errors across 5 repeats", title="Class-specific confusion persistence")
    ax.legend(fontsize=8)
    save(fig, "04_class_errors")

    # 05 — Jaccard error overlap.
    masks = [(pred[k] != y[None, :]).ravel() for k in NAMES]
    J = np.eye(len(NAMES))

    for a in range(len(NAMES)):
        for b in range(a+1, len(NAMES)):
            common = np.logical_and(masks[a], masks[b]).sum()
            union = np.logical_or(masks[a], masks[b]).sum()
            J[a, b] = J[b, a] = common/max(union, 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(NAMES)), NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(NAMES)), NAMES, fontsize=9)
    fig.colorbar(im, ax=ax)
    ax.set_title("Paired error overlap — Jaccard")
    save(fig, "05_jaccard")

    # 06 — Selected weights, reference vs regularized.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, k in zip(axs, (REF, REG)):
        selected = np.asarray([p["factors"] for p in params[k]])

        for col, key in enumerate(EXTRA_KEYS):
            ax.plot(np.arange(1, 26), selected[:, col], ".-", lw=1, ms=3, label=key)

        ax.set(title=k, xlabel="Outer fold", ylabel="Multiplier", ylim=(.35, 1.65))
        ax.legend(fontsize=8, ncol=2)

    save(fig, "06_weights")

    # 07 — Penalty sensitivity, inner selection only.
    ps = np.asarray(PENALTY_SWEEP)
    freq, deviations, agreement = [], [], []

    baseline = [
        tuple(sensitivity[t][str(PENALTY)]["factors"])
        for t in range(25)
    ]

    for p in ps:
        choices = [
            tuple(sensitivity[t][str(p)]["factors"])
            for t in range(25)
        ]

        freq.append(sum(x == (1., 1., 1., 1.) for x in choices))
        deviations.append(np.mean([sum(abs(np.log(z)) for z in x[:3]) for x in choices]))
        agreement.append(sum(x == z for x, z in zip(choices, baseline)))

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    series = [
        (freq, "Exactly unit weights", "Folds / 25"),
        (deviations, "Mean absolute log-weight", "Mean Σ|log λ|"),
        (agreement, "Agreement with penalty=0.0005", "Folds / 25")
    ]

    for ax, (v, title, ylabel) in zip(axs, series):
        ax.plot(ps, v, "o-")
        ax.axvline(PENALTY, ls="--", color="gray")
        ax.set(xlabel="Inner penalty", ylabel=ylabel, title=title)

    save(fig, "07_penalty_sensitivity")

    # 08 — Corrected vs newly introduced errors.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    for ax, baseline_name in zip(axs, [REF, FIX]):
        variants = [k for k in NAMES if k != baseline_name]

        for j, k in enumerate(variants):
            a = pred[baseline_name] != y[None, :]
            b = pred[k] != y[None, :]

            corrected = np.logical_and(a, ~b).sum()
            new = np.logical_and(~a, b).sum()

            ax.bar(j-.18, corrected, .36, color="tab:blue")
            ax.bar(j+.18, new, .36, color="tab:orange")

        ax.set_xticks(
            np.arange(len(variants)),
            [s.replace("R5-", "") for s in variants],
            rotation=25, ha="right", fontsize=8
        )
        ax.set(title=f"Compared with {baseline_name}", ylabel="OOF decisions over five repeats")

    axs[0].bar([], [], color="tab:blue", label="Corrected")
    axs[0].bar([], [], color="tab:orange", label="New errors")
    axs[0].legend(fontsize=8)
    save(fig, "08_corrections")

    # 09 — SVM decision-score diagnostics.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))

    a = svm_ovr[REF].ravel()
    b = svm_ovr[REG].ravel()

    err0 = (pred[REF] != y[None, :]).ravel()
    err1 = (pred[REG] != y[None, :]).ravel()

    axs[0].scatter(a, b, s=5, alpha=.15, color="gray", rasterized=True)

    axs[0].scatter(
        a[err0 & ~err1], b[err0 & ~err1],
        marker="o", s=45, facecolors="none",
        edgecolors="green", label="Corrected"
    )

    axs[0].scatter(
        a[~err0 & err1], b[~err0 & err1],
        marker="x", s=45, c="red", label="New error"
    )

    axs[0].axhline(0, color="gray", ls="--")
    axs[0].axvline(0, color="gray", ls="--")
    axs[0].set(
        xlabel="R5 reference — true-class OVR score gap",
        ylabel="R5 regularized — true-class OVR score gap",
        title="True class vs strongest competing class"
    )
    axs[0].legend(fontsize=8)

    wrong0 = svm_ovo[REF][pred[REF] != y[None, :]]
    wrong1 = svm_ovo[REG][pred[REG] != y[None, :]]

    axs[1].hist(wrong0, bins=20, alpha=.5, label=REF)
    axs[1].hist(wrong1, bins=20, alpha=.5, label=REG)
    axs[1].set(
        xlabel="Minimum signed one-vs-one score for true class",
        ylabel="Incorrect OOF decisions",
        title="SVM pairwise decision scores"
    )
    axs[1].legend(fontsize=8)
    save(fig, "09_svm_margins")

    # 10 — Persistent errors and nearest training neighbors.
    persistence = (pred[REG] != y[None, :]).sum(axis=0)

    hard = [
        int(i)
        for i in np.argsort(-persistence, kind="stable")
        if persistence[i] > 0
    ][:6]

    fig, axs = plt.subplots(len(hard), 3, figsize=(11, 2.4*len(hard)), squeeze=False)

    for j, t in enumerate(hard):
        r = int(np.flatnonzero(pred[REG][:, t] != y[t])[0])
        same, rival = neigh[r, t]

        trajectory(axs[j, 0], X[t], f"Held-out #{t} | {y[t]} | wrong {persistence[t]}/5")
        trajectory(axs[j, 1], X[same], f"Train same #{same} | {y[same]}")
        trajectory(axs[j, 2], X[rival], f"Train rival #{rival} | {y[rival]}")

    save(fig, "10_hard_cases")

    # 11 — Compare neighbor ratios and SVM margins.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    for t in hard[:5]:
        axs[0].plot(
            range(1, REPEATS+1),
            np.log10(np.maximum(ratios[:, t], 1e-12)),
            ".-", label=f"#{t} ({y[t]})"
        )

        axs[1].plot(
            range(1, REPEATS+1),
            svm_ovo[REG][:, t],
            ".-", label=f"#{t} ({y[t]})"
        )

    axs[0].axhline(0, color="gray", ls="--")
    axs[1].axhline(0, color="gray", ls="--")

    axs[0].set(
        title="Nearest-neighbor geometric distance ratio",
        ylabel="log10(rival / same)", xlabel="Repeat"
    )
    axs[1].set(
        title="Worst signed OVO score for true class",
        ylabel="SVM decision score", xlabel="Repeat"
    )

    for ax in axs:
        ax.set_xticks(range(1, REPEATS+1))
        ax.legend(fontsize=8)

    save(fig, "11_hard_margins")

    # 12 — MDS using the first training fold only.
    it, iv = first
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for ax, k in zip(axs, [REF, REG]):
        selected = tuple(params[k][0]["factors"])
        A, B = mix(pair_basis(D, it, iv), r5_weights(selected))
        train, val = mds_oos(A, B)

        ax.scatter(train[:, 0], train[:, 1], s=7, c="lightgray", alpha=.35)

        for label in ("7", "9", "10", "18"):
            mask = y[it] == label
            ax.scatter(
                train[mask, 0], train[mask, 1],
                s=12, label=f"Class {label}", alpha=.8
            )

        mask = pred[k][0, iv] != y[iv]
        ax.scatter(
            val[mask, 0], val[mask, 1],
            marker="x", s=60, c="black", label="Held-out errors"
        )

        ax.set(title=k, xlabel="MDS 1", ylabel="MDS 2")

    axs[1].legend(fontsize=8)
    save(fig, "12_train_mds")

    return hard


# ============================================================
# 7. MAIN — REPEATED NESTED CV
# ============================================================

def main():
    from aeon.datasets import load_classification

    X, y = load_classification(
        "CharacterTrajectories",
        split="train",
        load_equal_length=False
    )

    X = list(X)
    y = np.asarray(y, dtype=str)
    n = len(y)

    assert n == 1422
    assert len(np.unique(y)) == 20

    print(
        f"TRAIN={n} | CV={REPEATS}x{OUTER} | "
        f"INNER={INNER} | DISTINCT WEIGHTS={len(ALL_PATTERNS)}"
    )

    print("Extracting unchanged R5 symbolic features...", flush=True)

    features = extract(X)

    D = {
        k: squareform(pdist(v, "sqeuclidean")).astype(np.float32)
        for k, v in features.items()
    }

    predictions = {
        k: np.empty((REPEATS, n), dtype=y.dtype)
        for k in NAMES
    }

    parameters = {k: [] for k in NAMES}
    sensitivity = []

    svm_ovo = {k: np.full((REPEATS, n), np.nan) for k in NAMES}
    svm_ovr = {k: np.full((REPEATS, n), np.nan) for k in NAMES}

    neighbors = np.full((REPEATS, n, 2), -1, dtype=int)
    ratios = np.full((REPEATS, n), np.nan)

    cv = RepeatedStratifiedKFold(
        n_splits=OUTER,
        n_repeats=REPEATS,
        random_state=SEED
    )

    first = None

    print("\n" + "="*90)
    print("REPEATED NESTED TRAIN-ONLY CV")
    print("="*90)

    for step, (it, iv) in enumerate(cv.split(np.zeros(n), y)):
        r, f = divmod(step, OUTER)

        if first is None:
            first = (it.copy(), iv.copy())

        selected, inner_scores, penalty_sens = inner_tune(D, y, it, seed=1000+step)
        sensitivity.append(penalty_sens)

        row = []

        for name in NAMES:
            key = selected[name]
            factors, C, gamma = key

            yp, B, ovo, ovr = fit_predict(D, y, it, iv, key)

            predictions[name][r, iv] = yp
            svm_ovo[name][r, iv] = ovo
            svm_ovr[name][r, iv] = ovr

            parameters[name].append({
                "factors": list(factors),
                "C": float(C),
                "gamma": float(gamma),
                "inner_f1": float(inner_scores[key]),
                "penalized_objective": float(
                    inner_scores[key] - (penalty_cost(factors, PENALTY) if name == REG else 0.)
                )
            })

            row.append(f"{name}={int((yp != y[iv]).sum())}")

            if name == REG:
                for j, t in enumerate(iv):
                    same = np.flatnonzero(y[it] == y[t])
                    rival = np.flatnonzero(y[it] != y[t])

                    a = same[np.argmin(B[j, same])]
                    b = rival[np.argmin(B[j, rival])]

                    neighbors[r, t] = [it[a], it[b]]
                    ratios[r, t] = (B[j, b]+1e-12)/(B[j, a]+1e-12)

        print(
            f"Repeat {r+1}/{REPEATS} | Fold {f+1}/{OUTER} | "
            + " | ".join(row),
            flush=True
        )

    # Aggregation.
    report = {}

    print("\n" + "="*90)
    print("AGGREGATED TRAIN-ONLY CV")
    print("="*90)

    for name in NAMES:
        errors = [
            int((predictions[name][r] != y).sum())
            for r in range(REPEATS)
        ]

        f1s = [
            f1_score(y, predictions[name][r], average="macro", zero_division=0)
            for r in range(REPEATS)
        ]

        report[name] = {
            "errors": errors,
            "mean_errors": float(np.mean(errors)),
            "sd_errors": float(np.std(errors, ddof=1)),
            "accuracy": float(1-np.mean(errors)/n),
            "macro_f1": float(np.mean(f1s)),
            "parameters": parameters[name]
        }

        print(
            f"{name:26s} | Errors={errors} | "
            f"Mean={np.mean(errors):.2f} ± {np.std(errors, ddof=1):.2f} | "
            f"Acc={100*(1-np.mean(errors)/n):.4f}% | "
            f"F1={np.mean(f1s):.5f}"
        )

    # Exact reproducibility against previously obtained results.
    expected = {
        REF: [4, 6, 7, 5, 6],
        FIX: [6, 5, 6, 3, 7],
        REG: [6, 4, 5, 3, 7]
    }

    ok = all(report[k]["errors"] == v for k, v in expected.items())

    print("\nREPRODUCIBILITY:", "OK" if ok else "FAILED")

    if not ok:
        for k, v in expected.items():
            print(k, report[k]["errors"], "expected", v)

        raise RuntimeError(
            "Reference mismatch: check extractor, data split, package versions and CV seed."
        )

    # Figures.
    hard = figures(
        X, y, D, predictions, report, parameters, sensitivity,
        svm_ovo, svm_ovr, neighbors, ratios, first
    )

    # Save per-sample OOF predictions and SVM scores.
    np.savez_compressed(
        OUT/"haste_weightcheck_oof.npz",
        y=y,
        **{f"pred_{j}": predictions[k] for j, k in enumerate(NAMES)},
        **{f"ovo_{j}": svm_ovo[k] for j, k in enumerate(NAMES)},
        **{f"ovr_{j}": svm_ovr[k] for j, k in enumerate(NAMES)}
    )

    output = {
        "protocol": "Train-only repeated 5x5 nested CV, 3 inner folds",
        "model_order": NAMES,
        "penalty": PENALTY,
        "penalty_sweep_inner_only": PENALTY_SWEEP,
        "historical": HIST,
        "core": CORE,
        "extra": EXTRA,
        "models": report,
        "inner_only_penalty_sensitivity": sensitivity,
        "hard_indices": hard,
        "official_test_access": False
    }

    path = OUT/"haste_weightcheck_results.json"

    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "="*90)
    print("EXPERIMENT FINISHED")
    print("="*90)

    print("Saved:", path)
    print("Saved: results/haste_weightcheck_oof.npz")
    print("Saved: figures/haste_weightcheck_01...12.png")


if __name__ == "__main__":
    main()
