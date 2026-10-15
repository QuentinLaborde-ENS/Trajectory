# -*- coding: utf-8 -*-


import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import SVC


# ============================================================
# HASTE-AL-CS — ADDITIVE SYMBOLIC ABLATION
# ============================================================

SEED, K, H, N, NE = 2027, 24, 2, 241, 64
FT, FR, FS, FC, FE = 8, 16, 4, 4, 4
SIGMA, TURNING_SCALE = .5, .20

OUTER, REPEATS, INNER = 5, 5, 3
C_GRID = (10., 100.)
GAMMA_GRID = (.125, .25, .5, 1., 2., 4.)
LAMBDA_GRID = (.5, 1.5)

GRID, EGRID = np.linspace(0, 1, N), np.linspace(0, 1, NE)
SPEED_BINS, TURN_BINS = (.5, 1., 2.), (-.6, -.2, .2, .6)
SPEED_LEVELS, TURN_LEVELS = np.linspace(0, 1, 4), np.linspace(-1, 1, 5)

FIG, RESULTS = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


# ============================================================
# 1. MODELS AND DISTANCE WEIGHTS
# ============================================================

HIST = dict(time=1., speed_c=.25, arc_mix=4., curv_arc_c=2.,
            events_arc=4/3, legacy_arc=2/3)

CORE = dict(time=1., arc_q=4., events_arc=4/3, legacy_arc=2/3,
            speed_t=.25, turn_arc=2.)

EXTRAS = {
    "R1-TurnTime-93D": {"turn_time": 2.},
    "R2-SpeedArc-105D": {"speed_arc": .25},
    "R3-DualChannels-109D": {"turn_time": 2., "speed_arc": .25},
    "R4-MDSOldPos-109D": {"mds_delta": 4/3, "oldpos": .5},
    "R5-Dual+MDS-129D": {"turn_time": 2., "speed_arc": .25,
                           "mds_delta": 4/3, "oldpos": .5},
    "R6-AllSymbolic-145D": {"turn_time": 2., "speed_arc": .25,
                              "events_time": 4/3, "mds_delta": 4/3,
                              "oldpos": .5}
}

BASE_NAME = "S3-symbolic-89D"
HIST_NAME = "Legacy-hybrid-89D"
SELECT_NAME = "InnerSelected-symbolic"
MODELS = [HIST_NAME, BASE_NAME, *EXTRAS, SELECT_NAME]

SIZES = dict(time=32, speed_c=16, arc_mix=16, curv_arc_c=4,
             events_arc=16, legacy_arc=5, arc_q=16, speed_t=16,
             turn_arc=4, turn_time=4, speed_arc=16, events_time=16,
             mds_delta=16, oldpos=4)


# ============================================================
# 2. SYMBOLIC ENCODING
# ============================================================

def quantize(theta):
    return np.floor((theta+np.pi/K)/(2*np.pi/K)).astype(np.int16) % K


def harmonics(theta):
    return np.stack([fn(h*theta)*np.sqrt(2./K)
                     for h in range(1, H+1)
                     for fn in (np.cos, np.sin)], axis=1)


def interpolate_rank(x):

    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return np.zeros((NE,)+x.shape[1:])

    if len(x) == 1:
        return np.repeat(x, NE, axis=0)

    xp = np.linspace(0, 1, len(x))

    if x.ndim == 1:
        return np.interp(EGRID, xp, x)

    return np.column_stack([
        np.interp(EGRID, xp, x[:, j])
        for j in range(x.shape[1])
    ])


def event_features(ids):

    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]

    symbols = ids[starts]
    lengths = (ends-starts)/len(ids)

    delta = (
        symbols[1:].astype(int)-symbols[:-1].astype(int)+K//2
    ) % K-K//2

    m = len(delta)

    def compressed(x):
        return dct(interpolate_rank(x), axis=0,
                   type=2, norm="ortho")[:FE].ravel()

    events = compressed(harmonics(2*np.pi*delta/K)) if m else np.zeros(16)
    duration = compressed(np.log(np.maximum(len(lengths)*lengths, 1e-12)))
    density = np.array([np.log1p(m)/np.log1p(N-1)])

    return events, np.r_[duration, density], m, starts, symbols


# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================

def encode(case):

    v = np.nan_to_num(np.asarray(case, dtype=float)[:2],
                      nan=0., posinf=0., neginf=0.)

    if v.ndim != 2 or v.shape[0] != 2 or v.shape[1] < 2:
        raise ValueError("Expected 2xT velocity sample")

    n = v.shape[1]
    speed = np.hypot(v[0], v[1])

    positive = speed[speed > 1e-10]
    cutoff = max(.03*np.median(positive), 1e-10) if positive.size else 1e-10

    theta = np.arctan2(v[1], v[0])
    valid = speed > cutoff

    if valid.any():
        prev = np.where(valid, np.arange(n), -1)
        np.maximum.accumulate(prev, out=prev)
        prev[prev < 0] = np.flatnonzero(valid)[0]
        theta = theta[prev]
    else:
        theta = np.zeros(n)

    # Historical temporal orientation.
    ix = np.minimum((GRID*n).astype(int), n-1)
    theta_t = theta[ix]
    ids_t = quantize(theta_t)

    ft = dct(harmonics(2*np.pi*ids_t/K), axis=0,
             type=2, norm="ortho")[:FT].ravel()

    # Historical relative speed and its symbolic version.
    relative_speed = speed[ix]/max(float(speed[ix].mean()), 1e-8)

    speed_c = dct(relative_speed, type=2, norm="ortho")[:FR]
    speed_ids = np.digitize(relative_speed, SPEED_BINS)
    speed_t = dct(SPEED_LEVELS[speed_ids], type=2, norm="ortho")[:FR]

    # Arc-length parameterization, unchanged.
    s = np.r_[0., np.cumsum(.5*(speed[:-1]+speed[1:]))]

    if s[-1] > 1e-12:
        keep = np.r_[True, np.diff(s) > 1e-12]
        theta_arc = np.interp(GRID*s[-1], s[keep], np.unwrap(theta)[keep])
        arc_speed = np.interp(GRID*s[-1], s[keep], speed[keep])
    else:
        theta_arc = theta_t
        arc_speed = speed[ix]

    ids_arc = quantize(theta_arc)
    Za = harmonics(2*np.pi*ids_arc/K)

    arc_mix = dct(.5*Za+.5*harmonics(theta_arc),
                  axis=0, type=2, norm="ortho")[:FS].ravel()

    arc_q = dct(Za, axis=0, type=2, norm="ortho")[:FS].ravel()

    smooth_arc = gaussian_filter1d(theta_arc, sigma=SIGMA, mode="nearest")
    turning_arc = np.tanh(
        np.gradient(smooth_arc, GRID)/(TURNING_SCALE*(N-1))
    )

    curv_arc_c = dct(turning_arc, type=2, norm="ortho")[:FC]
    turn_arc_ids = np.digitize(turning_arc, TURN_BINS)
    turn_arc = dct(TURN_LEVELS[turn_arc_ids], type=2, norm="ortho")[:FC]

    events_arc, legacy_arc, m, starts, symbols = event_features(ids_arc)

    # End of unchanged Legacy + S3 features.

    # R1 — Temporal turning: additional symbolic channel.
    turn_t_sig = np.tanh(
        np.gradient(
            gaussian_filter1d(np.unwrap(theta_t), sigma=SIGMA, mode="nearest"),
            GRID
        )/(TURNING_SCALE*(N-1))
    )

    turn_time_ids = np.digitize(turn_t_sig, TURN_BINS)
    turn_time = dct(TURN_LEVELS[turn_time_ids], type=2, norm="ortho")[:FC]

    # R2 — Arc-length speed: same alphabet as temporal speed.
    arc_relative_speed = arc_speed/max(float(speed[ix].mean()), 1e-8)
    speed_arc_ids = np.digitize(arc_relative_speed, SPEED_BINS)
    speed_arc = dct(SPEED_LEVELS[speed_arc_ids], type=2, norm="ortho")[:FR]

    # R4 — Temporal events as a supplementary branch.
    events_time, _, _, _, _ = event_features(ids_t)

    # R4 — Symbolic MDS4 transition geometry and old event positions.
    # Harmonic embedding equals classical MDS4 up to orthogonal transformation.
    M = harmonics(2*np.pi*np.arange(K)/K)
    delta = M[symbols[1:]]-M[symbols[:-1]]

    mds_delta = (
        dct(interpolate_rank(delta), axis=0,
            type=2, norm="ortho")[:4].ravel()
        if len(delta) else np.zeros(16)
    )

    positions = starts[1:]/(N-1)
    oldpos = (
        dct(interpolate_rank(positions)-EGRID,
            type=2, norm="ortho")[:4]
        if len(positions) else np.zeros(4)
    )

    features = dict(
        time=ft, speed_c=speed_c, arc_mix=arc_mix,
        curv_arc_c=curv_arc_c, events_arc=events_arc,
        legacy_arc=legacy_arc, arc_q=arc_q,
        speed_t=speed_t, turn_arc=turn_arc,
        turn_time=turn_time, speed_arc=speed_arc,
        events_time=events_time, mds_delta=mds_delta,
        oldpos=oldpos
    )

    diagnostics = dict(
        ids_t=ids_t, ids_arc=ids_arc,
        speed_t=speed_ids, speed_arc=speed_arc_ids,
        turn_t=turn_time_ids, turn_arc=turn_arc_ids,
        starts=starts
    )

    return features, diagnostics


def extract(X):

    encoded = [encode(x) for x in X]

    features = {
        k: np.asarray([row[0][k] for row in encoded], dtype=np.float32)
        for k in SIZES
    }

    for k, dim in SIZES.items():
        assert features[k].shape == (len(X), dim), (k, features[k].shape)

    assert sum(SIZES[k] for k in HIST) == 89
    assert sum(SIZES[k] for k in CORE) == 89

    for name, extra in EXTRAS.items():
        assert 89+sum(SIZES[k] for k in extra) == int(name.split("-")[-1][:-1])

    return features, [row[1] for row in encoded]


# ============================================================
# 4. TRAIN-ONLY DISTANCE NORMALIZATION
# ============================================================

def precompute(features):
    return {
        k: squareform(pdist(z, metric="sqeuclidean")).astype(np.float32)
        for k, z in features.items()
    }


def pair_basis(D, it, iv):

    result = {}

    for key, matrix in D.items():

        A = matrix[np.ix_(it, it)]
        scale = max(float(A.sum())/(len(it)*(len(it)-1)), 1e-12)

        result[key] = (
            np.asarray(A, dtype=np.float64)/scale,
            np.asarray(matrix[np.ix_(iv, it)], dtype=np.float64)/scale
        )

    return result


def combine(basis, weights):

    total = sum(weights.values())

    return tuple(
        sum(float(w)*basis[k][j] for k, w in weights.items())/total
        for j in (0, 1)
    )


def weights_for(name, lam):

    if name == HIST_NAME:
        return HIST

    if name == BASE_NAME:
        return CORE

    return {
        **CORE,
        **{k: lam*w for k, w in EXTRAS[name].items()}
    }


def hyper_grid(name):
    return (0.,) if name in (HIST_NAME, BASE_NAME) else LAMBDA_GRID


# ============================================================
# 5. NESTED HYPERPARAMETER SELECTION
# ============================================================

def tune(D, y, it, seed):

    cv = StratifiedKFold(n_splits=INNER, shuffle=True, random_state=seed)

    splits = [
        (it[a], it[b])
        for a, b in cv.split(np.zeros(len(it)), y[it])
    ]

    scores = {
        (name, lam, C, gamma): []
        for name in MODELS[:-1]
        for lam in hyper_grid(name)
        for C in C_GRID
        for gamma in GAMMA_GRID
    }

    for ia, ib in splits:

        basis = pair_basis(D, ia, ib)

        for name in MODELS[:-1]:

            for lam in hyper_grid(name):

                A, B = combine(basis, weights_for(name, lam))

                for gamma in GAMMA_GRID:

                    Ktr = np.exp(-gamma*A)
                    Kva = np.exp(-gamma*B)

                    for C in C_GRID:

                        clf = SVC(C=C, kernel="precomputed").fit(Ktr, y[ia])

                        scores[(name, lam, C, gamma)].append(
                            f1_score(
                                y[ib], clf.predict(Kva),
                                average="macro", zero_division=0
                            )
                        )

        del basis

    winners = {}

    for name in MODELS[:-1]:

        candidates = [
            (float(np.mean(values)), lam, C, gamma)
            for (model, lam, C, gamma), values in scores.items()
            if model == name
        ]

        winners[name] = max(candidates, key=lambda x: x[0])

    # Global symbolic model selection, excluding the hybrid baseline.
    choice = max([
        (values[0], -i, name, *values[1:])
        for i, (name, values) in enumerate(winners.items())
        if name != HIST_NAME
    ])

    return winners, choice[2]


def predict(D, y, it, iv, name, lam, C, gamma):

    basis = pair_basis(D, it, iv)
    A, B = combine(basis, weights_for(name, lam))

    clf = SVC(C=C, kernel="precomputed").fit(
        np.exp(-gamma*A), y[it]
    )

    return clf.predict(np.exp(-gamma*B)), B


# ============================================================
# 6. DIAGNOSTIC FIGURE HELPERS
# ============================================================

def render_path(ax, sample, title):

    v = np.asarray(sample, float)[:2]
    xy = np.cumsum(v, axis=1).T
    xy -= xy[0]

    length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    xy /= max(float(length), 1e-12)

    ax.plot(xy[:, 0], xy[:, 1], lw=1.1)
    ax.scatter(*xy[0], s=15)
    ax.scatter(*xy[-1], marker="x", s=20)

    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def savefig(fig, name):

    fig.tight_layout()
    fig.savefig(FIG/name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def mds_oos(A, B):

    row, grand = A.mean(axis=1), A.mean()
    G = -.5*(A-row[:, None]-row[None, :]+grand)

    values, vectors = eigh(
        G, subset_by_index=(len(A)-2, len(A)-1),
        check_finite=False
    )

    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 1e-12)
    vectors = vectors[:, order]

    train = vectors*np.sqrt(values)

    holdout = (
        -.5*(B-B.mean(axis=1, keepdims=True)-row[None, :]+grand)
    )@vectors/np.sqrt(values)

    return train, holdout


# ============================================================
# 7. TEN DIAGNOSTIC FIGURES
# ============================================================

def figures(X, y, diag, D, preds, first, neighbors, margins, selected):

    names = MODELS
    classes = sorted(np.unique(y), key=int)

    E = np.asarray([
        [(preds[n][r] != y).sum() for r in range(REPEATS)]
        for n in names
    ])

    # 01 — Ablation.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(names, E.mean(axis=1), xerr=E.std(axis=1, ddof=1), capsize=3)
    ax.invert_yaxis()
    ax.set(xlabel="Errors / 1422 trajectories",
           title="Symbolic add-only ablation — nested CV")
    savefig(fig, "haste_sym2_01_ablation.png")

    # 02 — Repeat stability.
    fig, ax = plt.subplots(figsize=(12, 4))

    for i, name in enumerate(names):
        ax.plot(range(1, REPEATS+1), E[i], ".-", lw=1, label=name)

    ax.set(xlabel="Repeat", ylabel="Errors", title="Repeated CV stability")
    ax.set_xticks(range(1, REPEATS+1))
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, "haste_sym2_02_repeats.png")

    # 03 — Confusion matrices.
    shown = [HIST_NAME, BASE_NAME, "R5-Dual+MDS-129D", SELECT_NAME]
    fig, axs = plt.subplots(1, len(shown), figsize=(18, 5))

    for ax, name in zip(axs, shown):

        cm = confusion_matrix(
            np.tile(y, REPEATS), preds[name].ravel(), labels=classes
        )

        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6)
        ax.set_yticks(range(len(classes)), classes, fontsize=6)
        ax.set(xlabel="Predicted", ylabel="True", title=name)

    savefig(fig, "haste_sym2_03_confusion.png")

    # 04 — Class errors.
    fig, ax = plt.subplots(figsize=(12, 4))
    x, width = np.arange(len(classes)), .20

    for j, name in enumerate(shown):

        counts = [
            int((preds[name][:, y == c] != c).sum())
            for c in classes
        ]

        ax.bar(x+(j-1.5)*width, counts, width, label=name)

    ax.set_xticks(x, classes)
    ax.set(xlabel="Class", ylabel="Errors in 5 repeats",
           title="Persistent class errors")
    ax.legend(fontsize=7)
    savefig(fig, "haste_sym2_04_class_errors.png")

    # 05 — Paired error overlap.
    wrong = [(preds[n] != y[None, :]).ravel() for n in names]
    J = np.eye(len(names))

    for i in range(len(names)):
        for j in range(i+1, len(names)):

            intersection = np.logical_and(wrong[i], wrong[j]).sum()
            union = np.logical_or(wrong[i], wrong[j]).sum()

            J[i, j] = J[j, i] = intersection/max(1, union)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Repeated out-of-fold error overlap")
    savefig(fig, "haste_sym2_05_error_overlap.png")

    # 06 — Persistent errors and train-only neighbors.
    persistence = (preds[SELECT_NAME] != y[None, :]).sum(axis=0)
    hard = np.argsort(-persistence, kind="stable")[:6]

    fig, axs = plt.subplots(len(hard), 3, figsize=(12, len(hard)*2.6),
                            squeeze=False)

    for j, t in enumerate(hard):

        r = next(
            (r for r in range(REPEATS) if preds[SELECT_NAME][r, t] != y[t]), 0
        )

        a, b = neighbors[r, t]

        render_path(axs[j, 0], X[t],
                    f"Held-out #{t} | true {y[t]} | err {persistence[t]}/5")

        if a >= 0:
            render_path(axs[j, 1], X[a], f"Train same #{a} | {y[a]}")

        if b >= 0:
            render_path(axs[j, 2], X[b], f"Train rival #{b} | {y[b]}")

    savefig(fig, "haste_sym2_06_hard_cases.png")

    # 07 — Symbolic profiles.
    fig, axs = plt.subplots(4, 4, figsize=(15, 10), squeeze=False)

    for j, t in enumerate(hard[:4]):

        z = diag[t]

        for k, key in enumerate(("ids_t", "ids_arc", "speed_t", "turn_arc")):
            axs[j, k].step(GRID, z[key], where="mid", lw=1)

        axs[j, 0].set_ylabel(f"#{t} | {y[t]}")

        for ax in axs[j]:
            ax.set_xlabel("Normalized time or arc")
            ax.spines[["top", "right"]].set_visible(False)

    titles = ("Time orientation", "Arc orientation",
              "Time speed symbols", "Arc turning symbols")

    for ax, title in zip(axs[0], titles):
        ax.set_title(title)

    savefig(fig, "haste_sym2_07_symbolic_profiles.png")

    # 08 — Nearest-neighbor margins.
    fig, ax = plt.subplots(figsize=(10, 4))

    for t in hard[:5]:

        ax.plot(
            range(1, REPEATS+1),
            np.log10(np.maximum(margins[:, t], 1e-12)),
            ".-", label=f"#{t}: {y[t]}"
        )

    ax.axhline(0, color="gray", ls="--")
    ax.set_xticks(range(1, REPEATS+1))
    ax.set(xlabel="Repeat", ylabel="log10(rival / same)",
           title="Inner-selected nearest-neighbor margins")
    ax.legend(fontsize=8)
    savefig(fig, "haste_sym2_08_neighbor_margins.png")

    # 09 — MDS fitted on training fold only.
    it, iv = first
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    for ax, name, lam in (
        (axs[0], BASE_NAME, 0.),
        (axs[1], "R5-Dual+MDS-129D", 1.5)
    ):

        basis = pair_basis(D, it, iv)
        A, B = combine(basis, weights_for(name, lam))
        ztr, zva = mds_oos(A, B)

        ax.scatter(ztr[:, 0], ztr[:, 1], s=7, c="lightgray", alpha=.4)

        for c in ("7", "9", "10", "18"):

            mask = y[it] == c

            ax.scatter(ztr[mask, 0], ztr[mask, 1], s=12,
                       alpha=.75, label=f"Class {c}")

        errors = preds[name][0, iv] != y[iv]

        ax.scatter(zva[errors, 0], zva[errors, 1],
                   marker="x", c="black", s=65)

        ax.set(xlabel="MDS1", ylabel="MDS2", title=name)

    axs[1].legend(fontsize=8)
    savefig(fig, "haste_sym2_09_train_only_mds.png")

    # 10 — Model selection frequencies.
    counts = Counter(row["model"] for row in selected)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.barh(list(counts.keys()), list(counts.values()))
    ax.invert_yaxis()
    ax.set(xlabel="Outer folds (out of 25)",
           title="Inner-fold model selection frequencies")
    savefig(fig, "haste_sym2_10_selected_models.png")

    return hard.tolist()


# ============================================================
# 8. REPEATED NESTED CROSS-VALIDATION
# ============================================================

def main():

    from aeon.datasets import load_classification

    X, y = load_classification(
        "CharacterTrajectories", split="train", load_equal_length=False
    )

    X = list(X)
    y = np.asarray(y, dtype=str)
    n = len(y)

    assert n == 1422 and len(np.unique(y)) == 20

    print(
        f"TRAIN={n} | CLASSES=20 | "
        f"CV={REPEATS}x{OUTER}, inner={INNER} | MODELS={len(MODELS)}"
    )

    features, diagnostics = extract(X)
    D = precompute(features)

    predictions = {
        name: np.empty((REPEATS, n), dtype=y.dtype)
        for name in MODELS
    }

    parameters = {name: [] for name in MODELS}
    selections = []

    neighbors = np.full((REPEATS, n, 2), -1, dtype=int)
    margins = np.full((REPEATS, n), np.nan)

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER, n_repeats=REPEATS, random_state=SEED
    )

    first = None

    print("\n"+"="*95)
    print("REPEATED NESTED TRAIN-ONLY CROSS-VALIDATION")
    print("="*95)

    for step, (it, iv) in enumerate(outer.split(np.zeros(n), y)):

        r, f = divmod(step, OUTER)

        if first is None:
            first = (it.copy(), iv.copy())

        tuned, picked = tune(D, y, it, seed=1000+step)
        row = []

        for name in MODELS:

            source = picked if name == SELECT_NAME else name
            score, lam, C, gamma = tuned[source]

            yp, B = predict(D, y, it, iv, source, lam, C, gamma)

            predictions[name][r, iv] = yp
            err = int((yp != y[iv]).sum())

            row.append(f"{name}={err}")

            parameters[name].append([
                source, float(lam), float(C),
                float(gamma), float(score)
            ])

            if name == SELECT_NAME:

                selections.append(dict(
                    repeat=r+1, fold=f+1, model=source,
                    lam=float(lam), C=float(C),
                    gamma=float(gamma), errors=err
                ))

                for j, t in enumerate(iv):

                    same = np.flatnonzero(y[it] == y[t])
                    rival = np.flatnonzero(y[it] != y[t])

                    a = same[np.argmin(B[j, same])]
                    b = rival[np.argmin(B[j, rival])]

                    neighbors[r, t] = (it[a], it[b])

                    margins[r, t] = (
                        (B[j, b]+1e-12)/(B[j, a]+1e-12)
                    )

        print(
            f"Repeat {r+1}/{REPEATS} | Fold {f+1}/{OUTER} | "
            + " | ".join(row),
            flush=True
        )

    # Aggregation.
    report = {}

    print("\n"+"="*95)
    print("AGGREGATED TRAIN-ONLY CV")
    print("="*95)

    for name in MODELS:

        errors = [
            int((predictions[name][r] != y).sum())
            for r in range(REPEATS)
        ]

        f1 = [
            f1_score(y, predictions[name][r],
                     average="macro", zero_division=0)
            for r in range(REPEATS)
        ]

        report[name] = dict(
            errors=errors,
            mean_errors=float(np.mean(errors)),
            sd_errors=float(np.std(errors, ddof=1)),
            accuracy=float(1-np.mean(errors)/n),
            macro_f1=float(np.mean(f1)),
            params=parameters[name]
        )

        print(
            f"{name:32s} | Errors={errors} | "
            f"Mean={np.mean(errors):.2f} ± "
            f"{np.std(errors, ddof=1):.2f} | "
            f"Acc={100*(1-np.mean(errors)/n):.4f}% | "
            f"F1={np.mean(f1):.5f}"
        )

    # Historical reproducibility.
    expected = {
        HIST_NAME: [5, 5, 7, 4, 6],
        BASE_NAME: [9, 5, 6, 9, 8]
    }

    print("\nHISTORICAL REPRODUCIBILITY")

    for name, reference in expected.items():

        observed = report[name]["errors"]

        print(
            f"{name} | "
            f"{'OK' if observed == reference else 'DIFFERENT'} | "
            f"expected={reference}, observed={observed}"
        )

    if any(report[name]["errors"] != ref for name, ref in expected.items()):
        raise RuntimeError(
            "Historical baselines differ. "
            "Verify code, dataset, random seed and CV settings."
        )

    print("\nInner-selected variants:")
    print(Counter(row["model"] for row in selections))

    # Diagnostic figures.
    hard = figures(
        X, y, diagnostics, D, predictions,
        first, neighbors, margins, selections
    )

    # Save OOF predictions.
    np.savez_compressed(
        RESULTS/"haste_sym2_oof.npz",
        y=y,
        **{
            f"model_{i}": predictions[name]
            for i, name in enumerate(MODELS)
        }
    )

    output = dict(
        seed=SEED,
        n_samples=n,
        models=report,
        selection=selections,
        model_order=MODELS,
        weights_core=CORE,
        weights_historical=HIST,
        extra_blocks=EXTRAS,
        lambda_grid=LAMBDA_GRID,
        hard_cases=hard,
        protocol="train-only repeated nested 5x5 CV; inner 3-fold",
        official_test_access=False
    )

    path = RESULTS/"haste_sym2_results.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n"+"="*95)
    print("EXPERIMENT FINISHED")
    print("="*95)
    print(f"Saved: {path}")
    print("Saved: results/haste_sym2_oof.npz")
    print("Saved: figures/haste_sym2_01...10.png")


if __name__ == "__main__":
    main()
