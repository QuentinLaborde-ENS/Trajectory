# -*- coding: utf-8 -*-


import json
from collections import Counter
from itertools import product
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
# HASTE-AL-CS — SYMBOLIC WEIGHTING ABLATION
# ============================================================

SEED, K, H, N, NE = 2027, 24, 2, 241, 64
FT, FR, FS, FC, FE = 8, 16, 4, 4, 4
SIGMA, TURNING_SCALE = .5, .20

OUTER, REPEATS, INNER = 5, 5, 3
C_GRID = (10., 100.)
GAMMA_GRID = (.125, .25, .5, 1., 2., 4.)

FACTORS = (.5, 1., 1.5)
PENALTY = .0005

GRID = np.linspace(0, 1, N)
EGRID = np.linspace(0, 1, NE)

FIG, RESULTS = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


# ============================================================
# 1. HISTORICAL AND SYMBOLIC DISTANCE WEIGHTS
# ============================================================

HIST = dict(
    time=1., speed_c=.25, arc_mix=4.,
    curv_arc_c=2., events_arc=4/3, legacy_arc=2/3
)

CORE = dict(
    time=1., arc_q=4.,
    events_arc=4/3, legacy_arc=2/3,
    speed_t=.25, turn_arc=2.
)

EXTRA = dict(
    turn_time=2.,
    speed_arc=.25,
    mds_delta=4/3,
    oldpos=.5
)

EXTRA_KEYS = list(EXTRA)

MODEL_NAMES = [
    "Legacy-hybrid-89D",
    "S3-symbolic-89D",
    "R5-Reference-129D",
    "R5-Fixed-129D",
    "R5-2Group-129D",
    "R5-3Group-129D",
    "R5-3Group-Reg-129D",
    "R5-4Group-Local-129D"
]

REF, FIX, GROUP2, GROUP3, REG3, LOCAL4 = MODEL_NAMES[2:]


# ============================================================
# 2. CANDIDATE WEIGHT CONFIGURATIONS
# ============================================================

# Each tuple follows:
# (turn_time, speed_arc, mds_delta, oldpos)

REF_CAND = [
    (v, v, v, v)
    for v in (.5, 1.5)
]

FIX_CAND = [(1., 1., 1., 1.)]

GROUP2_CAND = [
    (a, a, b, b)
    for a, b in product(FACTORS, repeat=2)
]

GROUP3_CAND = [
    (a, b, c, c)
    for a, b, c in product(FACTORS, repeat=3)
]

LOCAL4_CAND = [(1., 1., 1., 1.)]

for j in range(4):
    for p in (.5, 1.5):
        x = [1.] * 4
        x[j] = p
        LOCAL4_CAND.append(tuple(x))

FAMILIES = {
    REF: REF_CAND,
    FIX: FIX_CAND,
    GROUP2: GROUP2_CAND,
    GROUP3: GROUP3_CAND,
    REG3: GROUP3_CAND,
    LOCAL4: LOCAL4_CAND
}

# Avoid refitting duplicate weight configurations.
CANDIDATES = list(dict.fromkeys([
    p for group in FAMILIES.values()
    for p in group
]))


# ============================================================
# 3. SYMBOLIC REPRESENTATION
# ============================================================

def quantize(theta):
    return np.floor((theta+np.pi/K)/(2*np.pi/K)).astype(np.int16) % K


def harmonics(theta):
    return np.stack([
        f(h*theta)*np.sqrt(2./K)
        for h in range(1, H+1)
        for f in (np.cos, np.sin)
    ], axis=1)


def interp_rank(x):

    x = np.asarray(x, float)

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

    starts = np.flatnonzero(
        np.r_[True, ids[1:] != ids[:-1]]
    )

    ends = np.r_[starts[1:], len(ids)]
    symbols = ids[starts]
    lengths = (ends-starts)/len(ids)

    delta = (
        symbols[1:].astype(int)
        - symbols[:-1].astype(int)
        + K//2
    ) % K-K//2

    def pack(x):
        return dct(
            interp_rank(x),
            axis=0, type=2, norm="ortho"
        )[:FE].ravel()

    events = (
        pack(harmonics(2*np.pi*delta/K))
        if len(delta) else np.zeros(16)
    )

    durations = pack(
        np.log(np.maximum(len(lengths)*lengths, 1e-12))
    )

    density = np.array([
        np.log1p(len(delta))/np.log1p(N-1)
    ])

    return events, np.r_[durations, density], starts, symbols


# ============================================================
# 4. FEATURE EXTRACTION — HISTORICAL R5 UNCHANGED
# ============================================================

def encode(case):

    v = np.nan_to_num(
        np.asarray(case, float)[:2],
        nan=0., posinf=0., neginf=0.
    )

    if v.shape[0] != 2 or v.shape[1] < 2:
        raise ValueError("Expected two-channel velocity trajectory")

    n = v.shape[1]
    speed = np.hypot(v[0], v[1])

    positive = speed[speed > 1e-10]
    cutoff = (
        max(.03*np.median(positive), 1e-10)
        if len(positive) else 1e-10
    )

    theta = np.arctan2(v[1], v[0])
    valid = speed > cutoff

    if valid.any():

        prev = np.where(valid, np.arange(n), -1)
        np.maximum.accumulate(prev, out=prev)

        prev[prev < 0] = np.flatnonzero(valid)[0]
        theta = theta[prev]

    else:
        theta = np.zeros(n)

    # --------------------------------------------------------
    # Time-domain orientation and speed
    # --------------------------------------------------------

    ix = np.minimum((GRID*n).astype(int), n-1)

    theta_t = theta[ix]
    ids_t = quantize(theta_t)

    ft = dct(
        harmonics(2*np.pi*ids_t/K),
        axis=0, type=2, norm="ortho"
    )[:FT].ravel()

    rel = speed[ix]/max(float(speed[ix].mean()), 1e-8)

    speed_c = dct(
        rel, type=2, norm="ortho"
    )[:FR]

    speed_t = dct(
        np.linspace(0, 1, 4)[
            np.digitize(rel, (.5, 1., 2.))
        ],
        type=2, norm="ortho"
    )[:FR]

    # --------------------------------------------------------
    # Arc-length parameterization
    # --------------------------------------------------------

    s = np.r_[
        0.,
        np.cumsum(.5*(speed[:-1]+speed[1:]))
    ]

    if s[-1] > 1e-12:

        keep = np.r_[True, np.diff(s) > 1e-12]

        theta_arc = np.interp(
            GRID*s[-1],
            s[keep],
            np.unwrap(theta)[keep]
        )

        arc_speed = np.interp(
            GRID*s[-1],
            s[keep],
            speed[keep]
        )

    else:
        theta_arc = theta_t
        arc_speed = speed[ix]

    ids_arc = quantize(theta_arc)
    Za = harmonics(2*np.pi*ids_arc/K)

    arc_mix = dct(
        .5*Za+.5*harmonics(theta_arc),
        axis=0, type=2, norm="ortho"
    )[:FS].ravel()

    arc_q = dct(
        Za, axis=0, type=2, norm="ortho"
    )[:FS].ravel()

    smooth_arc = gaussian_filter1d(
        theta_arc, sigma=SIGMA, mode="nearest"
    )

    turning_arc = np.tanh(
        np.gradient(smooth_arc, GRID) /
        (TURNING_SCALE*(N-1))
    )

    curv_arc_c = dct(
        turning_arc, type=2, norm="ortho"
    )[:FC]

    turn_arc = dct(
        np.linspace(-1, 1, 5)[
            np.digitize(turning_arc, (-.6, -.2, .2, .6))
        ],
        type=2, norm="ortho"
    )[:FC]

    events_arc, legacy_arc, starts, symbols = (
        event_features(ids_arc)
    )

    # --------------------------------------------------------
    # Additional symbolic R5 branches
    # --------------------------------------------------------

    turn_t_sig = np.tanh(
        np.gradient(
            gaussian_filter1d(
                np.unwrap(theta_t),
                sigma=SIGMA,
                mode="nearest"
            ),
            GRID
        )/(TURNING_SCALE*(N-1))
    )

    turn_time = dct(
        np.linspace(-1, 1, 5)[
            np.digitize(turn_t_sig, (-.6, -.2, .2, .6))
        ],
        type=2, norm="ortho"
    )[:FC]

    speed_arc_rel = (
        arc_speed/max(float(speed[ix].mean()), 1e-8)
    )

    speed_arc = dct(
        np.linspace(0, 1, 4)[
            np.digitize(speed_arc_rel, (.5, 1., 2.))
        ],
        type=2, norm="ortho"
    )[:FR]

    # MDS4-equivalent circular harmonic embedding.
    M = harmonics(2*np.pi*np.arange(K)/K)

    delta = (
        M[symbols[1:]]
        - M[symbols[:-1]]
    )

    mds_delta = (
        dct(
            interp_rank(delta),
            axis=0, type=2, norm="ortho"
        )[:4].ravel()
        if len(delta) else np.zeros(16)
    )

    positions = starts[1:]/(N-1)

    oldpos = (
        dct(
            interp_rank(positions)-EGRID,
            type=2, norm="ortho"
        )[:4]
        if len(positions) else np.zeros(4)
    )

    return dict(
        time=ft,
        speed_c=speed_c,
        arc_mix=arc_mix,
        curv_arc_c=curv_arc_c,
        events_arc=events_arc,
        legacy_arc=legacy_arc,
        arc_q=arc_q,
        speed_t=speed_t,
        turn_arc=turn_arc,
        turn_time=turn_time,
        speed_arc=speed_arc,
        mds_delta=mds_delta,
        oldpos=oldpos
    )


def extract(X):

    blocks = [encode(x) for x in X]

    features = {
        k: np.stack([
            b[k] for b in blocks
        ]).astype(np.float32)
        for k in blocks[0]
    }

    assert sum(features[k].shape[1] for k in HIST) == 89
    assert sum(features[k].shape[1] for k in CORE) == 89

    assert (
        sum(features[k].shape[1] for k in CORE)
        + sum(features[k].shape[1] for k in EXTRA)
    ) == 129

    return features


# ============================================================
# 5. DISTANCE NORMALIZATION AND WEIGHTING
# ============================================================

def pair_basis(D, it, iv):

    basis = {}

    for key, matrix in D.items():

        A = matrix[np.ix_(it, it)]

        scale = max(
            float(A.sum())/(len(it)*(len(it)-1)),
            1e-12
        )

        basis[key] = (
            np.asarray(A, dtype=np.float64)/scale,
            np.asarray(
                matrix[np.ix_(iv, it)],
                dtype=np.float64
            )/scale
        )

    return basis


def mix(basis, weights):

    total = sum(weights.values())

    return tuple(
        sum(
            weight*basis[k][j]
            for k, weight in weights.items()
        )/total
        for j in (0, 1)
    )


def r5_weights(factors):

    return {
        **CORE,
        **{
            k: EXTRA[k]*f
            for k, f in zip(EXTRA_KEYS, factors)
        }
    }


# ============================================================
# 6. INNER CROSS-VALIDATION
# ============================================================

def inner_tune(D, y, it, seed):

    cv = StratifiedKFold(
        n_splits=INNER,
        shuffle=True,
        random_state=seed
    )

    splits = [
        (it[a], it[b])
        for a, b in cv.split(
            np.zeros(len(it)), y[it]
        )
    ]

    # Historical and symbolic baselines.
    score = {
        ("hist", None, C, g): []
        for C in C_GRID
        for g in GAMMA_GRID
    }

    score.update({
        ("core", None, C, g): []
        for C in C_GRID
        for g in GAMMA_GRID
    })

    # All distinct R5 weight configurations.
    score.update({
        ("r5", m, C, g): []
        for m in CANDIDATES
        for C in C_GRID
        for g in GAMMA_GRID
    })

    for ia, ib in splits:

        basis = pair_basis(D, ia, ib)

        configurations = (
            [("hist", None), ("core", None)]
            + [("r5", m) for m in CANDIDATES]
        )

        for kind, factors in configurations:

            if kind == "hist":
                weights = HIST

            elif kind == "core":
                weights = CORE

            else:
                weights = r5_weights(factors)

            A, B = mix(basis, weights)

            for gamma in GAMMA_GRID:

                Ktr = np.exp(-gamma*A)
                Kval = np.exp(-gamma*B)

                for C in C_GRID:

                    clf = SVC(
                        C=C, kernel="precomputed"
                    ).fit(
                        Ktr, y[ia]
                    )

                    yp = clf.predict(Kval)

                    score[
                        (kind, factors, C, gamma)
                    ].append(
                        f1_score(
                            y[ib], yp,
                            average="macro",
                            zero_division=0
                        )
                    )

    means = {
        k: float(np.mean(values))
        for k, values in score.items()
    }

    winners = {}

    for name in MODEL_NAMES:

        if name == MODEL_NAMES[0]:

            keys = [
                ("hist", None, C, g)
                for C in C_GRID
                for g in GAMMA_GRID
            ]

        elif name == MODEL_NAMES[1]:

            keys = [
                ("core", None, C, g)
                for C in C_GRID
                for g in GAMMA_GRID
            ]

        else:

            keys = [
                ("r5", m, C, g)
                for m in FAMILIES[name]
                for C in C_GRID
                for g in GAMMA_GRID
            ]

        def objective(key):

            if name != REG3:
                return means[key]

            # Fixed regularization toward unit multipliers.
            a, b, c, _ = key[1]

            penalty = PENALTY*sum(
                np.log(v)**2
                for v in (a, b, c)
            )

            return means[key]-penalty

        best = max(keys, key=objective)

        winners[name] = (
            *best,
            means[best],
            objective(best)
        )

    return winners


# ============================================================
# 7. FIT AND PREDICT
# ============================================================

def pred_fold(D, y, it, iv, key):

    kind, factors, C, gamma = key

    basis = pair_basis(D, it, iv)

    if kind == "hist":
        weights = HIST

    elif kind == "core":
        weights = CORE

    else:
        weights = r5_weights(factors)

    A, B = mix(basis, weights)

    clf = SVC(
        C=C, kernel="precomputed"
    ).fit(
        np.exp(-gamma*A),
        y[it]
    )

    yp = clf.predict(
        np.exp(-gamma*B)
    )

    return yp, B


# ============================================================
# 8. VISUALIZATION HELPERS
# ============================================================

def save(fig, name):

    fig.tight_layout()

    fig.savefig(
        FIG/f"haste_weights_{name}.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_path(ax, sample, title):

    v = np.nan_to_num(
        np.asarray(sample, float)[:2]
    )

    xy = np.cumsum(v.T, axis=0)
    xy -= xy[0]

    length = np.linalg.norm(
        np.diff(xy, axis=0),
        axis=1
    ).sum()

    xy /= max(float(length), 1e-12)

    ax.plot(xy[:, 0], xy[:, 1], lw=1.2)
    ax.scatter(*xy[0], s=15)
    ax.scatter(*xy[-1], s=20, marker="x")

    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines[["top", "right"]].set_visible(False)


def mds_oos(A, B):

    row = A.mean(axis=1)
    grand = A.mean()

    G = -.5*(
        A-row[:, None]-row[None, :]+grand
    )

    values, U = eigh(
        G,
        subset_by_index=(len(A)-2, len(A)-1),
        check_finite=False
    )

    order = np.argsort(values)[::-1]

    values = np.maximum(values[order], 1e-12)
    U = U[:, order]

    train = U*np.sqrt(values)

    holdout = (
        -.5*(
            B-B.mean(axis=1, keepdims=True)
            -row[None, :]+grand
        )
    )@U/np.sqrt(values)

    return train, holdout


# ============================================================
# 9. DIAGNOSTIC FIGURES
# ============================================================

def make_figures(X, y, D, pred, rep_errors,
                 params, first, neighbors, margins):

    names = MODEL_NAMES
    labels = sorted(np.unique(y), key=int)

    E = np.asarray([
        rep_errors[n]
        for n in names
    ])

    avg = E.mean(axis=1)

    # --------------------------------------------------------
    # Figure 1 — ablation
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.barh(
        names, avg,
        xerr=E.std(axis=1, ddof=1),
        capsize=3
    )

    ax.invert_yaxis()

    ax.set(
        xlabel="Errors per 1422 samples",
        title="R5 — symbolic block weighting (nested CV)"
    )

    save(fig, "01_ablation")

    # --------------------------------------------------------
    # Figure 2 — repeated CV
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 4))

    for j, name in enumerate(names):

        ax.plot(
            range(1, REPEATS+1),
            E[j],
            ".-",
            lw=1,
            label=name
        )

    ax.set(
        xlabel="CV repeat",
        ylabel="Errors",
        title="Paired repeated CV"
    )

    ax.set_xticks(range(1, REPEATS+1))
    ax.legend(fontsize=7, ncol=2)

    save(fig, "02_repeats")

    # --------------------------------------------------------
    # Figure 3 — confusion matrices
    # --------------------------------------------------------

    shown = [
        MODEL_NAMES[i]
        for i in (0, 1, 2, 5, 6)
    ]

    fig, axs = plt.subplots(
        1, len(shown),
        figsize=(20, 4.5)
    )

    for ax, name in zip(axs, shown):

        cm = confusion_matrix(
            np.tile(y, REPEATS),
            pred[name].ravel(),
            labels=labels
        )

        ax.imshow(cm, cmap="Blues")

        ax.set_xticks(
            range(len(labels)),
            labels,
            rotation=90,
            fontsize=6
        )

        ax.set_yticks(
            range(len(labels)),
            labels,
            fontsize=6
        )

        ax.set(
            xlabel="Predicted",
            ylabel="True",
            title=name
        )

    save(fig, "03_confusions")

    # --------------------------------------------------------
    # Figure 4 — class-specific errors
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 4))

    xs = np.arange(len(labels))
    width = .16

    for j, name in enumerate(shown):

        counts = [
            int((pred[name][:, y == c] != c).sum())
            for c in labels
        ]

        ax.bar(
            xs+(j-2)*width,
            counts,
            width,
            label=name
        )

    ax.set_xticks(xs, labels)

    ax.set(
        xlabel="True class",
        ylabel="Errors across five repeats",
        title="Error concentration"
    )

    ax.legend(fontsize=7)

    save(fig, "04_class_errors")

    # --------------------------------------------------------
    # Figure 5 — paired error overlap
    # --------------------------------------------------------

    wrong = [
        (pred[name] != y[None, :]).ravel()
        for name in names
    ]

    J = np.eye(len(names))

    for i in range(len(names)):

        for j in range(i+1, len(names)):

            common = np.logical_and(
                wrong[i], wrong[j]
            ).sum()

            union = np.logical_or(
                wrong[i], wrong[j]
            ).sum()

            J[i, j] = J[j, i] = common/max(1, union)

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(
        J, cmap="viridis",
        vmin=0, vmax=1
    )

    ax.set_xticks(
        range(len(names)), names,
        rotation=55, ha="right",
        fontsize=8
    )

    ax.set_yticks(
        range(len(names)), names,
        fontsize=8
    )

    fig.colorbar(im, ax=ax)

    ax.set_title("Paired OOF error overlap — Jaccard")

    save(fig, "05_overlap")

    # --------------------------------------------------------
    # Figure 6 — selected weight coefficients
    # --------------------------------------------------------

    fig, axs = plt.subplots(
        2, 2,
        figsize=(12, 7),
        sharex=True,
        sharey=True
    )

    selected_models = [
        REF, GROUP2, GROUP3, REG3
    ]

    for ax, name in zip(axs.flat, selected_models):

        p = np.asarray([
            row["factors"]
            for row in params[name]
        ], float)

        for j, key in enumerate(EXTRA_KEYS):

            ax.plot(
                range(1, len(p)+1),
                p[:, j],
                ".-",
                lw=1,
                ms=3,
                label=key
            )

        ax.set(
            title=name,
            xlabel="Outer fold (ordered)",
            ylabel="Multiplier"
        )

        ax.set_ylim(.35, 1.7)
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, ncol=2)

    save(fig, "06_selected_weights")

    # --------------------------------------------------------
    # Figure 7 — stability of selected weight patterns
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 4))

    for name in [
        REF, GROUP2, GROUP3, REG3, LOCAL4
    ]:

        tuples = [
            tuple(p["factors"])
            for p in params[name]
        ]

        freq = Counter(tuples)

        ax.plot(
            np.arange(1, len(freq)+1),
            sorted(freq.values(), reverse=True),
            ".-",
            label=name
        )

    ax.set(
        xlabel="Weight-pattern popularity rank",
        ylabel="Number of outer folds",
        title="Stability of selected weighting patterns"
    )

    ax.legend(fontsize=7)

    save(fig, "07_weight_stability")

    # --------------------------------------------------------
    # Figure 8 — corrected vs new errors
    # --------------------------------------------------------

    fig, axs = plt.subplots(
        1, 2,
        figsize=(11, 4)
    )

    variants = [
        FIX, GROUP2, GROUP3, REG3, LOCAL4
    ]

    for ax, baseline in zip(
        axs, [MODEL_NAMES[1], REF]
    ):

        wins = []
        losses = []

        for name in variants:

            previous = pred[baseline] == y[None, :]
            current = pred[name] == y[None, :]

            wins.append(
                int((~previous & current).sum())
            )

            losses.append(
                int((previous & ~current).sum())
            )

        x = np.arange(len(variants))

        ax.bar(
            x-.19, wins, .38,
            label="Corrected"
        )

        ax.bar(
            x+.19, losses, .38,
            label="New errors"
        )

        ax.set_xticks(
            x,
            [v.replace("R5-", "") for v in variants],
            rotation=30,
            ha="right",
            fontsize=8
        )

        ax.set(
            title=f"Compared with {baseline}",
            ylabel="OOF decisions over five repeats"
        )

        ax.legend(fontsize=8)

    save(fig, "08_pairwise_corrections")

    # --------------------------------------------------------
    # Figures 9–10 — persistent errors and margins
    # --------------------------------------------------------

    target = REG3

    persistence = (
        pred[target] != y[None, :]
    ).sum(axis=0)

    hard = [
        int(i)
        for i in np.argsort(-persistence, kind="stable")
        if persistence[i] > 0
    ][:6]

    if hard:

        fig, axs = plt.subplots(
            len(hard), 3,
            figsize=(12, 2.6*len(hard)),
            squeeze=False
        )

        for j, t in enumerate(hard):

            r = next(
                r for r in range(REPEATS)
                if pred[target][r, t] != y[t]
            )

            a, b = neighbors[r, t]

            plot_path(
                axs[j, 0], X[t],
                f"Held-out #{t} | {y[t]} | wrong {persistence[t]}/5"
            )

            plot_path(
                axs[j, 1], X[a],
                f"Train same #{a} | {y[a]}"
            )

            plot_path(
                axs[j, 2], X[b],
                f"Train rival #{b} | {y[b]}"
            )

        save(fig, "09_hard_cases")

        fig, ax = plt.subplots(figsize=(11, 4))

        for t in hard[:5]:

            ax.plot(
                range(1, REPEATS+1),
                np.log10(
                    np.maximum(margins[:, t], 1e-12)
                ),
                ".-",
                label=f"#{t}: {y[t]}"
            )

        ax.axhline(
            0,
            ls="--",
            color="gray"
        )

        ax.set_xticks(range(1, REPEATS+1))

        ax.set(
            xlabel="CV repeat",
            ylabel="log10(nearest rival / nearest same)",
            title="R5 regularized — held-out neighbor margins"
        )

        ax.legend(fontsize=8)

        save(fig, "10_neighbor_margins")

    # --------------------------------------------------------
    # Figure 11 — train-only MDS
    # --------------------------------------------------------

    it, iv = first

    fig, axs = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    comparisons = [
        (axs[0], REF, tuple(params[REF][0]["factors"])),
        (axs[1], REG3, tuple(params[REG3][0]["factors"]))
    ]

    for ax, name, factors in comparisons:

        basis = pair_basis(D, it, iv)

        A, B = mix(
            basis,
            r5_weights(factors)
        )

        train, val = mds_oos(A, B)

        ax.scatter(
            train[:, 0],
            train[:, 1],
            s=7,
            c="lightgray",
            alpha=.4
        )

        for c in ("7", "9", "10", "18"):

            mask = y[it] == c

            ax.scatter(
                train[mask, 0],
                train[mask, 1],
                s=12,
                alpha=.75,
                label=f"Class {c}"
            )

        errors = (
            pred[name][0, iv] != y[iv]
        )

        ax.scatter(
            val[errors, 0],
            val[errors, 1],
            s=60,
            c="black",
            marker="x"
        )

        ax.set(
            title=name,
            xlabel="MDS 1",
            ylabel="MDS 2"
        )

    axs[1].legend(fontsize=8)

    save(fig, "11_train_only_mds")

    return hard


# ============================================================
# 10. MAIN — REPEATED NESTED TRAIN-ONLY CV
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
        f"TRAIN={n} | "
        f"OUTER={REPEATS}x{OUTER} | "
        f"INNER={INNER} | "
        f"Unique R5 weight patterns={len(CANDIDATES)}"
    )

    print("Extracting features...", flush=True)

    features = extract(X)

    D = {
        k: squareform(
            pdist(v, "sqeuclidean")
        ).astype(np.float32)
        for k, v in features.items()
    }

    predictions = {
        name: np.empty(
            (REPEATS, n),
            dtype=y.dtype
        )
        for name in MODEL_NAMES
    }

    parameters = {
        name: []
        for name in MODEL_NAMES
    }

    neighbors = np.full(
        (REPEATS, n, 2),
        -1,
        dtype=int
    )

    margins = np.full(
        (REPEATS, n),
        np.nan
    )

    cv = RepeatedStratifiedKFold(
        n_splits=OUTER,
        n_repeats=REPEATS,
        random_state=SEED
    )

    first = None

    print("\n"+"="*95)
    print("REPEATED NESTED TRAIN-ONLY CV")
    print("="*95)

    for step, (it, iv) in enumerate(
        cv.split(np.zeros(n), y)
    ):

        r, f = divmod(step, OUTER)

        if first is None:
            first = (it.copy(), iv.copy())

        winners = inner_tune(
            D, y, it,
            seed=1000+step
        )

        row = []

        for name in MODEL_NAMES:

            kind, factors, C, gamma, inner, objective = (
                winners[name]
            )

            yp, B = pred_fold(
                D, y, it, iv,
                (kind, factors, C, gamma)
            )

            predictions[name][r, iv] = yp

            err = int(
                (yp != y[iv]).sum()
            )

            row.append(
                f"{name}={err}"
            )

            parameters[name].append({
                "factors": list(factors) if factors else [],
                "C": float(C),
                "gamma": float(gamma),
                "inner_F1": inner,
                "inner_objective": objective
            })

            # Train-only neighbor diagnostics for REG3.
            if name == REG3:

                for j, t in enumerate(iv):

                    same = np.flatnonzero(
                        y[it] == y[t]
                    )

                    rival = np.flatnonzero(
                        y[it] != y[t]
                    )

                    a = same[
                        np.argmin(B[j, same])
                    ]

                    b = rival[
                        np.argmin(B[j, rival])
                    ]

                    neighbors[r, t] = (
                        it[a], it[b]
                    )

                    margins[r, t] = (
                        (B[j, b]+1e-12) /
                        (B[j, a]+1e-12)
                    )

        print(
            f"Repeat {r+1}/{REPEATS} | "
            f"Fold {f+1}/{OUTER} | "
            + " | ".join(row),
            flush=True
        )

    # --------------------------------------------------------
    # Aggregated results
    # --------------------------------------------------------

    report = {}

    print("\n"+"="*95)
    print("AGGREGATED TRAIN-ONLY CV")
    print("="*95)

    for name in MODEL_NAMES:

        errors = [
            int(
                (predictions[name][r] != y).sum()
            )
            for r in range(REPEATS)
        ]

        f1s = [
            f1_score(
                y, predictions[name][r],
                average="macro",
                zero_division=0
            )
            for r in range(REPEATS)
        ]

        report[name] = {
            "errors": errors,
            "mean_errors": float(np.mean(errors)),
            "sd_errors": float(np.std(errors, ddof=1)),
            "accuracy": float(1-np.mean(errors)/n),
            "macro_f1": float(np.mean(f1s)),
            "params": parameters[name]
        }

        print(
            f"{name:28s} | "
            f"Errors={errors} | "
            f"Mean={np.mean(errors):.2f} ± "
            f"{np.std(errors, ddof=1):.2f} | "
            f"Acc={100*(1-np.mean(errors)/n):.4f}% | "
            f"F1={np.mean(f1s):.5f}"
        )

    # --------------------------------------------------------
    # Reproducibility controls
    # --------------------------------------------------------

    expected = {
        MODEL_NAMES[0]: [5, 5, 7, 4, 6],
        MODEL_NAMES[1]: [9, 5, 6, 9, 8],
        REF: [4, 6, 7, 5, 6]
    }

    print("\nHISTORICAL REPRODUCIBILITY")

    for name, reference in expected.items():

        observed = report[name]["errors"]

        print(
            f"{name} | "
            f"{'OK' if observed == reference else 'DIFFERENT'} | "
            f"expected={reference}"
        )

    if any(
        report[name]["errors"] != reference
        for name, reference in expected.items()
    ):

        raise RuntimeError(
            "Historical reproducibility failed. "
            "Do not interpret the weighting ablation."
        )

    # --------------------------------------------------------
    # Diagnostic figures
    # --------------------------------------------------------

    hard = make_figures(
        X, y, D,
        predictions,
        {
            name: report[name]["errors"]
            for name in MODEL_NAMES
        },
        parameters,
        first,
        neighbors,
        margins
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    np.savez_compressed(
        RESULTS/"haste_weights_oof.npz",
        y=y,
        **{
            f"model_{i}": predictions[name]
            for i, name in enumerate(MODEL_NAMES)
        }
    )

    output = {
        "protocol": "Train-only repeated nested CV 5x5; inner 3",
        "model_order": MODEL_NAMES,
        "core": CORE,
        "extra": EXTRA,
        "penalty": PENALTY,
        "factor_grid": FACTORS,
        "models": report,
        "hard_cases": hard,
        "official_test_access": False
    }

    path = RESULTS/"haste_weights_results.json"

    path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n"+"="*95)
    print("EXPERIMENT FINISHED")
    print("="*95)

    print(f"Saved: {path}")
    print("Saved: results/haste_weights_oof.npz")
    print("Saved: figures/haste_weights_01...11.png")


if __name__ == "__main__":
    main()
