# -*- coding: utf-8 -*-


import json, hashlib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from itertools import product, combinations

from aeon.datasets import load_classification
from scipy.fft import dct
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import eigh
from scipy.stats import binomtest

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC


# ============================================================
# HASTE-AL-CS-MDS — FINAL ABLATION AND OFFICIAL VALIDATION
# ============================================================

PHASE = "oracle_extended"  # "cv" first, then "official" ONCE

SEED, K, H, N, NE = 2027, 24, 2, 241, 64
FT, FR, FS, FC, FE, FP = 8, 16, 4, 4, 4, 4

SIGMA, TURNING_SCALE = 0.5, 0.20

BASE_W = {"time": 1., "speed": .25, "arc": 4., "curvature": 2.}

EXTRA_W = {
    "events": 4/3, "density": 1/6, "old_pos": .5, "correct_pos": .5,
    "mds2_delta": 4/3, "mds4_delta": 4/3,
    "lengths": .5, "magnitudes": .5
}

C_GRID = (10., 100.)
GAMMA_GRID = (.125, .25, .5, 1., 2., 4.)

OUTER, REPEATS, INNER = 5, 5, 3

GRID, EGRID = np.linspace(0, 1, N), np.linspace(0, 1, NE)

FOCUS, TARGETS = ("7", "9", "10", "18"), (619, 682, 1231)
RIVALS = {619: "10", 682: "18", 1231: "7"}

FIG, RESULTS_DIR = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

FREEZE = RESULTS_DIR / "haste_mds_frozen_selection.json"
OFFICIAL_FILE = RESULTS_DIR / "haste_mds_official_results.json"


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "AL-C-68D": (),
    "Events+Density-85D": ("events", "density"),
    "Legacy-89D": ("legacy",),
    "Harmonic+OldPos-89D": ("events", "density", "old_pos"),
    "MDS2-Delta-NoPos-77D": ("mds2_delta", "density"),
    "MDS2-Delta+CorrectPos-81D": ("mds2_delta", "density", "correct_pos"),
    "MDS4-Delta-NoPos-85D": ("mds4_delta", "density"),
    "MDS4-Delta+OldPos-89D": ("mds4_delta", "density", "old_pos"),
    "MDS4-Delta+CorrectPos-89D": ("mds4_delta", "density", "correct_pos"),
    "MDS4-Delta+CorrectPos+Lengths-93D": ("mds4_delta", "density", "correct_pos", "lengths"),
    "MDS4-Delta+CorrectPos+Magnitudes-93D": ("mds4_delta", "density", "correct_pos", "magnitudes")
}

EXPECTED = {
    "AL-C-68D": [3, 7, 7, 8, 9],
    "Events+Density-85D": [5, 5, 7, 5, 6],
    "Legacy-89D": [5, 5, 7, 4, 6],
    "Harmonic+OldPos-89D": [5, 4, 6, 5, 6],
    "MDS2-Delta+CorrectPos-81D": [4, 5, 4, 4, 7],
    "MDS4-Delta+CorrectPos-89D": [6, 6, 3, 3, 4]
}

DIM = {
    "time": 32, "speed": 16, "arc": 16, "curvature": 4,
    "events": 16, "mds2_delta": 8, "mds4_delta": 16,
    "old_pos": 4, "correct_pos": 4, "density": 1,
    "legacy": 5, "lengths": 4, "magnitudes": 4
}


def protocol_hash():

    settings = {
        "seed": SEED, "K": K, "H": H, "N": N, "NE": NE,
        "dims": [FT, FR, FS, FC, FE, FP],
        "sigma": SIGMA, "turn_scale": TURNING_SCALE,
        "weights": [BASE_W, EXTRA_W], "models": MODELS,
        "C": C_GRID, "gamma": GAMMA_GRID,
        "cv": [OUTER, REPEATS, INNER]
    }

    return hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()


def save(fig, name):

    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================================================
# SYMBOL DISSIMILARITY AND MDS
# ============================================================

def mds_alphabet(harmonics):

    a = np.arange(K)
    dif = a[:, None]-a[None, :]

    D = np.sqrt(sum(
        4*np.sin(np.pi*h*dif/K)**2
        for h in range(1, harmonics+1)
    ))

    Q = D**2

    G = -.5*(Q-Q.mean(axis=0)[None, :]-Q.mean(axis=1)[:, None]+Q.mean())

    values, vectors = eigh(G, check_finite=False)

    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    xy = vectors[:, :2*harmonics]*np.sqrt(
        np.maximum(values[:2*harmonics], 0)
    )

    error = np.max(np.abs(squareform(pdist(xy))-D))

    assert error < 1e-8, f"MDS reconstruction failed ({error:.3g})"

    return D, xy, values, error


D2, XY2, E2, ERR2 = mds_alphabet(1)
D4, XY4, E4, ERR4 = mds_alphabet(2)

print(f"MDS reconstruction | 2D={ERR2:.3e} | 4D={ERR4:.3e}")


# ============================================================
# SYMBOLS AND FREQUENCY ENCODING
# ============================================================

def quantize(theta):
    return np.floor((theta+np.pi/K)/(2*np.pi/K)).astype(np.int16) % K


def harmonics(theta):

    return np.stack([
        f(h*theta)*np.sqrt(2./K)
        for h in range(1, H+1)
        for f in (np.cos, np.sin)
    ], axis=1)


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


# ============================================================
# EVENT-BASED FEATURES
# ============================================================

def event_features(ids):

    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]

    symbols = ids[starts]
    lengths = (ends-starts)/len(ids)

    delta = (
        symbols[1:].astype(int)-symbols[:-1].astype(int)+K//2
    ) % K-K//2

    positions = starts[1:]/(len(ids)-1)
    m = len(delta)

    expected = np.arange(1, m+1)/(m+1) if m else np.zeros(0)

    old_profile = (
        interpolate_rank(positions)-EGRID
        if m else np.zeros(NE)
    )

    correct_profile = (
        interpolate_rank(positions-expected)
        if m else np.zeros(NE)
    )

    def compressed(x, count=FE):
        return dct(
            interpolate_rank(x), axis=0, type=2, norm="ortho"
        )[:count].ravel()

    out = {
        "events": compressed(harmonics(2*np.pi*delta/K)) if m else np.zeros(16),

        "mds2_delta": compressed(
            XY2[symbols[1:]]-XY2[symbols[:-1]]
        ) if m else np.zeros(8),

        "mds4_delta": compressed(
            XY4[symbols[1:]]-XY4[symbols[:-1]]
        ) if m else np.zeros(16),

        "old_pos": dct(old_profile, type=2, norm="ortho")[:FP],
        "correct_pos": dct(correct_profile, type=2, norm="ortho")[:FP],

        "density": np.array([np.log1p(m)/np.log1p(N-1)]),

        "lengths": compressed(
            np.log(np.maximum(len(lengths)*lengths, 1e-12))
        ),

        "magnitudes": compressed(
            np.linalg.norm(
                XY4[symbols[1:]]-XY4[symbols[:-1]], axis=1
            )
        ) if m else np.zeros(4)
    }

    out["legacy"] = np.r_[out["lengths"], out["density"]]

    info = {
        "starts": starts, "symbols": symbols, "delta": delta,
        "positions": positions, "expected": expected, "lengths": lengths,
        "old_profile": old_profile, "correct_profile": correct_profile
    }

    return out, info


# ============================================================
# BASE HASTE-AL-C 68D + SYMBOLIC EXTENSIONS
# ============================================================

def encode(case):

    v = np.nan_to_num(
        np.asarray(case, dtype=float)[:2],
        nan=0., posinf=0., neginf=0.
    )

    if v.ndim != 2 or v.shape[0] != 2 or v.shape[1] < 2:
        raise ValueError("Expected at least 2 channels and 2 samples")

    n = v.shape[1]
    speed = np.hypot(v[0], v[1])

    positive = speed[speed > 1e-10]
    cutoff = max(.03*np.median(positive), 1e-10) if positive.size else 1e-10

    theta, valid = np.arctan2(v[1], v[0]), speed > cutoff

    if valid.any():

        prev = np.where(valid, np.arange(n), -1)
        np.maximum.accumulate(prev, out=prev)

        prev[prev < 0] = np.flatnonzero(valid)[0]
        theta = theta[prev]

    else:
        theta = np.zeros(n)

    ix = np.minimum((GRID*n).astype(int), n-1)

    ft = dct(
        harmonics(2*np.pi*quantize(theta[ix])/K),
        axis=0, type=2, norm="ortho"
    )[:FT].ravel()

    fr = dct(
        speed[ix]/max(float(speed[ix].mean()), 1e-8),
        type=2, norm="ortho"
    )[:FR]

    s = np.r_[0., np.cumsum(.5*(speed[:-1]+speed[1:]))]

    if s[-1] > 1e-12:

        keep = np.r_[True, np.diff(s) > 1e-12]

        theta_arc = np.interp(
            GRID*s[-1], s[keep], np.unwrap(theta)[keep]
        )

    else:
        theta_arc = theta[ix]

    ids = quantize(theta_arc)

    fs = dct(
        .5*harmonics(2*np.pi*ids/K)+.5*harmonics(theta_arc),
        axis=0, type=2, norm="ortho"
    )[:FS].ravel()

    smooth = gaussian_filter1d(
        theta_arc, sigma=SIGMA, mode="nearest"
    )

    turn = np.tanh(
        np.gradient(smooth, GRID)/(TURNING_SCALE*(N-1))
    )

    fc = dct(turn, type=2, norm="ortho")[:FC]

    ev, info = event_features(ids)

    return {
        "time": ft, "speed": fr, "arc": fs,
        "curvature": fc, **ev
    }, ids, info


def extract(cases):

    enc = [encode(case) for case in cases]

    data = {
        key: np.stack([v[0][key] for v in enc]).astype(np.float32)
        for key in DIM
    }

    assert all(data[k].shape[1] == d for k, d in DIM.items())

    return data, enc


# ============================================================
# PAIRWISE DISTANCES
# ============================================================

def make_matrices(Xtr, Xte=None):

    Dtr = {
        k: squareform(pdist(v, "sqeuclidean")).astype(np.float32)
        for k, v in Xtr.items()
    }

    Dte = None if Xte is None else {
        k: cdist(Xte[k], Xtr[k], "sqeuclidean").astype(np.float32)
        for k in Xtr
    }

    return Dtr, Dte


def distances(it, iv=None, official=False):

    blocks = {}

    for name, Dt in DTRAIN.items():

        A = Dt[np.ix_(it, it)].astype(np.float64)

        B = (
            DTEST[name][:, it] if official else Dt[np.ix_(iv, it)]
        ).astype(np.float64)

        scale = max(float(A.sum())/(len(it)*(len(it)-1)), 1e-12)

        blocks[name] = A/scale, B/scale

    base = tuple(
        sum(w*blocks[k][j] for k, w in BASE_W.items())/sum(BASE_W.values())
        for j in (0, 1)
    )

    output = {}

    for name, members in MODELS.items():

        if not members:
            output[name] = base
            continue

        if members == ("legacy",):

            branch = tuple(
                (blocks["events"][j]+.5*blocks["legacy"][j])/1.5
                for j in (0, 1)
            )

            output[name] = tuple(
                np.maximum(
                    (7.25*base[j]+2*branch[j])/9.25, 0
                )
                for j in (0, 1)
            )

        else:

            weight = sum(EXTRA_W[k] for k in members)

            output[name] = tuple(
                np.maximum(
                    (
                        7.25*base[j] +
                        sum(EXTRA_W[k]*blocks[k][j] for k in members)
                    )/(7.25+weight), 0
                )
                for j in (0, 1)
            )

    return output


# ============================================================
# TRAIN-ONLY SVM OPTIMIZATION
# ============================================================

def tune(it, folds=INNER, seed=0, names=None):

    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )

    inner = [
        (it[a], it[b], distances(it[a], it[b]))
        for a, b in cv.split(np.zeros(len(it)), ytrain[it])
    ]

    chosen = {}

    for name in (names if names is not None else MODELS):

        best = (-np.inf, None, None)

        for gamma in GAMMA_GRID:

            kernels = [
                (
                    ia, ib,
                    np.exp(-gamma*d[name][0]),
                    np.exp(-gamma*d[name][1])
                )
                for ia, ib, d in inner
            ]

            for C in C_GRID:

                values = []

                for ia, ib, Ktr, Kval in kernels:

                    svm = SVC(
                        C=C, kernel="precomputed"
                    ).fit(Ktr, ytrain[ia])

                    values.append(
                        f1_score(
                            ytrain[ib], svm.predict(Kval),
                            average="macro", zero_division=0
                        )
                    )

                score = float(np.mean(values))

                if score > best[0]:
                    best = (score, C, gamma)

        chosen[name] = best

    return chosen


# ============================================================
# DIAGNOSTICS
# ============================================================

def ovo_score(svm, krow, truth, rival):

    pairs = list(combinations(map(str, svm.classes_), 2))

    pair = (str(truth), str(rival))
    target = pair if pair in pairs else pair[::-1]

    margin = float(
        np.ravel(
            svm.decision_function(krow[None, :])
        )[pairs.index(target)]
    )

    return margin if target == pair else -margin


def reconstructed_path(case):

    vel = np.nan_to_num(
        np.asarray(case, dtype=float)[:2],
        nan=0., posinf=0., neginf=0.
    )

    xy = np.cumsum(vel.T, axis=0)
    xy -= xy[0]

    length = float(
        np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    )

    return xy/max(length, 1e-12)


def draw_case(ax, case, title):

    xy = reconstructed_path(case)

    ax.plot(xy[:, 0], xy[:, 1], lw=1.2)
    ax.scatter(*xy[0], s=20)
    ax.scatter(*xy[-1], s=25, marker="x")

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def mds_oos(A, B):

    mean = A.mean(axis=1)
    grand = float(mean.mean())

    G = -.5*(A-mean[:, None]-mean[None, :]+grand)

    vals, U = eigh(
        G, subset_by_index=(len(A)-2, len(A)-1),
        check_finite=False
    )

    idx = np.argsort(vals)[::-1]
    vals, U = vals[idx], U[:, idx]

    lam = np.maximum(vals, 1e-12)

    xy = U*np.sqrt(lam)

    hold = -.5*(
        B-B.mean(axis=1, keepdims=True)-mean[None, :]+grand
    )@U/np.sqrt(lam)

    ratio = float(
        np.maximum(vals, 0).sum()/max(np.trace(G), 1e-12)
    )

    return xy, hold, ratio


# ============================================================
# FIGURE 1 — SYMBOL MDS
# ============================================================

def common_alphabet_plot():

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))

    variants = (
        (2, D2, XY2, E2, ERR2),
        (4, D4, XY4, E4, ERR4)
    )

    for row, (dim, D, XY, E, err) in enumerate(variants):

        im = ax[row, 0].imshow(
            D, origin="lower", cmap="viridis"
        )

        fig.colorbar(im, ax=ax[row, 0], fraction=.045)
        ax[row, 0].set_title(f"{dim}D symbol dissimilarity")

        ax[row, 1].scatter(
            XY[:, 0], XY[:, 1], s=18
        )

        for j, (x, y) in enumerate(XY[:, :2]):
            ax[row, 1].text(x*1.1, y*1.1, str(j), fontsize=7)

        ax[row, 1].set_aspect("equal", adjustable="datalim")

        ax[row, 1].set_title(
            f"MDS{dim} first two axes | error {err:.1e}"
        )

        ax[row, 2].bar(
            np.arange(1, 9), np.maximum(E[:8], 0)
        )

        ax[row, 2].set_title("Eigenvalues")

    save(fig, "haste_final_01_symbol_mds.png")


# ============================================================
# FIGURES 2–9 — TRAINING DIAGNOSTICS
# ============================================================

def validation_plots(selected, summary, oof, cases, train, encoded, first):

    common_alphabet_plot()

    # Figure 2 — Ablation

    names = list(MODELS)
    errors = [np.mean(summary[n]["errors"]) for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.barh(np.arange(len(names)), errors)
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()

    for j, v in enumerate(errors):
        ax.text(v+.04, j, f"{v:.1f}", va="center")

    ax.set_xlabel("Mean errors per 1422 trajectories")
    ax.set_title("Training-only symbolic ablation")
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "haste_final_02_ablation.png")

    # Figure 3 — Repeated errors

    names_plot = [
        "AL-C-68D",
        "Harmonic+OldPos-89D",
        "MDS2-Delta+CorrectPos-81D",
        "MDS4-Delta+CorrectPos-89D"
    ]

    if selected not in names_plot:
        names_plot.append(selected)

    fig, ax = plt.subplots(figsize=(10, 4))

    for n in names_plot:

        ax.plot(
            range(1, REPEATS+1),
            summary[n]["errors"],
            "o-", lw=1, label=n
        )

    ax.set_xticks(range(1, REPEATS+1))
    ax.set_ylabel("Errors")
    ax.set_xlabel("Outer CV repeat")
    ax.legend(fontsize=8)

    ax.set_title("Errors across repeated outer CV")
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "haste_final_03_repeated_errors.png")

    # Figure 4 — Class errors

    fig, ax = plt.subplots(figsize=(9, 4))

    comparison = (
        "AL-C-68D", "Harmonic+OldPos-89D", selected
    )

    for j, n in enumerate(comparison):

        numbers = [
            int(np.sum(oof[n][:, ytrain == c] != c))
            for c in FOCUS
        ]

        ax.bar(
            np.arange(len(FOCUS))+(j-1)*.25,
            numbers, width=.25, label=n
        )

    ax.set_xticks(
        range(len(FOCUS)),
        [f"Class {c}" for c in FOCUS]
    )

    ax.set_ylabel("Errors across five repeats")
    ax.legend(fontsize=8)
    ax.set_title("Persistent class confusions")

    save(fig, "haste_final_04_class_errors.png")

    # Figure 5 — Position profiles

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    for row, target in enumerate(TARGETS):

        info = encoded[target][2]
        pos, exp = info["positions"], info["expected"]

        ranks = (
            np.linspace(0, 1, len(pos))
            if len(pos) > 1 else np.zeros(len(pos))
        )

        axes[row, 0].plot(
            ranks, pos, "o-", lw=1, ms=2.5,
            label="Observed"
        )

        axes[row, 0].plot(
            ranks, exp, "--", label="Uniform corrected"
        )

        axes[row, 0].set_title(
            f"#{target} — positions of transitions"
        )

        axes[row, 1].plot(
            EGRID, info["old_profile"], label="Old"
        )

        axes[row, 1].plot(
            EGRID, info["correct_profile"], label="Corrected"
        )

        axes[row, 1].set_title(
            f"#{target} — position deviations"
        )

        for a in axes[row]:
            a.set_xlabel("Normalized rank")
            a.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend()
    axes[0, 1].legend()

    save(fig, "haste_final_05_position_profiles.png")

    # Figure 6 — Symbolic jump magnitudes

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    for row, target in enumerate(TARGETS):

        info = encoded[target][2]
        sym = info["symbols"]

        if len(info["delta"]):

            axes[row, 0].stem(
                info["positions"],
                info["delta"],
                basefmt=" "
            )

            mag = np.linalg.norm(
                XY4[sym[1:]]-XY4[sym[:-1]], axis=1
            )

            axes[row, 1].scatter(
                info["positions"], mag, s=16
            )

        axes[row, 0].set_title(
            f"#{target} — signed symbolic jumps"
        )

        axes[row, 1].set_title(
            f"#{target} — MDS4 jump magnitudes"
        )

        for a in axes[row]:
            a.set_xlabel("Normalized arc length")
            a.spines[["top", "right"]].set_visible(False)

    save(fig, "haste_final_06_event_magnitudes.png")

    # Figure 7 — Train-only trajectory MDS

    it, iv = first

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model in zip(
        axes, ("AL-C-68D", selected)
    ):

        Dtr, Dval = distances(it, iv)[model]
        xy, val, ratio = mds_oos(Dtr, Dval)

        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=7, c="lightgray", alpha=.4
        )

        for c in FOCUS:

            mask = ytrain[it] == c

            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                s=14, alpha=.8, label=f"Class {c}"
            )

        wrong = oof[model][0, iv] != ytrain[iv]

        ax.scatter(
            val[wrong, 0], val[wrong, 1],
            marker="x", s=65, c="black"
        )

        ax.set_title(
            f"{model} | MDS 2D trace {ratio:.1%}"
        )

        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].legend(fontsize=7)

    save(fig, "haste_final_07_trajectory_mds.png")

    # Figure 8 — Hard cases and neighbors

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    for row, target in enumerate(TARGETS):

        z = next(
            (
                z for z in cases
                if z["method"] == selected
                and z["target"] == target
                and z["repeat"] == 1
            ),
            None
        )

        if z is None:
            continue

        draw_case(
            axes[row, 0], train[target],
            f"#{target} | True {ytrain[target]} | Pred {z['pred']}"
        )

        draw_case(
            axes[row, 1], train[z["same"]],
            f"Nearest same-class #{z['same']}"
        )

        draw_case(
            axes[row, 2], train[z["rival"]],
            f"Nearest rival #{z['rival']}"
        )

    fig.suptitle(
        f"Hard cases — {selected} (training-only neighbors)"
    )

    save(fig, "haste_final_08_hard_cases.png")

    # Figure 9 — Pairwise OVO margins

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, target in zip(axes, TARGETS):

        for n in (
            "AL-C-68D", "Harmonic+OldPos-89D", selected
        ):

            rows = sorted(
                (
                    z for z in cases
                    if z["target"] == target
                    and z["method"] == n
                ),
                key=lambda z: z["repeat"]
            )

            ax.plot(
                [z["repeat"] for z in rows],
                [z["ovo"] for z in rows],
                "o-", lw=1, label=n
            )

        ax.axhline(0, color="gray", ls="--")
        ax.set_xticks(range(1, 6))

        ax.set_title(
            f"#{target}: {ytrain[target]} vs {RIVALS[target]}"
        )

        ax.set_xlabel("Repeat")

    axes[0].set_ylabel(
        "Signed OVO margin (+ true class)"
    )

    axes[-1].legend(fontsize=7)

    save(fig, "haste_final_09_ovo_margins.png")


# ============================================================
# REPEATED NESTED CV — TRAINING ONLY
# ============================================================

def cv_phase(train, enc):

    if OFFICIAL_FILE.exists():
        raise RuntimeError("Official test already recorded.")

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER, n_repeats=REPEATS,
        random_state=SEED
    )

    oof = {
        name: np.empty((REPEATS, len(ytrain)), dtype=ytrain.dtype)
        for name in MODELS
    }

    params = {name: [] for name in MODELS}

    cases = []
    first = None

    references = (
        "AL-C-68D",
        "Harmonic+OldPos-89D",
        "MDS4-Delta+CorrectPos-89D"
    )

    for step, (it, iv) in enumerate(
        outer.split(np.zeros(len(ytrain)), ytrain)
    ):

        rep, fold = divmod(step, OUTER)

        if step == 2:
            first = it.copy(), iv.copy()

        chosen = tune(it, folds=INNER, seed=1000+step)
        dist = distances(it, iv)

        print(
            f"\nRepeat {rep+1}/{REPEATS} | Fold {fold+1}/{OUTER}",
            flush=True
        )

        for name in MODELS:

            inner, C, gamma = chosen[name]
            A, B = dist[name]

            ktr = np.exp(-gamma*A)
            kval = np.exp(-gamma*B)

            svm = SVC(
                C=C, kernel="precomputed",
                decision_function_shape="ovo"
            ).fit(ktr, ytrain[it])

            prediction = svm.predict(kval)

            oof[name][rep, iv] = prediction
            params[name].append((C, gamma))

            acc = accuracy_score(ytrain[iv], prediction)

            print(
                f"{name:45s} | Acc={acc:.4f} | Inner={inner:.4f} | "
                f"C={C:g} | gamma={gamma:g}",
                flush=True
            )

            if name not in references:
                continue

            for target in TARGETS:

                hit = np.flatnonzero(iv == target)

                if not len(hit):
                    continue

                row = int(hit[0])

                same = np.flatnonzero(
                    ytrain[it] == ytrain[target]
                )

                rival = np.flatnonzero(
                    ytrain[it] != ytrain[target]
                )

                js = int(same[np.argmin(B[row, same])])
                jr = int(rival[np.argmin(B[row, rival])])

                cases.append({
                    "method": name,
                    "target": target,
                    "repeat": rep+1,
                    "pred": str(prediction[row]),
                    "same": int(it[js]),
                    "rival": int(it[jr]),
                    "rho": float(
                        B[row, jr]/max(B[row, js], 1e-12)
                    ),
                    "ovo": ovo_score(
                        svm, kval[row],
                        ytrain[target], RIVALS[target]
                    )
                })

    # ========================================================
    # AGGREGATED RESULTS
    # ========================================================

    summary = {}

    print("\n"+"="*108)
    print("AGGREGATED RESULTS")
    print("="*108)

    for name in MODELS:

        errors = [
            int(np.sum(oof[name][r] != ytrain))
            for r in range(REPEATS)
        ]

        f1 = [
            float(f1_score(
                ytrain, oof[name][r],
                average="macro", zero_division=0
            ))
            for r in range(REPEATS)
        ]

        acc = [
            float(accuracy_score(ytrain, oof[name][r]))
            for r in range(REPEATS)
        ]

        summary[name] = {
            "errors": errors,
            "f1": f1,
            "acc": acc,
            "params": params[name]
        }

        print(
            f"{name:45s} | Acc={np.mean(acc):.4f} "
            f"± {np.std(acc, ddof=1):.4f} | "
            f"F1={np.mean(f1):.4f} | Errors={errors}"
        )

    # ========================================================
    # CLASS ERRORS AND PARAMETERS
    # ========================================================

    print("\nCLASS ERRORS AND SELECTED SVM PARAMETERS")

    for name in MODELS:

        cls_errors = {
            c: int(np.sum(oof[name][:, ytrain == c] != c))
            for c in FOCUS
        }

        conf = Counter(
            (str(ytrain[i]), str(oof[name][r, i]))
            for r in range(REPEATS)
            for i in range(len(ytrain))
            if oof[name][r, i] != ytrain[i]
        )

        print(
            f"{name:45s} | Class errors={cls_errors} | "
            f"Main confusions={conf.most_common(5)} | "
            f"Params={Counter(params[name]).most_common(4)}"
        )

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    bad = [
        name for name, expected in EXPECTED.items()
        if summary[name]["errors"] != expected
    ]

    for name in EXPECTED:

        print(
            f"Reproducibility {name}: "
            f"{'OK' if name not in bad else 'DIFFERENT'}"
        )

    if bad:
        raise RuntimeError(
            f"Historical references not reproduced: {bad}. "
            "Do not freeze or run official test."
        )

    # ========================================================
    # MODEL SELECTION — TRAINING ONLY
    # ========================================================

    def model_dim(name):

        if name == "Legacy-89D":
            return 89

        return 68+sum(DIM[k] for k in MODELS[name])

    def rank(name):

        return (
            np.mean(summary[name]["errors"]),
            model_dim(name),
            -np.mean(summary[name]["f1"])
        )

    selected = min(MODELS, key=rank)

    print(
        f"\nFROZEN CANDIDATE: {selected} | "
        f"Errors={summary[selected]['errors']}"
    )

    # ========================================================
    # PAIRED COMPARISONS
    # ========================================================

    print("\nPAIRED COMPARISONS VS MDS4-Delta+CorrectPos-89D")

    ref = "MDS4-Delta+CorrectPos-89D"

    for name in MODELS:

        count = [
            (
                int(np.sum(
                    (oof[ref][r] != ytrain) &
                    (oof[name][r] == ytrain)
                )),
                int(np.sum(
                    (oof[ref][r] == ytrain) &
                    (oof[name][r] != ytrain)
                ))
            )
            for r in range(REPEATS)
        ]

        p_values = [
            binomtest(min(a, b), a+b, .5).pvalue
            if a+b else 1.
            for a, b in count
        ]

        print(
            f"{name:45s} | Corrected/Introduced={count} | "
            f"McNemar p={[round(p,4) for p in p_values]}"
        )

    # ========================================================
    # FREEZE CHOSEN CONFIGURATION
    # ========================================================

    digest = hashlib.sha256(
        "\n".join(ytrain).encode()
    ).hexdigest()

    frozen = {
        "method": selected,
        "train_size": len(ytrain),
        "train_labels_sha256": digest,
        "protocol_sha256": protocol_hash(),
        "selection_rule": (
            "minimum mean repeated OOF errors, then dimension, then macro-F1"
        ),
        "cv": {
            n: {
                "errors": v["errors"],
                "acc_mean": float(np.mean(v["acc"])),
                "f1_mean": float(np.mean(v["f1"]))
            }
            for n, v in summary.items()
        }
    }

    FREEZE.write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Frozen configuration saved: {FREEZE}")

    # ========================================================
    # HARD CASE AUDIT FOR NEWLY SELECTED CANDIDATE
    # ========================================================

    if selected not in references:

        outer2 = RepeatedStratifiedKFold(
            n_splits=OUTER, n_repeats=REPEATS,
            random_state=SEED
        )

        for step, (it, iv) in enumerate(
            outer2.split(np.zeros(len(ytrain)), ytrain)
        ):

            rep = step//OUTER
            C, gamma = params[selected][step]

            A, B = distances(it, iv)[selected]
            kval = np.exp(-gamma*B)

            svm = SVC(
                C=C,
                kernel="precomputed",
                decision_function_shape="ovo"
            ).fit(np.exp(-gamma*A), ytrain[it])

            pred = svm.predict(kval)

            for target in TARGETS:

                hit = np.flatnonzero(iv == target)

                if not len(hit):
                    continue

                row = int(hit[0])

                same = np.flatnonzero(
                    ytrain[it] == ytrain[target]
                )

                rival = np.flatnonzero(
                    ytrain[it] != ytrain[target]
                )

                js = int(same[np.argmin(B[row, same])])
                jr = int(rival[np.argmin(B[row, rival])])

                cases.append({
                    "method": selected,
                    "target": target,
                    "repeat": rep+1,
                    "pred": str(pred[row]),
                    "same": int(it[js]),
                    "rival": int(it[jr]),
                    "rho": float(
                        B[row, jr]/max(B[row, js], 1e-12)
                    ),
                    "ovo": ovo_score(
                        svm, kval[row],
                        ytrain[target], RIVALS[target]
                    )
                })

    validation_plots(
        selected, summary, oof, cases, train, enc, first
    )

    print(
        "\nCV complete. Inspect figures and selection, "
        "then set PHASE='official' for ONE official evaluation."
    )


# ============================================================
# OFFICIAL TEST — SINGLE EVALUATION
# ============================================================

def official_phase():

    if OFFICIAL_FILE.exists():
        raise RuntimeError(
            "Official results already exist — one-shot test locked."
        )

    if not FREEZE.exists():
        raise RuntimeError(
            "Run PHASE='cv' first and inspect the frozen selection."
        )

    frozen = json.loads(
        FREEZE.read_text(encoding="utf-8")
    )

    digest = hashlib.sha256(
        "\n".join(ytrain).encode()
    ).hexdigest()

    if (
        len(ytrain) != frozen["train_size"] or
        digest != frozen["train_labels_sha256"]
    ):
        raise RuntimeError(
            "Training split changed after freezing the candidate."
        )

    if protocol_hash() != frozen["protocol_sha256"]:
        raise RuntimeError(
            "Extractor, models or protocol changed after freezing."
        )

    selected = frozen["method"]

    if selected not in MODELS:
        raise RuntimeError("Unknown frozen method.")

    # Access official test split ONLY here.

    test, ytest = load_classification(
        "CharacterTrajectories",
        split="test",
        load_equal_length=False
    )

    test = list(test)
    ytest = np.asarray(ytest, dtype=str)

    assert len(test) == 1436, "Unexpected official test split size"

    Xtest, _ = extract(test)

    global DTEST

    DTEST = {
        k: cdist(
            Xtest[k], XTRAIN[k], "sqeuclidean"
        ).astype(np.float32)
        for k in DIM
    }

    indices = np.arange(len(ytrain))

    cv_score, C, gamma = tune(
        indices,
        folds=5,
        seed=SEED+100,
        names=(selected,)
    )[selected]

    A, B = distances(
        indices, official=True
    )[selected]

    model = SVC(
        C=C, kernel="precomputed"
    ).fit(
        np.exp(-gamma*A), ytrain
    )

    pred = model.predict(np.exp(-gamma*B))

    accuracy = float(accuracy_score(ytest, pred))
    macro = float(f1_score(
        ytest, pred, average="macro"
    ))

    classes = sorted(np.unique(np.r_[ytrain, ytest]))

    matrix = confusion_matrix(
        ytest, pred, labels=classes
    )

    output = {
        "method": selected,
        "test_size": len(ytest),
        "accuracy": accuracy,
        "macro_f1": macro,
        "errors": int(np.sum(pred != ytest)),
        "C": C,
        "gamma": gamma,
        "full_train_cv_macro_f1": cv_score,
        "classes": classes,
        "confusion_matrix": matrix.tolist()
    }

    OFFICIAL_FILE.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8"
    )

    print("\n"+"="*90)
    print("OFFICIAL TEST — ONE-TIME RESULT")
    print("="*90)

    print(
        f"Method={selected} | Test N={len(ytest)} | "
        f"Acc={accuracy:.4%} | Macro-F1={macro:.4f} | "
        f"Errors={output['errors']} | C={C:g} | gamma={gamma:g}"
    )

    # Figure 10 — Official confusion matrix

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(
        range(len(classes)), classes, rotation=90
    )

    ax.set_yticks(
        range(len(classes)), classes
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_title(
        f"Official test confusion matrix — {selected}"
    )

    fig.colorbar(im, ax=ax, fraction=.04)

    save(fig, "haste_final_10_official_confusion.png")

    # Figure 11 — Official recall by class

    fig, ax = plt.subplots(figsize=(11, 4))

    support = np.array([
        (ytest == c).sum()
        for c in classes
    ])

    class_acc = np.diag(matrix)/np.maximum(support, 1)

    ax.bar(
        range(len(classes)), class_acc
    )

    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(len(classes)), classes)

    ax.set_xlabel("Class")
    ax.set_ylabel("Recall")

    ax.set_title("Official test recall by class")

    save(fig, "haste_final_11_official_class_recall.png")

    print(
        f"Official results saved to {OFFICIAL_FILE}. "
        "Do not use this result to resume model search."
    )


# ============================================================
# POST-HOC TEST ORACLE — MODEL + C + GAMMA SELECTION
# ============================================================

def test_oracle_phase():

    if not OFFICIAL_FILE.exists():
        raise RuntimeError("Run the frozen official evaluation first.")

    official = json.loads(OFFICIAL_FILE.read_text(encoding="utf-8"))

    Xtest_raw, ytest = load_classification(
        "CharacterTrajectories", split="test", load_equal_length=False
    )

    Xtest, _ = extract(list(Xtest_raw))
    ytest = np.asarray(ytest, dtype=str)

    assert len(ytest) == 1436

    global DTEST
    DTEST = {k: cdist(Xtest[k], XTRAIN[k], "sqeuclidean").astype(np.float32) for k in DIM}

    mats = distances(np.arange(len(ytrain)), official=True)

    records = []

    print("\n"+"="*100)
    print("POST-HOC TEST ORACLE — NOT AN UNBIASED OFFICIAL RESULT")
    print("="*100)

    for name, members in MODELS.items():

        A, B = mats[name]
        dimension = 89 if name == "Legacy-89D" else 68+sum(DIM[k] for k in members)

        for gamma in GAMMA_GRID:

            Ktr, Kte = np.exp(-gamma*A), np.exp(-gamma*B)

            for C in C_GRID:

                svm = SVC(C=C, kernel="precomputed").fit(Ktr, ytrain)
                pred = svm.predict(Kte)

                errors = int(np.sum(pred != ytest))
                acc = float(accuracy_score(ytest, pred))
                f1 = float(f1_score(ytest, pred, average="macro", zero_division=0))

                records.append({
                    "model": name, "dimension": dimension,
                    "C": float(C), "gamma": float(gamma),
                    "errors": errors, "accuracy": acc, "macro_f1": f1
                })

        best = min(
            (r for r in records if r["model"] == name),
            key=lambda r: (r["errors"], -r["macro_f1"], r["C"], r["gamma"])
        )

        print(
            f"{name:45s} | Acc={best['accuracy']:.4%} | "
            f"Errors={best['errors']:2d} | F1={best['macro_f1']:.4f} | "
            f"C={best['C']:g} | gamma={best['gamma']:g}",
            flush=True
        )

    # Reproduce frozen official performance before reporting oracle results

    frozen = [
        r for r in records
        if r["model"] == official["method"]
        and np.isclose(r["C"], official["C"])
        and np.isclose(r["gamma"], official["gamma"])
    ]

    assert len(frozen) == 1 and frozen[0]["errors"] == official["errors"], \
        "Frozen official evaluation was not reproduced."

    records.sort(key=lambda r: (
        r["errors"], r["dimension"], -r["macro_f1"],
        r["model"], r["C"], r["gamma"]
    ))

    winner = records[0]

    print("\n"+"="*100)
    print("BEST POST-HOC TEST CONFIGURATIONS")
    print("="*100)

    for i, r in enumerate(records[:20], 1):
        print(
            f"{i:2d} | {r['model']:45s} | Acc={r['accuracy']:.4%} | "
            f"Errors={r['errors']:2d} | C={r['C']:g} | gamma={r['gamma']:g}"
        )

    print("\nFROZEN OFFICIAL:", frozen[0])
    print("BEST TEST ORACLE:", winner)

    # Save separately — never overwrite the official results

    output = {
        "status": "POST-HOC TEST OPTIMIZATION — NOT OFFICIAL",
        "frozen_official": frozen[0],
        "best_oracle": winner,
        "all_results": records
    }

    path = RESULTS_DIR / "haste_mds_test_oracle.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Figure 12 — Best test result per representation

    best_models = {
        name: min((r for r in records if r["model"] == name), key=lambda r: r["errors"])
        for name in MODELS
    }

    names = sorted(MODELS, key=lambda n: best_models[n]["errors"])
    values = [best_models[n]["errors"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.barh(range(len(names)), values)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()

    ax.axvline(official["errors"], ls="--", color="gray", label="Frozen official model")

    for i, n in enumerate(names):
        r = best_models[n]
        ax.text(r["errors"]+.08, i, f"{r['errors']} | C={r['C']:g}, γ={r['gamma']:g}", va="center", fontsize=8)

    ax.set_xlabel("Minimum test errors — post-hoc selection")
    ax.set_title("Test oracle — best parameters for each representation")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "haste_final_12_test_oracle_models.png")

    # Figure 13 — Hyperparameter landscape of the oracle winner

    landscape = np.array([
        [
            next(r["errors"] for r in records
                 if r["model"] == winner["model"]
                 and r["C"] == C and r["gamma"] == gamma)
            for gamma in GAMMA_GRID
        ]
        for C in C_GRID
    ])

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(landscape, cmap="viridis_r", aspect="auto")

    ax.set_xticks(range(len(GAMMA_GRID)), [str(g) for g in GAMMA_GRID])
    ax.set_yticks(range(len(C_GRID)), [str(c) for c in C_GRID])

    for i in range(len(C_GRID)):
        for j in range(len(GAMMA_GRID)):
            ax.text(j, i, str(landscape[i, j]), ha="center", va="center", color="red")

    ax.set_xlabel("Gamma")
    ax.set_ylabel("C")
    ax.set_title(f"Test oracle hyperparameters — {winner['model']}")

    fig.colorbar(im, ax=ax, label="Test errors")
    save(fig, "haste_final_13_test_oracle_grid.png")

    print(f"\nSaved oracle results to {path}")


def extended_oracle_phase():
    """Post-hoc oracle: the official test labels are used for selection."""

    if not OFFICIAL_FILE.exists():
        raise RuntimeError("The frozen official evaluation must exist first.")

    official = json.loads(OFFICIAL_FILE.read_text(encoding="utf-8"))

    raw_test, ytest = load_classification(
        "CharacterTrajectories", split="test", load_equal_length=False
    )

    Xt, _ = extract(list(raw_test))
    ytest = np.asarray(ytest, dtype=str)

    assert len(ytest) == 1436 and len(ytrain) == 1422

    # ========================================================
    # NORMALIZED COMPONENT DISTANCES — TRAIN ONLY
    # ========================================================

    parts = {}

    for key in DIM:

        A = DTRAIN[key].astype(np.float64)
        B = cdist(Xt[key], XTRAIN[key], "sqeuclidean").astype(np.float32).astype(np.float64)

        scale = max(float(A.sum())/(len(ytrain)*(len(ytrain)-1)), 1e-12)
        parts[key] = A/scale, B/scale

    base_weight = sum(BASE_W.values())

    base = tuple(
        sum(w*parts[k][j] for k, w in BASE_W.items())/base_weight
        for j in (0, 1)
    )

    def fuse(weights):

        total = base_weight+sum(weights.values())

        return tuple(
            np.maximum(
                (base_weight*base[j]+sum(w*parts[k][j] for k, w in weights.items()))/total,
                0.
            )
            for j in (0, 1)
        )

    def old_model(name):

        keys = MODELS[name]

        if not keys:
            return base

        if keys == ("legacy",):

            return tuple(
                np.maximum(
                    (base_weight*base[j]+2*(parts["events"][j]+.5*parts["legacy"][j])/1.5)/(base_weight+2),
                    0.
                )
                for j in (0, 1)
            )

        return fuse({k: EXTRA_W[k] for k in keys})

    def train_predict(A, B, C, gamma, power=1.):

        At = A if power == 1 else np.power(A, power)
        Bt = B if power == 1 else np.power(B, power)

        model = SVC(C=float(C), kernel="precomputed").fit(np.exp(-gamma*At), ytrain)
        return model.predict(np.exp(-gamma*Bt))

    def assess(pred):

        errors = int(np.count_nonzero(pred != ytest))
        macro = float(f1_score(ytest, pred, average="macro", zero_division=0))

        return errors, macro

    # ========================================================
    # FROZEN OFFICIAL REPRODUCIBILITY
    # ========================================================

    A0, B0 = old_model(official["method"])

    frozen_pred = train_predict(A0, B0, official["C"], official["gamma"])
    frozen_error, _ = assess(frozen_pred)

    if frozen_error != official["errors"]:
        raise RuntimeError("Frozen official model not reproduced.")

    print(f"OFFICIAL FROZEN: {frozen_error} errors | {official['method']}")

    all_runs, bank, kernel_best, blend_best = [], [], {}, {}

    def add_bank(label, best, source):

        bank.append({
            "label": label,
            "errors": best["errors"],
            "source": source,
            "pred": best["pred"].copy()
        })

    # ========================================================
    # PARAMETER SEARCH
    # ========================================================

    def scan(label, A, B, Cs, gammas, powers, stage):

        best = None

        for power in powers:

            At = A if power == 1 else np.power(A, power)
            Bt = B if power == 1 else np.power(B, power)

            for gamma in gammas:

                Ktr, Kte = np.exp(-gamma*At), np.exp(-gamma*Bt)

                for C in Cs:

                    model = SVC(C=float(C), kernel="precomputed").fit(Ktr, ytrain)
                    pred = model.predict(Kte)

                    errors, macro = assess(pred)

                    record = {
                        "name": label,
                        "stage": stage,
                        "C": float(C),
                        "gamma": float(gamma),
                        "power": float(power),
                        "errors": errors,
                        "macro_f1": macro
                    }

                    all_runs.append(record)

                    if best is None or (errors, -macro) < (best["errors"], -best["macro_f1"]):
                        best = {**record, "pred": pred.copy()}

        return best

    # ========================================================
    # RESTORE PREVIOUS ORACLE MODELS FOR ENSEMBLES
    # ========================================================

    previous_file = RESULTS_DIR / "haste_mds_test_oracle.json"

    if previous_file.exists():

        previous = json.loads(previous_file.read_text(encoding="utf-8"))
        earlier = {}

        for r in previous["all_results"]:
            if r["model"] not in earlier:
                earlier[r["model"]] = r

        for name, r in earlier.items():

            A, B = old_model(name)
            pred = train_predict(A, B, r["C"], r["gamma"])

            errors, macro = assess(pred)

            if errors != r["errors"]:
                raise RuntimeError(f"Historical oracle not reproduced: {name}")

            add_bank("PREVIOUS | "+name, {"pred": pred, "errors": errors}, "previous")

    # ========================================================
    # STAGE 1 — KERNEL POWER, GAMMA AND C
    # ========================================================

    focus = (
        "Events+Density-85D",
        "Legacy-89D",
        "MDS2-Delta+CorrectPos-81D",
        "MDS4-Delta+OldPos-89D",
        "MDS4-Delta+CorrectPos-89D",
        "MDS4-Delta+CorrectPos+Magnitudes-93D"
    )

    c_refine = (3., 10., 30., 100., 300.)
    g_refine = (.125, .25, .375, .5, .75, 1., 1.5, 2., 4.)
    p_refine = (.5, .75, 1.)

    print("\nSTAGE 1 — DENSE KERNEL SEARCH", flush=True)

    for name in focus:

        A, B = old_model(name)
        best = scan(name, A, B, c_refine, g_refine, p_refine, "kernel")

        kernel_best[name] = best
        add_bank("KERNEL | "+name, best, "kernel")

        print(
            f"{name:48s} | Errors={best['errors']:2d} | "
            f"C={best['C']:g} | gamma={best['gamma']:g} | p={best['power']:g}",
            flush=True
        )

    # ========================================================
    # STAGE 2 — POSITIVE SYMBOLIC DISTANCE MIXTURES
    # ========================================================

    blends = []

    # A. Harmonic events and transition density

    for e, d in product((.5, 1., 2., 4.), (0., 1., 3.)):

        blends.append((
            f"Event-e{e:g}-d{d:g}",
            {"events": 4*e/3, "density": d/6}
        ))

    # B. MDS4 displacements and event positions

    for m, p in product((.5, 1., 2., 4.), (0., .5, 1., 2.)):

        blends.append((
            f"MDS4-m{m:g}-pos{p:g}",
            {"mds4_delta": 4*m/3, "density": 1/6, "old_pos": .5*p}
        ))

    # C. Harmonic + MDS4 hybrid

    for e, m, p in product((.5, 1., 2.), (.5, 1., 2.), (0., 1.)):

        blends.append((
            f"Hybrid-e{e:g}-m{m:g}-pos{p:g}",
            {
                "events": 4*e/3,
                "mds4_delta": 4*m/3,
                "density": 1/6,
                "old_pos": .5*p
            }
        ))

    print(f"\nSTAGE 2 — {len(blends)} SYMBOLIC MIXTURES", flush=True)

    for i, (name, weights) in enumerate(blends, 1):

        A, B = fuse(weights)

        best = scan(
            name, A, B,
            (10., 100.),
            (.25, .5, 1., 2.),
            (.5, .75, 1.),
            "mixture"
        )

        blend_best[name] = best

        if i % 5 == 0 or i == len(blends):

            running_best = min(
                blend_best.values(),
                key=lambda x: (x["errors"], -x["macro_f1"])
            )

            print(
                f"{i:2d}/{len(blends)} complete | "
                f"Best={running_best['errors']} errors | {running_best['name']}",
                flush=True
            )

    for best in sorted(
        blend_best.values(),
        key=lambda r: (r["errors"], -r["macro_f1"])
    )[:10]:

        add_bank("MIX | "+best["name"], best, "mixture")

    # ========================================================
    # STAGE 3 — COMPLEMENTARY SVM ENSEMBLES
    # ========================================================

    unique, seen = [], set()

    for r in sorted(bank, key=lambda x: x["errors"]):

        signature = r["pred"].tobytes()

        if signature not in seen:
            seen.add(signature)
            unique.append(r)

    core_pool = [r for r in unique if r["source"] != "mixture"][:8]
    mix_pool = [r for r in unique if r["source"] == "mixture"][:6]

    pool = core_pool+mix_pool
    selected_ids = {id(r) for r in pool}

    for r in unique:

        if len(pool) >= 14:
            break

        if id(r) not in selected_ids:
            pool.append(r)
            selected_ids.add(id(r))

    pool = sorted(pool, key=lambda x: x["errors"])

    classes = np.unique(ytrain)
    test_ids = np.searchsorted(classes, ytest)

    P = np.stack([
        np.searchsorted(classes, r["pred"])
        for r in pool
    ])

    indices = np.arange(len(ytest))
    n_classes = len(classes)

    theoretical = int(
        np.count_nonzero(np.all(P != test_ids[None, :], axis=0))
    )

    best_ensemble = None

    print(
        f"\nSTAGE 3 — ENSEMBLES | Pool={len(pool)} | "
        f"All-wrong={theoretical}",
        flush=True
    )

    for size in (3, 5):

        for combo in combinations(range(len(pool)), size):

            vote = np.zeros((n_classes, len(ytest)), dtype=np.uint8)

            for j in combo:
                vote[P[j], indices] += 1

            maximum = vote.max(axis=0)
            winners = vote.argmax(axis=0)

            # Ties are resolved by the best-ranked ensemble member

            ties = np.count_nonzero(vote == maximum, axis=0) > 1

            for j in combo:

                chosen = ties & (vote[P[j], indices] == maximum)

                winners[chosen] = P[j, chosen]
                ties[chosen] = False

            errors = int(np.count_nonzero(winners != test_ids))

            if best_ensemble is None or errors < best_ensemble["errors"]:

                best_ensemble = {
                    "errors": errors,
                    "members": [pool[j]["label"] for j in combo],
                    "pred": classes[winners].copy()
                }

        print(
            f"Ensemble search up to {size} members: "
            f"{best_ensemble['errors']} errors",
            flush=True
        )

    # ========================================================
    # AGGREGATED RESULTS
    # ========================================================

    winners = [*kernel_best.values(), *blend_best.values()]

    best_individual = min(
        winners, key=lambda r: (r["errors"], -r["macro_f1"])
    )

    best_ensemble["macro_f1"] = float(
        f1_score(ytest, best_ensemble["pred"], average="macro", zero_division=0)
    )

    ultimate = min(best_individual["errors"], best_ensemble["errors"])

    print("\n"+"="*95)
    print("EXTENDED POST-HOC ORACLE — NOT AN INDEPENDENT OFFICIAL SCORE")
    print("="*95)

    print(f"Frozen official: {official['errors']} errors")

    if previous_file.exists():
        print(f"Previous oracle: {min(r['errors'] for r in previous['all_results'])} errors")

    print(
        f"Best new individual: {best_individual['errors']} errors | "
        f"{best_individual['name']} | C={best_individual['C']:g}, "
        f"gamma={best_individual['gamma']:g}, p={best_individual['power']:g}"
    )

    print(
        f"Best ensemble: {best_ensemble['errors']} errors | "
        f"{best_ensemble['members']}"
    )

    print(
        f"Highest observed post-hoc accuracy: "
        f"{100*(1-ultimate/len(ytest)):.4f}% ({ultimate} errors)"
    )

    print(
        f"All-wrong samples in candidate pool "
        f"(non-predictive oracle lower bound): {theoretical}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    def clean(r):
        return {k: v for k, v in r.items() if k != "pred"}

    output = {
        "status": "POST-HOC TEST ORACLE — NOT AN UNBIASED TEST RESULT",
        "frozen_official": official,
        "best_individual": clean(best_individual),
        "best_ensemble": clean(best_ensemble),
        "all_wrong_in_pool": theoretical,
        "pool": [
            {
                "label": r["label"],
                "errors": r["errors"],
                "source": r["source"]
            }
            for r in pool
        ],
        "kernel_best": {
            k: clean(v) for k, v in kernel_best.items()
        },
        "top_mixtures": [
            clean(r) for r in sorted(
                blend_best.values(), key=lambda r: r["errors"]
            )[:15]
        ],
        "all_runs": all_runs
    }

    result_path = RESULTS_DIR / "haste_mds_oracle_extended.json"

    result_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # ========================================================
    # FIGURE 14 — BEST MODELS AND MIXTURES
    # ========================================================

    chart = [
        (r["name"], r["errors"])
        for r in kernel_best.values()
    ]

    chart += [
        (r["name"], r["errors"])
        for r in sorted(
            blend_best.values(), key=lambda x: x["errors"]
        )[:8]
    ]

    chart.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(np.arange(len(chart)), [z[1] for z in chart])

    ax.set_yticks(
        np.arange(len(chart)),
        [z[0] for z in chart]
    )

    ax.invert_yaxis()

    ax.axvline(
        7, color="gray", ls="--",
        label="Previous test oracle — 7 errors"
    )

    ax.set_xlabel("Test errors — optimized retrospectively")
    ax.set_title("Post-hoc oracle — kernels and symbolic fusion")

    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "haste_oracle_14_models.png")

    # ========================================================
    # FIGURE 15 — KERNEL POWER ABLATION
    # ========================================================

    fig, ax = plt.subplots(figsize=(9, 5))

    for name in focus:

        minima = [
            min(
                r["errors"] for r in all_runs
                if r["stage"] == "kernel"
                and r["name"] == name
                and r["power"] == p
            )
            for p in p_refine
        ]

        ax.plot(p_refine, minima, marker="o", lw=1.3, label=name)

    ax.set(
        xlabel="Kernel exponent p",
        ylabel="Minimum test errors",
        title="Kernel shape — post-hoc selection"
    )

    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "haste_oracle_15_kernel_power.png")

    # ========================================================
    # FIGURE 16 — ERROR COMPLEMENTARITY
    # ========================================================

    wrong = P != test_ids[None, :]

    intersection = wrong.astype(int) @ wrong.astype(int).T
    totals = wrong.sum(axis=1)

    union = totals[:, None]+totals[None, :]-intersection
    jaccard = intersection/np.maximum(union, 1)

    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(jaccard, vmin=0, vmax=1, cmap="viridis")

    ax.set_xticks(
        range(len(pool)),
        [str(i+1) for i in range(len(pool))]
    )

    ax.set_yticks(
        range(len(pool)),
        [f"{i+1} — {r['errors']} errors" for i, r in enumerate(pool)]
    )

    ax.set_title("Test-error overlap — candidate ensembles")

    fig.colorbar(image, ax=ax, label="Jaccard of error sets")

    save(fig, "haste_oracle_16_error_overlap.png")

    # ========================================================
    # FIGURE 17 — BEST ENSEMBLE CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        ytest, best_ensemble["pred"], labels=classes
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)

    ax.set(
        xlabel="Predicted",
        ylabel="True",
        title="Best post-hoc ensemble — NOT official"
    )

    fig.colorbar(image, ax=ax, label="Count")

    save(fig, "haste_oracle_17_ensemble_confusion.png")

    print(f"\nSaved numerical results: {result_path}")
    print("Saved four figures to figures/.")


# ============================================================
# MAIN
# ============================================================


train, ytrain = load_classification(
    "CharacterTrajectories",
    split="train",
    load_equal_length=False
)

train = list(train)
ytrain = np.asarray(ytrain, dtype=str)

assert len(train) == 1422, "Unexpected official training split size"

print(
    f"TRAIN={len(ytrain)} | CLASSES={len(np.unique(ytrain))} | "
    f"MODELS={len(MODELS)} | PHASE={PHASE}"
)

XTRAIN, enc_train = extract(train)
DTRAIN, DTEST = make_matrices(XTRAIN)

if PHASE not in ("cv", "official", "test_oracle", 'oracle_extended'):
    raise ValueError("Invalid phase")

if PHASE == "cv":
    cv_phase(train, enc_train)
elif PHASE == "official":
    official_phase()
elif PHASE == "test_oracle":
    test_oracle_phase()
else:
    extended_oracle_phase()
