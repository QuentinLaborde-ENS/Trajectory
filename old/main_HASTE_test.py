# -*- coding: utf-8 -*-


import json
from dataclasses import dataclass
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import dct
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC


# ============================================================
# HASTE-AL -- SYMBOLIC A/B ABLATION
# Character Trajectories | Training-only nested CV
# ============================================================

SEED, N, K = 2026, 241, 24
OUTER_SPLITS, REPEATS, INNER_SPLITS = 5, 5, 3
CS = (10., 100.)
GAMMAS = (0.125, 0.25, 0.5, 1., 2.)

SAVE_DIR, FIG_DIR = Path("results"), Path("figures")
SAVE_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Variant:
    name: str
    angular_mode: str = "base"
    time_speed: bool = False
    arc_speed: bool = False


MODELS = [
    Variant("S4-reimplementation-73D"),
    Variant("A-SpeedTime-89D", time_speed=True),
    Variant("A-SpeedArc-89D", arc_speed=True),
    Variant("A-TimeArc-105D", time_speed=True, arc_speed=True),
    Variant("B-Hysteresis-73D", "hys"),
    Variant("B-MinRun3-73D", "run3"),
    Variant("B-MinRun5-73D", "run5"),
    Variant("B-Hysteresis+Run3-73D", "hysrun3"),
    Variant("AB-TimeArc+Regularized-105D", "hysrun3", True, True),
]

MODES = ["base", "hys", "run3", "run5", "hysrun3"]
EXTRA = "InnerSelected-AB"


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_data():

    from aeon.datasets import load_classification

    X, y = load_classification(
        "CharacterTrajectories",
        split="train",
        load_equal_length=False
    )

    X = list(X)
    y = np.asarray(y).astype(str)

    assert len(X) == len(y) == 1422
    assert len(np.unique(y)) == 20

    return X, y


# ============================================================
# 2. PREPROCESSING
# ============================================================

def clean_sample(sample):

    a = np.asarray(sample, dtype=float)

    if a.ndim != 2:
        raise ValueError(f"Series must be 2D, got {a.shape}")

    if a.shape[0] <= 8 and a.shape[1] > a.shape[0]:
        a = a[:2].T
    else:
        a = a[:, :2]

    if a.shape[0] < 3:
        raise ValueError("Too few temporal samples")

    for k in range(2):

        good = np.isfinite(a[:, k])
        t = np.arange(len(a))

        a[:, k] = (
            np.interp(t, t[good], a[good, k])
            if good.any() else 0.
        )

    return gaussian_filter1d(
        a, sigma=0.7, axis=0, mode="nearest"
    )


def interpolate_circle(u, angle, grid):

    z = np.exp(1j * angle)

    zr = (
        np.interp(grid, u, z.real)
        + 1j * np.interp(grid, u, z.imag)
    )

    return np.angle(zr + 1e-12)


def raw_grids(sample):

    v = clean_sample(sample)
    sp = np.linalg.norm(v, axis=1)

    t = np.linspace(0., 1., len(v))

    arc = np.r_[
        0.,
        np.cumsum((sp[1:] + sp[:-1]) / 2)
    ]

    arc = arc / max(arc[-1], 1e-12)

    keep = np.r_[True, np.diff(arc) > 1e-12]

    if keep.sum() < 2:
        arc = t
        keep = np.ones(len(t), bool)

    grid = np.linspace(0., 1., N)
    theta = np.arctan2(v[:, 1], v[:, 0])

    theta_t = interpolate_circle(t, theta, grid)
    theta_s = interpolate_circle(arc[keep], theta[keep], grid)

    sp_t = np.interp(grid, t, sp)
    sp_s = np.interp(grid, arc[keep], sp[keep])

    positive = sp[sp > 1e-10]

    scale = max(
        float(np.median(positive)) if len(positive) else 0.,
        1e-10
    )

    return theta_t, theta_s, sp_t / scale, sp_s / scale


# ============================================================
# 3. ANGULAR SYMBOLIZATION
# ============================================================

def quantize_angle(theta):

    return np.mod(
        np.rint(
            np.mod(theta, 2*np.pi) / (2*np.pi/K)
        ).astype(np.int16),
        K
    )


def hysteresis(theta, margin=0.20):

    q = quantize_angle(theta)
    out = q.copy()

    step = 2*np.pi/K

    for i in range(1, len(q)):

        candidate = int(q[i])
        previous = int(out[i-1])

        e_new = abs(np.angle(
            np.exp(1j * (theta[i] - candidate*step))
        ))

        e_old = abs(np.angle(
            np.exp(1j * (theta[i] - previous*step))
        ))

        out[i] = (
            candidate
            if e_old - e_new > margin*step
            else previous
        )

    return out


def short_run_filter(q, min_run=3, circular=True, alphabet=K):

    q = np.asarray(q, dtype=np.int16).copy()

    for _ in range(len(q)):

        cuts = np.r_[
            0,
            np.flatnonzero(np.diff(q) != 0) + 1,
            len(q)
        ]

        lengths = np.diff(cuts)
        small = np.flatnonzero(lengths < min_run)

        if len(small) == 0 or len(cuts) <= 2:
            break

        k = int(small[np.argmin(lengths[small])])

        start, end = cuts[k:k+2]

        left = k-1 if k > 0 else None
        right = k+1 if k < len(cuts)-2 else None

        if left is None and right is None:
            break

        if left is None:
            target = int(q[cuts[right]])

        elif right is None:
            target = int(q[cuts[left]])

        else:

            def distance(a, b):
                d = abs(int(a)-int(b))
                return min(d, alphabet-d) if circular else d

            dl = distance(q[start], q[cuts[left]])
            dr = distance(q[start], q[cuts[right]])

            target = int(q[cuts[left if dl <= dr else right]])

        q[start:end] = target

    return q


def angular_states(theta, mode):

    q = (
        hysteresis(theta)
        if mode in ("hys", "hysrun3")
        else quantize_angle(theta)
    )

    n = {
        "run3": 3,
        "run5": 5,
        "hysrun3": 3
    }.get(mode, 1)

    return short_run_filter(q, n) if n > 1 else q


# ============================================================
# 4. SYMBOLIC FEATURES
# ============================================================

def coefficients(signal, n):

    return dct(
        np.asarray(signal, float),
        type=2,
        norm="ortho",
        axis=0
    )[:n].T.ravel()


def angular_blocks(qt, qs):

    # S1: orientation states -- 32D

    angt = 2*np.pi*qt/K
    angs = 2*np.pi*qs/K

    st = np.column_stack((np.cos(angt), np.sin(angt)))
    ss = np.column_stack((np.cos(angs), np.sin(angs)))

    states = np.r_[
        coefficients(st, 8),
        coefficients(ss, 8)
    ]

    # S2: circular jumps + density -- 16D

    dj = (
        (np.diff(qs.astype(float), prepend=float(qs[0])) + K/2)
        % K - K/2
    ) / (K/2)

    density = gaussian_filter1d(np.abs(dj), sigma=3)

    jumps = np.r_[
        coefficients(dj, 8),
        coefficients(density, 8)
    ]

    # S3: second-order symbolic variations -- 16D

    turning = gaussian_filter1d(
        np.abs(np.diff(dj, prepend=dj[0])),
        sigma=3
    )

    bending = gaussian_filter1d(
        np.abs(np.diff(dj, n=2, prepend=[dj[0], dj[0]])),
        sigma=3
    )

    second = np.r_[
        coefficients(turning, 8),
        coefficients(bending, 8)
    ]

    # S4: transition positions, run lengths and density -- 9D

    cuts = np.flatnonzero(np.diff(qs) != 0) + 1
    pos = cuts/N

    if len(pos) == 0:
        position = np.zeros(32)

    else:

        rank = (np.arange(len(pos)) + 0.5)/len(pos)
        dev = pos-rank

        position = np.interp(
            np.linspace(0, 1, 32),
            rank, dev,
            left=dev[0],
            right=dev[-1]
        )

    lengths = np.diff(np.r_[0, cuts, N])/N
    ranks = (np.arange(len(lengths)) + 0.5)/len(lengths)

    run_profile = np.interp(
        np.linspace(0, 1, 32),
        ranks, lengths,
        left=lengths[0],
        right=lengths[-1]
    )

    meta = np.r_[
        coefficients(position, 4),
        coefficients(run_profile, 4),
        len(cuts)/N
    ]

    assert (len(states), len(jumps), len(second), len(meta)) == (
        32, 16, 16, 9
    )

    return (states, jumps, second, meta), len(cuts)


# ============================================================
# 5. A -- MULTICHANNEL SPEED SYMBOLIZATION
# ============================================================

def speed_features(relative_speed, regularize=False):

    s = np.digitize(
        relative_speed,
        [0.5, 1., 2.]
    ).astype(np.int16)

    if regularize:
        s = short_run_filter(
            s, min_run=3,
            circular=False, alphabet=4
        )

    onehot = np.eye(4, dtype=float)[s]

    # Four states x four DCT coefficients = 16D
    return coefficients(onehot, 4)


def extract_bank(X):

    keys = ("states", "jumps", "second", "meta")

    bag = {
        f"{mode}_{key}": []
        for mode in MODES for key in keys
    }

    bag.update({
        "speed_t": [],
        "speed_s": [],
        "speed_t_reg": [],
        "speed_s_reg": []
    })

    event_counts = {mode: [] for mode in MODES}

    for i, x in enumerate(X):

        tt, ts, vt, vs = raw_grids(x)

        for mode in MODES:

            qt = angular_states(tt, mode)
            qs = angular_states(ts, mode)

            (st, ju, so, me), ne = angular_blocks(qt, qs)

            for key, value in zip(keys, (st, ju, so, me)):
                bag[f"{mode}_{key}"].append(value)

            event_counts[mode].append(int(ne))

        for key, v, reg in [
            ("speed_t", vt, False),
            ("speed_s", vs, False),
            ("speed_t_reg", vt, True),
            ("speed_s_reg", vs, True)
        ]:
            bag[key].append(speed_features(v, reg))

        if (i+1) % 250 == 0:
            print(f"Features extracted {i+1}/{len(X)}", flush=True)

    bank = {
        k: np.asarray(v, np.float32)
        for k, v in bag.items()
    }

    return bank, event_counts


# ============================================================
# 6. COMPOSITE SYMBOLIC DISTANCES
# ============================================================

def feature_keys(v):

    keys = [
        (f"{v.angular_mode}_states", 1.0),
        (f"{v.angular_mode}_jumps", 0.5),
        (f"{v.angular_mode}_second", 0.25),
        (f"{v.angular_mode}_meta", 0.5)
    ]

    regular = v.angular_mode != "base"

    if v.time_speed:
        keys.append((
            "speed_t_reg" if regular else "speed_t",
            0.4
        ))

    if v.arc_speed:
        keys.append((
            "speed_s_reg" if regular else "speed_s",
            0.4
        ))

    return keys


def precompute_q(bank):

    Q = {}

    for k, F in bank.items():

        z = np.einsum("ij,ij->i", F, F)

        q = (
            z[:, None]
            + z[None, :]
            - 2*(F @ F.T)
        )

        Q[k] = np.maximum(q, 0).astype(np.float32)
        np.fill_diagonal(Q[k], 0.)

        print(f"Pairwise distances: {k}: {F.shape[1]}D", flush=True)

    return Q


def composed_distance(Q, model, train, query=None):

    train = np.asarray(train, int)
    query = train if query is None else np.asarray(query, int)

    total = np.zeros((len(query), len(train)), np.float32)
    wsum = 0.

    for key, w in feature_keys(model):

        q = Q[key]

        # Normalization uses training observations only.
        scale = max(
            float(q[np.ix_(train, train)].mean()),
            1e-10
        )

        total += w * (q[np.ix_(query, train)] / scale)
        wsum += w

    return np.maximum(total / wsum, 0.)


# ============================================================
# 7. INNER MODEL SELECTION
# ============================================================

def optimize(Q, model, y, outer_train, inner_splits):

    scores = {
        (c, g): [0., 0.]
        for c in CS for g in GAMMAS
    }

    for inner_tr, inner_va in inner_splits:

        tr = outer_train[inner_tr]
        va = outer_train[inner_va]

        Dtr = composed_distance(Q, model, tr)
        Dva = composed_distance(Q, model, tr, va)

        for gamma in GAMMAS:

            Ktr = np.exp(-gamma*Dtr)
            Kva = np.exp(-gamma*Dva)

            for C in CS:

                clf = SVC(
                    C=C,
                    kernel="precomputed",
                    decision_function_shape="ovo"
                ).fit(Ktr, y[tr])

                yp = clf.predict(Kva)

                scores[(C, gamma)][0] += accuracy_score(y[va], yp)
                scores[(C, gamma)][1] += f1_score(
                    y[va], yp,
                    average="macro",
                    zero_division=0
                )

    ranking = sorted(
        scores,
        key=lambda t: (
            scores[t][0],
            scores[t][1],
            -t[0],
            -t[1]
        ),
        reverse=True
    )

    C, gamma = ranking[0]

    return (
        C,
        gamma,
        scores[(C, gamma)][0] / len(inner_splits),
        scores[(C, gamma)][1] / len(inner_splits)
    )


def fit_outer(Q, model, tr, va, y, C, gamma):

    Dtr = composed_distance(Q, model, tr)
    Dva = composed_distance(Q, model, tr, va)

    clf = SVC(
        C=C,
        kernel="precomputed"
    ).fit(np.exp(-gamma*Dtr), y[tr])

    return clf.predict(np.exp(-gamma*Dva))


# ============================================================
# 8. TRAIN-ONLY MDS
# ============================================================

def mds_train_only(Dtr, Dva):

    n = len(Dtr)

    row = Dtr.mean(axis=1)
    grand = row.mean()

    gram = -0.5 * (
        Dtr
        - row[:, None]
        - row[None, :]
        + grand
    )

    eigval, eigvec = eigh(
        gram,
        subset_by_index=[n-2, n-1]
    )

    eigval = eigval[::-1].copy()
    eigvec = eigvec[:, ::-1].copy()

    lam = np.maximum(eigval, 1e-12)

    Ztr = eigvec * np.sqrt(lam)

    cross = -0.5 * (
        Dva
        - row[None, :]
        - Dva.mean(axis=1, keepdims=True)
        + grand
    )

    Zva = (cross @ eigvec) / np.sqrt(lam)

    return Ztr, Zva


# ============================================================
# 9. FIGURE UTILITIES
# ============================================================

def style(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)


def savefig(name, fig):

    fig.savefig(
        FIG_DIR / name,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_summaries(y, names, pred, repeats, fold_errors,
                   chosen, Q, models, X, event_counts, first_split):

    E = np.array([
        [(pred[n][r] != y).sum() for r in range(repeats)]
        for n in names
    ])

    # --------------------------------------------------------
    # Figure 1: accuracy ablation and repeat stability
    # --------------------------------------------------------

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    means = E.mean(axis=1)
    sd = E.std(axis=1, ddof=1)

    ax[0].barh(np.arange(len(names)), means, xerr=sd, capsize=3)
    ax[0].set_yticks(range(len(names)), names)
    ax[0].invert_yaxis()

    ax[0].axvline(
        21.6, color="gray", ls="--",
        label="Historical S4: 21.6 (different code)"
    )

    ax[0].axvline(
        5., color="black", ls=":",
        label="Historical MDS4: 5.0 (different code)"
    )

    ax[0].set_xlabel("Errors per repeat / 1422")
    ax[0].legend(fontsize=8)
    style(ax[0])

    for j, n in enumerate(names):
        ax[1].plot(
            np.arange(1, repeats+1),
            E[j],
            ".-",
            lw=1,
            label=n
        )

    ax[1].set_xticks(range(1, repeats+1))
    ax[1].set_xlabel("Outer CV repeat")
    ax[1].set_ylabel("Errors")
    ax[1].legend(fontsize=7, ncol=2)
    style(ax[1])

    fig.suptitle("A/B repeated nested CV -- mean and stability")
    savefig("haste_AB_01_ablation.png", fig)

    # --------------------------------------------------------
    # Figure 2: paired error overlap
    # --------------------------------------------------------

    wrong = np.array([
        (pred[n] != y[None, :]).ravel()
        for n in names
    ])

    J = np.eye(len(names))

    for i in range(len(names)):

        for j in range(i+1, len(names)):

            num = np.logical_and(wrong[i], wrong[j]).sum()
            den = np.logical_or(wrong[i], wrong[j]).sum()

            J[i, j] = J[j, i] = num/max(den, 1)

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")

    ax.set_xticks(
        range(len(names)), names,
        rotation=65, ha="right", fontsize=8
    )

    ax.set_yticks(range(len(names)), names, fontsize=8)

    fig.colorbar(im, ax=ax, label="Error Jaccard")
    ax.set_title("A/B -- paired error overlap (25 outer folds)")

    savefig("haste_AB_02_error_overlap.png", fig)

    # --------------------------------------------------------
    # Figure 3: confusion matrices
    # --------------------------------------------------------

    order = sorted(
        np.unique(y),
        key=lambda v: int(v) if v.isdigit() else v
    )

    shown = [names[0], names[-2], names[-1]]

    fig, ax = plt.subplots(1, len(shown), figsize=(19, 6))

    for a, n in zip(ax, shown):

        cm = confusion_matrix(
            np.tile(y, repeats),
            pred[n].ravel(),
            labels=order
        )

        a.imshow(cm, cmap="Blues")

        a.set_xticks(
            range(len(order)), order,
            rotation=90, fontsize=7
        )

        a.set_yticks(range(len(order)), order, fontsize=7)
        a.set_title(n, fontsize=10)

        a.set_xlabel("Predicted")
        a.set_ylabel("True")

    fig.suptitle(
        "Train-only out-of-fold confusion matrices (5 repeats)"
    )

    savefig("haste_AB_03_confusions.png", fig)

    # --------------------------------------------------------
    # Figure 4: errors per class
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 5))

    pos = np.arange(len(order))
    width = 0.25

    for k, n in enumerate(shown):

        count = [
            (
                (pred[n] != y[None, :])
                & (y[None, :] == c)
            ).sum()
            for c in order
        ]

        ax.bar(
            pos+(k-1)*width,
            count,
            width,
            label=n
        )

    ax.set_xticks(pos, order)
    ax.set_xlabel("True class")
    ax.set_ylabel("Repeated held-out errors")
    ax.legend(fontsize=8)

    style(ax)
    savefig("haste_AB_04_errors_by_class.png", fig)

    # --------------------------------------------------------
    # Figure 5: train-only MDS
    # --------------------------------------------------------

    tr, va = first_split
    focus = ["7", "9", "10", "18"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for a, model in zip(axes, [models[0], models[-1]]):

        Dtr = composed_distance(Q, model, tr)
        Dva = composed_distance(Q, model, tr, va)

        ztr, zva = mds_train_only(Dtr, Dva)

        a.scatter(
            ztr[:, 0], ztr[:, 1],
            s=8, c="lightgray", alpha=0.32
        )

        for c in focus:

            ix = y[tr] == c

            a.scatter(
                ztr[ix, 0], ztr[ix, 1],
                s=13, alpha=0.8,
                label=f"Class {c}"
            )

        wrong_va = pred[model.name][0, va] != y[va]

        a.scatter(
            zva[wrong_va, 0],
            zva[wrong_va, 1],
            marker="x",
            s=85,
            c="black",
            label="Held-out errors"
        )

        a.set_title(model.name)
        a.set_xlabel("MDS 1")
        a.set_ylabel("MDS 2")
        style(a)

    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Training-fold MDS with out-of-sample validation points"
    )

    savefig("haste_AB_05_train_only_mds.png", fig)

    # --------------------------------------------------------
    # Figure 6: hard cases and neighbors
    # --------------------------------------------------------

    target = names[-2]

    errs = (
        pred[target] != y[None, :]
    ).sum(axis=0)

    # Only targets held out in the first outer fold.
    bad = va[
        np.argsort(-errs[va], kind="stable")[:5]
    ]

    fig, ax = plt.subplots(
        len(bad), 3,
        figsize=(12, 3*len(bad)),
        squeeze=False
    )

    model = models[-1]
    D = composed_distance(Q, model, tr, bad)

    for r, idx in enumerate(bad):

        same = np.flatnonzero(y[tr] == y[idx])
        diff = np.flatnonzero(y[tr] != y[idx])

        si = tr[same[np.argmin(D[r, same])]]
        di = tr[diff[np.argmin(D[r, diff])]]

        titles = [
            f"Held-out #{idx} | class {y[idx]} | errors {errs[idx]}/5",
            f"Train same #{si}",
            f"Train rival #{di} | class {y[di]}"
        ]

        for c, ix in enumerate([idx, si, di]):

            v = clean_sample(X[ix])

            xy = np.cumsum(v, axis=0)
            xy -= xy[0]

            a = ax[r, c]

            a.plot(xy[:, 0], xy[:, 1], lw=1.15)
            a.scatter(*xy[0], s=20)
            a.scatter(*xy[-1], s=28, marker="x")

            a.set_aspect("equal", adjustable="datalim")
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(titles[c], fontsize=9)

            style(a)

    fig.suptitle("A+B persistent errors and training-only neighbors")
    savefig("haste_AB_06_hard_cases.png", fig)

    # --------------------------------------------------------
    # Figure 7: symbolic profiles
    # --------------------------------------------------------

    fig, ax = plt.subplots(3, 3, figsize=(14, 9), squeeze=False)

    for r, idx in enumerate(bad[:3]):

        tt, ts, vt, vs = raw_grids(X[idx])

        base = angular_states(ts, "base")
        reg = angular_states(ts, "hysrun3")
        speed = np.digitize(vs, [0.5, 1., 2.])

        x = np.linspace(0, 1, N)

        ax[r, 0].step(
            x, base,
            where="mid", lw=1,
            label="Original"
        )

        ax[r, 0].step(
            x, reg,
            where="mid", lw=1,
            label="Regularized", alpha=0.75
        )

        ax[r, 1].step(x, speed, where="mid")

        ax[r, 2].plot(
            x,
            np.r_[0, np.diff(base) != 0].cumsum(),
            label="Original"
        )

        ax[r, 2].plot(
            x,
            np.r_[0, np.diff(reg) != 0].cumsum(),
            label="Regularized"
        )

        ax[r, 0].set_ylabel(f"#{idx} | class {y[idx]}")

        for a in ax[r]:
            style(a)
            a.set_xlabel("Normalized arc length")

    for a, title in zip(
        ax[0],
        [
            "Orientation symbols",
            "Speed symbols (4 states)",
            "Cumulative angular events"
        ]
    ):
        a.set_title(title)

    ax[0, 0].legend(fontsize=8)
    ax[0, 2].legend(fontsize=8)

    fig.suptitle(
        "Symbolization A and event regularization B"
    )

    savefig("haste_AB_07_symbolic_profiles.png", fig)

    # --------------------------------------------------------
    # Figure 8: event-count regularization
    # --------------------------------------------------------

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    bins = np.arange(0, 101, 2)

    for mode in MODES:

        ax[0].hist(
            event_counts[mode],
            bins=bins,
            density=True,
            histtype="step",
            label=mode
        )

    ax[0].set_xlabel("Events / trajectory")
    ax[0].set_ylabel("Density")
    ax[0].legend(fontsize=8)

    ax[1].scatter(
        event_counts["base"],
        event_counts["hysrun3"],
        s=8,
        alpha=0.3
    )

    ax[1].set_xlabel("Original event count")
    ax[1].set_ylabel("Regularized event count")

    for a in ax:
        style(a)

    fig.suptitle(
        "Effect of run filtering and hysteresis (training only)"
    )

    savefig("haste_AB_08_event_counts.png", fig)

    # --------------------------------------------------------
    # Figure 9: within/between-class distances
    # --------------------------------------------------------

    Dtr = composed_distance(Q, models[-1], tr)
    yy = y[tr]

    pairs = [
        ("7", "18"),
        ("9", "10"),
        ("10", "18"),
        ("7", "10")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    rng = np.random.default_rng(SEED)

    for a, (c1, c2) in zip(axes.flat, pairs):

        ia = np.flatnonzero(yy == c1)
        ib = np.flatnonzero(yy == c2)

        within_a = Dtr[np.ix_(ia, ia)][
            np.triu_indices(len(ia), k=1)
        ]

        within_b = Dtr[np.ix_(ib, ib)][
            np.triu_indices(len(ib), k=1)
        ]

        between = Dtr[np.ix_(ia, ib)].ravel()

        for vals, lab in [
            (within_a, f"{c1}-{c1}"),
            (within_b, f"{c2}-{c2}"),
            (between, f"{c1}-{c2}")
        ]:

            vals = rng.choice(
                vals,
                min(len(vals), 10000),
                replace=False
            )

            a.hist(
                vals,
                bins=35,
                density=True,
                histtype="step",
                label=lab,
                lw=1.2
            )

        a.set_title(f"Classes {c1} vs {c2}")
        a.set_xlabel("Normalized squared distance")
        a.set_ylabel("Density")
        a.legend(fontsize=8)
        style(a)

    fig.suptitle(
        "A+B -- training-fold within/between class distances"
    )

    savefig("haste_AB_09_class_geometry.png", fig)

    return E


# ============================================================
# 10. REPEATED NESTED CROSS-VALIDATION
# ============================================================

def main():

    X, y = load_data()

    print(
        f"TRAIN={len(X)} | CLASSES={len(np.unique(y))} | "
        f"OUTER={REPEATS}x{OUTER_SPLITS} | INNER={INNER_SPLITS}"
    )

    print("Extracting symbolic and geometric features...")

    bank, counts = extract_bank(X)
    Q = precompute_q(bank)

    all_names = [m.name for m in MODELS] + [EXTRA]

    pred = {
        n: np.empty((REPEATS, len(y)), dtype=object)
        for n in all_names
    }

    fold_errors = {
        n: np.zeros((REPEATS, OUTER_SPLITS), int)
        for n in all_names
    }

    selection = []
    first_split = None
    params = Counter()

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=REPEATS,
        random_state=SEED
    )

    print("\n" + "="*90)
    print("REPEATED NESTED TRAIN-ONLY CROSS-VALIDATION")
    print("="*90)

    for it, (tr, va) in enumerate(
        outer.split(np.zeros(len(y)), y)
    ):

        r, f = divmod(it, OUTER_SPLITS)

        if first_split is None:
            first_split = (tr, va)

        inner = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=SEED+it
        )

        splits = list(
            inner.split(np.zeros(len(tr)), y[tr])
        )

        tuned = []

        for model in MODELS:

            C, gamma, inner_acc, inner_f1 = optimize(
                Q, model, y, tr, splits
            )

            p = fit_outer(
                Q, model, tr, va,
                y, C, gamma
            )

            pred[model.name][r, va] = p

            errors = int(np.sum(p != y[va]))
            fold_errors[model.name][r, f] = errors

            tuned.append((
                inner_acc,
                inner_f1,
                model,
                C,
                gamma
            ))

            params[(model.name, C, gamma)] += 1

        # Model selection uses ONLY inner-CV results.
        _, _, best_model, C, gamma = max(
            enumerate(tuned),
            key=lambda z: (
                z[1][0],
                z[1][1],
                -z[0]
            )
        )[1]

        pred[EXTRA][r, va] = pred[best_model.name][r, va]

        err = int(
            np.sum(pred[EXTRA][r, va] != y[va])
        )

        fold_errors[EXTRA][r, f] = err

        selection.append({
            "repeat": r+1,
            "fold": f+1,
            "variant": best_model.name,
            "C": C,
            "gamma": gamma,
            "errors": err
        })

        print(
            f"Repeat {r+1}/{REPEATS} | Fold {f+1}/{OUTER_SPLITS} | "
            + " | ".join(
                f"{n[:12]}={fold_errors[n][r, f]}"
                for n in all_names
            ),
            flush=True
        )

    # --------------------------------------------------------
    # Aggregate train-only results
    # --------------------------------------------------------

    print("\n" + "="*90)
    print("AGGREGATED TRAIN-ONLY RESULTS")
    print("="*90)

    for n in all_names:

        er = np.sum(fold_errors[n], axis=1)

        acc = np.mean([
            accuracy_score(y, pred[n][r])
            for r in range(REPEATS)
        ])

        f1 = np.mean([
            f1_score(
                y, pred[n][r],
                average="macro"
            )
            for r in range(REPEATS)
        ])

        print(
            f"{n:38s} | "
            f"Errors={er.mean():.2f} ± {er.std(ddof=1):.2f} | "
            f"Acc={100*acc:.4f}% | "
            f"F1={f1:.5f} | "
            f"Repeats={er.tolist()}"
        )

    print("\nINTERNAL MODEL SELECTION FREQUENCIES")

    for name, count in Counter(
        x["variant"] for x in selection
    ).most_common():

        print(f"{name:38s} | {count}/25 folds")

    # --------------------------------------------------------
    # Diagnostic figures
    # --------------------------------------------------------

    print("\nGenerating diagnostic figures...")

    E = plot_summaries(
        y, all_names, pred, REPEATS,
        fold_errors, selection,
        Q, MODELS, X, counts, first_split
    )

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    result = {
        "protocol": "Train-only repeated nested 5x5, inner=3",
        "warning": (
            "S4 is an independently specified reimplementation, "
            "not guaranteed identical to historical S4."
        ),
        "hyperparameters": {
            "C": CS,
            "gamma": GAMMAS,
            "n": N,
            "k": K,
            "seed": SEED
        },
        "models": {
            n: {
                "errors_by_repeat": E[i].tolist(),
                "accuracy": float(
                    1-E[i].mean()/len(y)
                ),
                "f1_macro": float(np.mean([
                    f1_score(
                        y, pred[n][r],
                        average="macro"
                    )
                    for r in range(REPEATS)
                ]))
            }
            for i, n in enumerate(all_names)
        },
        "selected": selection,
        "event_counts": {
            k: np.asarray(v).tolist()
            for k, v in counts.items()
        }
    }

    with open(
        SAVE_DIR / "haste_symbolic_AB.json",
        "w",
        encoding="utf-8"
    ) as fh:

        json.dump(result, fh, indent=2)

    print("\n" + "="*90)
    print("EXPERIMENT FINISHED")
    print("="*90)
    print("Results: results/haste_symbolic_AB.json")
    print("Figures: figures/haste_AB_01...09.png")


if __name__ == "__main__":
    main()
