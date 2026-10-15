# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""HASTE-AL-CS-G — Explicit joint geometry vs joint-state MDS."""

import json
import runpy
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
# 0. CONFIGURATION — UNCHANGED R5
# ============================================================

REFERENCE_FILE = Path("reference.py")

if not REFERENCE_FILE.is_file():
    raise FileNotFoundError(f"Reference not found: {REFERENCE_FILE.resolve()}")

R = runpy.run_path(str(REFERENCE_FILE))

SEED, N, K, T = R["SEED"], R["N"], R["K"], R["T"]

FIG, OUT = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

QS = (2, 4, 8)
LAMBDAS = (0., .125, .25, .5, 1., 2.)

MDS_DIMS = {"os": 8, "ot": 8, "st": 3}

BASE = "T0-R5-3Group-Reg-129D"

MODELS = {BASE: ()}

for scheme in ("Explicit", "JointMDS"):

    for suffix, blocks in (
        ("OS", ("os",)),
        ("OT", ("ot",)),
        ("ST", ("st",)),
        ("AllPairs", ("os", "ot", "st")),
        ("Transitions", ("ev_os", "ev_ot", "ev_st", "ev_pos"))
    ):
        MODELS[f"{scheme}-{suffix}"] = (scheme, blocks)


# ============================================================
# 1. GEOMETRY OF THE SYMBOLIC ALPHABETS
# ============================================================

def classical_mds(D2, dim):

    """Classical MDS on a fixed symbolic distance matrix."""

    row = D2.mean(axis=1)

    G = -.5*(D2-row[:, None]-row[None, :]+D2.mean())

    values, vectors = eigh(G, check_finite=False)

    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    positive = int(np.sum(values > max(values[0], 1.)*1e-11))

    if positive < dim:
        raise ValueError(
            f"MDS requires {dim} positive eigenvalues; found {positive}"
        )

    Y = vectors[:, :dim]*np.sqrt(values[:dim])

    # Deterministic eigenvector orientation.
    for j in range(dim):
        pivot = np.argmax(np.abs(Y[:, j]))
        if Y[pivot, j] < 0:
            Y[:, j] *= -1

    return Y


def scale_alphabet(A):

    """Normalize mean squared distance between distinct states to one."""

    scale = np.sqrt(
        max(float(np.mean(pdist(A, "sqeuclidean"))), 1e-12)
    )

    return A/scale


def make_lookups():

    # Ordinal alphabets — MDS 1D is exact.
    speed_values = np.linspace(0, 1, 4)
    turn_values = np.linspace(-1, 1, 5)

    S = classical_mds(
        squareform(pdist(speed_values[:, None], "sqeuclidean")), 1
    )

    U = classical_mds(
        squareform(pdist(turn_values[:, None], "sqeuclidean")), 1
    )

    if np.corrcoef(S[:, 0], speed_values)[0, 1] < 0:
        S *= -1

    if np.corrcoef(U[:, 0], turn_values)[0, 1] < 0:
        U *= -1

    # Orientation — original harmonic geometry.
    O = R["harmonics"](2*np.pi*np.arange(K)/K)

    alph = {
        "o": scale_alphabet(O),
        "s": scale_alphabet(S),
        "t": scale_alphabet(U)
    }

    pairs = {
        "os": ("o", "s"),
        "ot": ("o", "t"),
        "st": ("s", "t")
    }

    lookups = {"Explicit": {}, "JointMDS": {}}
    dimensions = {"Explicit": {}, "JointMDS": {}}
    diagnostics = {}

    for key, (ak, bk) in pairs.items():

        A0, B0 = alph[ak], alph[bk]

        ia = np.repeat(np.arange(len(A0)), len(B0))
        ib = np.tile(np.arange(len(B0)), len(A0))

        A, B = A0[ia], B0[ib]

        # ----------------------------------------------------
        # CONSTRUCTION 1 — Marginals + interaction
        # ----------------------------------------------------

        product = np.einsum(
            "na,nb->nab", A, B
        ).reshape(len(A), -1)

        explicit = np.column_stack((A, B, product))

        # ----------------------------------------------------
        # CONSTRUCTION 2 — Nonlinear joint-state geometry
        # ----------------------------------------------------

        da = squareform(pdist(A, "sqeuclidean"))
        db = squareform(pdist(B, "sqeuclidean"))

        # Squared RBF-kernel distance.
        D2 = 2.*(-np.expm1(-.5*(da+db)))
        np.fill_diagonal(D2, 0.)

        embedded = classical_mds(D2, MDS_DIMS[key])

        reconstructed = squareform(
            pdist(embedded, "sqeuclidean")
        )

        stress = (
            np.linalg.norm(reconstructed-D2) /
            max(np.linalg.norm(D2), 1e-12)
        )

        lookups["Explicit"][key] = explicit
        lookups["JointMDS"][key] = embedded

        dimensions["Explicit"][key] = explicit.shape[1]
        dimensions["JointMDS"][key] = embedded.shape[1]

        diagnostics[key] = {
            "states": len(A),
            "mds_dim": MDS_DIMS[key],
            "relative_squared_distance_error": float(stress)
        }

        # Verify that projected symbols remain distinguishable.
        for representation in (explicit, embedded):
            assert np.min(pdist(representation)) > 1e-10

    return lookups, dimensions, diagnostics


LOOKUPS, DIMENSIONS, MDS_DIAGNOSTICS = make_lookups()


# ============================================================
# 2. SYMBOLIC STREAMS — EXACT R5 QUANTIZATION
# ============================================================

def symbolic_streams(case):

    v = np.nan_to_num(
        np.asarray(case, float)[:2],
        nan=0., posinf=0., neginf=0.
    )

    if v.shape[0] != 2 or v.shape[1] < 2:
        raise ValueError("Expected two velocity channels")

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

    ix = np.minimum((T*n).astype(int), n-1)

    s = np.r_[
        0., np.cumsum(.5*(speed[:-1]+speed[1:]))
    ]

    if s[-1] > 1e-12:

        keep = np.r_[True, np.diff(s) > 1e-12]

        theta_arc = np.interp(
            T*s[-1], s[keep], np.unwrap(theta)[keep]
        )

        arc_speed = np.interp(
            T*s[-1], s[keep], speed[keep]
        )

    else:
        theta_arc, arc_speed = theta[ix], speed[ix]

    ids = R["quantize"](theta_arc)

    rel = arc_speed/max(float(speed[ix].mean()), 1e-8)

    sb = np.digitize(
        rel, (.5, 1., 2.)
    ).astype(int)

    turn = np.tanh(
        np.gradient(
            gaussian_filter1d(
                theta_arc, sigma=R["SIGMA"], mode="nearest"
            ), T
        )/(R["TURNING_SCALE"]*(N-1))
    )

    tb = np.digitize(
        turn, (-.6, -.2, .2, .6)
    ).astype(int)

    return ids, sb, tb


def check_r5_streams(X, count=12):

    """Verify exact agreement with original R5 symbolic blocks."""

    selected = np.linspace(
        0, len(X)-1, min(count, len(X)), dtype=int
    )

    for i in selected:

        ids, sb, tb = symbolic_streams(X[i])
        reference = R["encode"](X[i])

        checks = {
            "arc_q": dct(
                R["harmonics"](2*np.pi*ids/K),
                axis=0, type=2, norm="ortho"
            )[:R["FS"]].ravel(),

            "speed_arc": dct(
                np.linspace(0, 1, 4)[sb],
                type=2, norm="ortho"
            )[:R["FR"]],

            "turn_arc": dct(
                np.linspace(-1, 1, 5)[tb],
                type=2, norm="ortho"
            )[:R["FC"]]
        }

        for key, value in checks.items():

            if not np.allclose(
                value, reference[key], atol=1e-10
            ):
                raise RuntimeError(
                    f"R5 symbolic mismatch: sample {i}, block {key}"
                )

    print("Symbolic states match unchanged R5: OK", flush=True)


# ============================================================
# 3. JOINT SYMBOLIC TRAJECTORIES
# ============================================================

def streams_for(case, scheme):

    ids, sb, tb = symbolic_streams(case)

    indices = {
        "os": ids*4+sb,
        "ot": ids*5+tb,
        "st": sb*5+tb
    }

    streams = {
        key: LOOKUPS[scheme][key][ix]
        for key, ix in indices.items()
    }

    return (ids, sb, tb), streams


def joint_features(case):

    ids, sb, tb = symbolic_streams(case)

    indices = {
        "os": ids*4+sb,
        "ot": ids*5+tb,
        "st": sb*5+tb
    }

    qmax = max(QS)
    result = {}

    for scheme in ("Explicit", "JointMDS"):

        for key, (a, b) in zip(
            ("os", "ot", "st"),
            ((ids, sb), (ids, tb), (sb, tb))
        ):

            z = LOOKUPS[scheme][key][indices[key]]

            # DCT of the ordered joint trajectory.
            result[f"{scheme}_{key}"] = dct(
                z, axis=0, type=2, norm="ortho"
            )[:qmax]

            # Pairwise union of independent event boundaries.
            starts = np.flatnonzero(
                np.r_[
                    True,
                    (a[1:] != a[:-1]) |
                    (b[1:] != b[:-1])
                ]
            )

            delta = np.diff(z[starts], axis=0)

            series = (
                R["interp_rank"](delta)
                if len(delta)
                else np.zeros((R["NE"], z.shape[1]))
            )

            result[f"{scheme}_ev_{key}"] = dct(
                series, axis=0, type=2, norm="ortho"
            )[:qmax]

    # Positions of all joint events along normalized arc length.
    starts = np.flatnonzero(
        np.r_[
            True,
            (ids[1:] != ids[:-1]) |
            (sb[1:] != sb[:-1]) |
            (tb[1:] != tb[:-1])
        ]
    )

    positions = starts[1:]/(N-1)

    series = (
        R["interp_rank"](positions)-R["E"]
        if len(positions)
        else np.zeros(R["NE"])
    )

    result["ev_pos"] = dct(
        series, type=2, norm="ortho"
    )[:qmax, None]

    return result


def extract_joint(X):

    rows = [joint_features(case) for case in X]

    return {
        q: {
            key: np.stack([
                row[key][:q].ravel()
                for row in rows
            ]).astype(np.float32)
            for key in rows[0]
        }
        for q in QS
    }


# ============================================================
# 4. DISTANCES AND CLASSIFIER
# ============================================================

def matrices(features):

    return {
        key: squareform(
            pdist(values, "sqeuclidean")
        ).astype(np.float32)
        for key, values in features.items()
    }


def combined(base, joint, descriptor, lam):

    if lam == 0:
        return base

    scheme, keys = descriptor

    blocks = [
        "ev_pos" if key == "ev_pos" else f"{scheme}_{key}"
        for key in keys
    ]

    extra = tuple(
        sum(joint[key][side] for key in blocks)/len(blocks)
        for side in (0, 1)
    )

    return tuple(
        (base[side]+lam*extra[side])/(1.+lam)
        for side in (0, 1)
    )


def predict(pair, y, it, C, gamma):

    A, B = pair

    clf = SVC(
        C=C, kernel="precomputed"
    ).fit(np.exp(-gamma*A), y[it])

    return clf.predict(np.exp(-gamma*B))


# ============================================================
# 5. INNER CV — SELECTION OF q AND lambda
# ============================================================

def tune_joint(Dbase, Dq, y, it, seed, selected):

    factors, C, gamma = selected

    scores = {
        name: {
            (q, lam): []
            for q in QS for lam in LAMBDAS
        }
        for name in list(MODELS)[1:]
    }

    cv = StratifiedKFold(
        n_splits=R["INNER"],
        shuffle=True,
        random_state=seed
    )

    for ai, bi in cv.split(np.zeros(len(it)), y[it]):

        ia, ib = it[ai], it[bi]

        b0 = R["mix"](
            R["pair_basis"](Dbase, ia, ib),
            R["r5_weights"](factors)
        )

        joints = {
            q: R["pair_basis"](Dq[q], ia, ib)
            for q in QS
        }

        yp0 = predict(b0, y, ia, C, gamma)

        baseline = float(
            f1_score(
                y[ib], yp0,
                average="macro", zero_division=0
            )
        )

        for name, descriptor in list(MODELS.items())[1:]:

            for q in QS:

                scores[name][q, 0.].append(baseline)

                for lam in LAMBDAS[1:]:

                    pair = combined(
                        b0, joints[q], descriptor, lam
                    )

                    yp = predict(pair, y, ia, C, gamma)

                    scores[name][q, lam].append(
                        float(f1_score(
                            y[ib], yp,
                            average="macro", zero_division=0
                        ))
                    )

    means = {
        name: {
            key: float(np.mean(values))
            for key, values in group.items()
        }
        for name, group in scores.items()
    }

    # Prefer lower lambda and lower q in case of ties.
    winners = {
        name: max(
            values,
            key=lambda key: (
                values[key], -key[1], -key[0]
            )
        )
        for name, values in means.items()
    }

    return winners, means


# ============================================================
# 6. HISTORICAL FIGURES 01 AND 02
# ============================================================

def plot_error_figures(report, prefix="haste_geom"):

    names = list(report)

    errors = np.asarray(
        [report[name]["errors"] for name in names],
        dtype=float
    )

    means = errors.mean(axis=1)
    stds = errors.std(axis=1, ddof=1)

    # Figure 01 — Mean errors ± standard deviation.
    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.barh(
        names, means,
        xerr=stds, capsize=3,
        error_kw={"elinewidth": 1.2}
    )

    ax.invert_yaxis()

    ax.set(
        xlabel="Errors per 1422 trajectories",
        title="Symbolic geometry ablation — repeated nested CV"
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)

    for i, (m, sd) in enumerate(zip(means, stds)):
        ax.text(
            m+sd+.08, i,
            f"{m:.2f} ± {sd:.2f}",
            va="center", fontsize=8
        )

    ax.set_xlim(right=max(means+stds)+1.5)

    fig.tight_layout()
    fig.savefig(
        FIG/f"{prefix}_01_ablation.png",
        dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    # Figure 02 — Errors across CV repeats.
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, errors.shape[1]+1)

    for name, values in zip(names, errors):

        ax.plot(
            x, values, ".-",
            linewidth=1.25,
            markersize=3.5,
            label=name
        )

    ax.set(
        xlabel="CV repeat",
        ylabel="Classification errors",
        title="Paired repeated nested cross-validation"
    )

    ax.set_xticks(x)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(
        FIG/f"{prefix}_02_repeats.png",
        dpi=220, bbox_inches="tight"
    )
    plt.close(fig)


def save(fig, name):

    fig.tight_layout()
    fig.savefig(
        FIG/f"haste_geom_{name}.png",
        dpi=220, bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# 7. ADDITIONAL DIAGNOSTIC FIGURES 03–12
# ============================================================

def diagnostic_figures(X, y, predictions, report, choices,
                       nearest, folds, params, Dbase, Dq):

    names = list(MODELS)
    others = names[1:]
    repeats = R["REPEATS"]
    classes = sorted(np.unique(y), key=int)

    # --------------------------------------------------------
    # 03 — Confusion matrices
    # --------------------------------------------------------

    fig, axs = plt.subplots(2, 6, figsize=(21, 7))

    for ax, name in zip(axs.ravel(), names):

        cm = confusion_matrix(
            np.tile(y, repeats),
            predictions[name].ravel(),
            labels=classes
        )

        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=8)

        ax.set_xticks(
            range(len(classes)), classes,
            rotation=90, fontsize=5
        )

        ax.set_yticks(
            range(len(classes)), classes,
            fontsize=5
        )

    for ax in axs.ravel()[len(names):]:
        ax.axis("off")

    save(fig, "03_confusions")

    # --------------------------------------------------------
    # 04 — Errors by class
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 4.5))
    x, width = np.arange(len(classes)), .075

    for j, name in enumerate(names):

        values = [
            int((predictions[name][:, y == c] != c).sum())
            for c in classes
        ]

        ax.bar(
            x+(j-5)*width, values,
            width, label=name
        )

    ax.set_xticks(x, classes)

    ax.set(
        xlabel="True class",
        ylabel="Errors over 5 repeats",
        title="Class-specific errors"
    )

    ax.legend(fontsize=7, ncol=3)
    save(fig, "04_class_errors")

    # --------------------------------------------------------
    # 05 — Held-out error overlap
    # --------------------------------------------------------

    wrong = [
        (predictions[name] != y[None, :]).ravel()
        for name in names
    ]

    J = np.eye(len(names))

    for i in range(len(names)):
        for j in range(i+1, len(names)):

            common = (wrong[i] & wrong[j]).sum()
            union = (wrong[i] | wrong[j]).sum()

            J[i, j] = J[j, i] = common/max(union, 1)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")

    ax.set_xticks(
        range(len(names)), names,
        rotation=45, ha="right", fontsize=7
    )

    ax.set_yticks(
        range(len(names)), names, fontsize=7
    )

    fig.colorbar(im, ax=ax)
    ax.set_title("Held-out error overlap — Jaccard")
    save(fig, "05_overlap")

    # --------------------------------------------------------
    # 06 — Corrections and new errors relative to T0
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(13, 4.5))
    a = predictions[BASE] != y[None, :]

    for j, name in enumerate(others):

        b = predictions[name] != y[None, :]

        ax.bar(
            j-.19, (a & ~b).sum(), .38,
            label="Corrected" if j == 0 else None
        )

        ax.bar(
            j+.19, (~a & b).sum(), .38,
            label="New error" if j == 0 else None
        )

    ax.set_xticks(
        range(len(others)),
        others, rotation=30, ha="right", fontsize=8
    )

    ax.set(
        ylabel="OOF decisions over five repeats",
        title="Corrections and new errors vs R5"
    )

    ax.legend()
    save(fig, "06_corrections")

    # --------------------------------------------------------
    # 07 — Selection of q and lambda
    # --------------------------------------------------------

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))

    for name in others:

        selected = choices[name]

        axs[0].plot(
            LAMBDAS,
            [
                sum(v["lambda"] == lam for v in selected)
                for lam in LAMBDAS
            ],
            ".-", lw=1, ms=3, label=name
        )

        axs[1].plot(
            QS,
            [
                sum(
                    v["lambda"] > 0 and v["q"] == q
                    for v in selected
                )
                for q in QS
            ],
            ".-", lw=1, ms=3, label=name
        )

    axs[0].set(
        xlabel="Selected lambda",
        ylabel="Outer folds / 25",
        title="Interaction weighting"
    )

    axs[1].set(
        xlabel="Selected q (active only)",
        ylabel="Outer folds / 25",
        title="DCT resolution"
    )

    axs[1].legend(fontsize=6, ncol=2)
    save(fig, "07_selection")

    # --------------------------------------------------------
    # 08 — Difficult trajectories
    # --------------------------------------------------------

    target = "JointMDS-AllPairs"

    persistence = (
        predictions[target] != y[None, :]
    ).sum(axis=0)

    hard = [
        i for i in (619, 682, 517, 1253, 1254)
        if i < len(y) and persistence[i] > 0
    ]

    hard.extend(
        int(i)
        for i in np.argsort(-persistence, kind="stable")
        if persistence[i] > 0 and i not in hard
    )

    hard = hard[:6]

    if hard:

        fig, axs = plt.subplots(
            len(hard), 3,
            figsize=(11, 2.35*len(hard)),
            squeeze=False
        )

        for j, t in enumerate(hard):

            r = int(np.flatnonzero(
                predictions[target][:, t] != y[t]
            )[0])

            same, rival = nearest[r, t]

            R["trajectory"](
                axs[j, 0], X[t],
                f"Held-out #{t} | {y[t]} | wrong {persistence[t]}/5"
            )

            R["trajectory"](
                axs[j, 1], X[same],
                f"Train same #{same} | {y[same]}"
            )

            R["trajectory"](
                axs[j, 2], X[rival],
                f"Train rival #{rival} | {y[rival]}"
            )

        save(fig, "08_hard_cases")

    # --------------------------------------------------------
    # 09 — Comparison of symbolic trajectories
    # --------------------------------------------------------

    t = hard[0] if hard else 0

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    for row, scheme in enumerate(("Explicit", "JointMDS")):

        _, streams = streams_for(X[t], scheme)
        z = streams["st"]

        coeff = dct(
            z, axis=0, type=2, norm="ortho"
        )[:8]

        axs[row, 0].imshow(
            z.T, origin="lower", aspect="auto",
            extent=(0, 1, 0, z.shape[1])
        )

        axs[row, 1].imshow(
            coeff.T, origin="lower", aspect="auto"
        )

        axs[row, 0].set(
            title=f"#{t} — {scheme} speed × turn",
            xlabel="Normalized arc length",
            ylabel="Symbolic dimension"
        )

        axs[row, 1].set(
            title=f"{scheme} DCT",
            xlabel="DCT coefficient",
            ylabel="Symbolic dimension"
        )

    save(fig, "09_joint_trajectories")

    # --------------------------------------------------------
    # 10 — DCT spectral information
    # --------------------------------------------------------

    sample = np.linspace(
        0, len(y)-1, min(len(y), 100), dtype=int
    )

    power = {
        (scheme, key): np.zeros(N)
        for scheme in ("Explicit", "JointMDS")
        for key in ("os", "ot", "st")
    }

    for i in sample:

        for scheme in ("Explicit", "JointMDS"):

            _, streams = streams_for(X[i], scheme)

            for key in ("os", "ot", "st"):

                coeff = dct(
                    streams[key],
                    axis=0, type=2, norm="ortho"
                )

                power[scheme, key] += (
                    (coeff**2).sum(axis=1)/len(sample)
                )

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))

    for (scheme, key), spectrum in power.items():

        label = f"{scheme}-{key}"
        total = max(spectrum.sum(), 1e-12)

        axs[0].plot(
            range(32), spectrum[:32]/total,
            lw=1, label=label
        )

        axs[1].plot(
            QS,
            [spectrum[:q].sum()/total for q in QS],
            ".-", lw=1, label=label
        )

    axs[0].set(
        xlabel="DCT coefficient (0 = DC)",
        ylabel="Energy fraction",
        title="Spectral components"
    )

    axs[1].set(
        xlabel="Retained coefficients q",
        ylabel="Energy fraction",
        title="DCT truncation"
    )

    axs[1].set_ylim(0, 1.02)
    axs[1].legend(fontsize=7, ncol=2)
    save(fig, "10_spectral_information")

    # --------------------------------------------------------
    # 11 — Train-only MDS of resulting trajectory distances
    # --------------------------------------------------------

    exp_name = "Explicit-AllPairs"
    mds_name = "JointMDS-AllPairs"

    # Prefer a fold where at least one interaction is active.
    step = next(
        (
            i for i in range(len(folds))
            if choices[exp_name][i]["lambda"] > 0
            or choices[mds_name][i]["lambda"] > 0
        ),
        0
    )

    it, iv = folds[step]
    factors, C, gamma = params[step]

    b0 = R["mix"](
        R["pair_basis"](Dbase, it, iv),
        R["r5_weights"](factors)
    )

    pairs = [(BASE, b0)]

    for name in (exp_name, mds_name):

        q = choices[name][step]["q"]
        lam = choices[name][step]["lambda"]

        joint = (
            R["pair_basis"](Dq[q], it, iv)
            if lam else None
        )

        pairs.append((
            name,
            combined(b0, joint, MODELS[name], lam)
        ))

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (name, pair) in zip(axs, pairs):

        train, val = R["mds_oos"](*pair)

        ax.scatter(
            train[:, 0], train[:, 1],
            s=8, c="lightgray", alpha=.3
        )

        for c in ("7", "9", "10", "18"):

            mask = y[it] == c

            ax.scatter(
                train[mask, 0], train[mask, 1],
                s=12, label=f"Class {c}"
            )

        repeat = step//R["OUTER"]

        bad = predictions[name][repeat, iv] != y[iv]

        ax.scatter(
            val[bad, 0], val[bad, 1],
            marker="x", s=55, c="black",
            label="Held-out error"
        )

        title = f"{name} | fold {step+1}"

        if name != BASE:
            title += f" | λ={choices[name][step]['lambda']:g}"

        ax.set(
            title=title,
            xlabel="MDS 1",
            ylabel="MDS 2"
        )

    axs[-1].legend(fontsize=7)
    save(fig, "11_mds_comparison")

    # --------------------------------------------------------
    # 12 — Fold-by-fold stability
    # --------------------------------------------------------

    qmat = np.array([
        [
            v["q"] if v["lambda"] else 0
            for v in choices[name]
        ]
        for name in others
    ])

    lmat = np.array([
        [v["lambda"] for v in choices[name]]
        for name in others
    ])

    fig, axs = plt.subplots(
        2, 1, figsize=(15, 7),
        sharex=True
    )

    for ax, data, title in zip(
        axs,
        (qmat, lmat),
        ("Selected q (0 = inactive)", "Selected joint weight")
    ):

        im = ax.imshow(
            data, aspect="auto",
            interpolation="nearest",
            cmap="viridis"
        )

        fig.colorbar(
            im, ax=ax,
            fraction=.014, pad=.02
        )

        ax.set_yticks(
            range(len(others)),
            others, fontsize=8
        )

        ax.set_title(title)

    axs[1].set_xticks(
        range(len(folds)),
        range(1, len(folds)+1),
        fontsize=7
    )

    axs[1].set_xlabel("Outer fold (repeated CV)")
    save(fig, "12_stability")

    return hard


# ============================================================
# 8. MAIN — REPEATED NESTED TRAIN-ONLY CV
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

    print("Symbolic dimensions:", DIMENSIONS)
    print("Joint MDS diagnostics:", MDS_DIAGNOSTICS, flush=True)

    # Validate agreement with historical symbolic states.
    check_r5_streams(X)

    print("Extracting original R5 features...", flush=True)

    B = R["extract"](X)

    assert (
        sum(B[k].shape[1] for k in R["CORE"]) +
        sum(B[k].shape[1] for k in R["EXTRA"])
    ) == 129

    print("Extracting both joint representations...", flush=True)

    Fq = extract_joint(X)

    Dbase = matrices(B)

    Dq = {
        q: matrices(Fq[q])
        for q in QS
    }

    predictions = {
        name: np.empty(
            (R["REPEATS"], n), dtype=y.dtype
        )
        for name in MODELS
    }

    choices = {
        name: []
        for name in list(MODELS)[1:]
    }

    params, scores_log, folds = [], [], []

    nearest = np.full(
        (R["REPEATS"], n, 2),
        -1, dtype=int
    )

    outer = RepeatedStratifiedKFold(
        n_splits=R["OUTER"],
        n_repeats=R["REPEATS"],
        random_state=SEED
    )

    print("\n"+"="*90)
    print("REPEATED NESTED TRAIN-ONLY CV")
    print("="*90)

    for step, (it, iv) in enumerate(
        outer.split(np.zeros(n), y)
    ):

        folds.append((it.copy(), iv.copy()))

        r, f = divmod(step, R["OUTER"])

        # ----------------------------------------------------
        # Historical R5 selection — unchanged
        # ----------------------------------------------------

        selected, _, _ = R["inner_tune"](
            Dbase, y, it, seed=1000+step
        )

        key = selected[R["REG"]]
        params.append(key)

        yp0, _, _, _ = R["fit_predict"](
            Dbase, y, it, iv, key
        )

        predictions[BASE][r, iv] = yp0

        # ----------------------------------------------------
        # Inner selection of q and lambda
        # ----------------------------------------------------

        winners, means = tune_joint(
            Dbase, Dq, y, it,
            seed=1000+step,
            selected=key
        )

        scores_log.append({
            name: [
                {
                    "q": int(q),
                    "lambda": float(lam),
                    "f1": score
                }
                for (q, lam), score in vals.items()
            ]
            for name, vals in means.items()
        })

        factors, C, gamma = key

        b0 = R["mix"](
            R["pair_basis"](Dbase, it, iv),
            R["r5_weights"](factors)
        )

        cache = {}
        messages = [f"T0={int((yp0 != y[iv]).sum())}"]

        # ----------------------------------------------------
        # Explicit and JointMDS predictions
        # ----------------------------------------------------

        for name, descriptor in list(MODELS.items())[1:]:

            q, lam = winners[name]

            choices[name].append({
                "q": int(q),
                "lambda": float(lam)
            })

            if lam and q not in cache:
                cache[q] = R["pair_basis"](
                    Dq[q], it, iv
                )

            pair = combined(
                b0, cache.get(q), descriptor, lam
            )

            yp = predict(
                pair, y, it, C, gamma
            )

            if lam == 0 and not np.array_equal(yp, yp0):
                raise RuntimeError(
                    f"Zero-weight baseline mismatch: "
                    f"fold {step}, model {name}"
                )

            predictions[name][r, iv] = yp

            messages.append(
                f"{name}={int((yp != y[iv]).sum())} "
                f"(q={q}, lambda={lam:g})"
            )

            # Training neighbors for geometric diagnostics.
            if name == "JointMDS-AllPairs":

                distances = pair[1]

                for j, index in enumerate(iv):

                    same = np.flatnonzero(
                        y[it] == y[index]
                    )

                    rival = np.flatnonzero(
                        y[it] != y[index]
                    )

                    nearest[r, index] = (
                        it[same[np.argmin(distances[j, same])]],
                        it[rival[np.argmin(distances[j, rival])]]
                    )

        print(
            f"Repeat {r+1}/{R['REPEATS']} | "
            f"Fold {f+1}/{R['OUTER']} | "
            + " | ".join(messages),
            flush=True
        )

    # ========================================================
    # AGGREGATION
    # ========================================================

    report = {}

    print("\n"+"="*90)
    print("AGGREGATED TRAIN-ONLY CV")
    print("="*90)

    for name in MODELS:

        errors = [
            int((predictions[name][r] != y).sum())
            for r in range(R["REPEATS"])
        ]

        f1s = [
            f1_score(
                y, predictions[name][r],
                average="macro", zero_division=0
            )
            for r in range(R["REPEATS"])
        ]

        report[name] = {
            "errors": errors,
            "mean_errors": float(np.mean(errors)),
            "sd_errors": float(np.std(errors, ddof=1)),
            "accuracy": float(1.-np.mean(errors)/n),
            "macro_f1": float(np.mean(f1s))
        }

        print(
            f"{name:29s} | "
            f"Errors={errors} | "
            f"Mean={np.mean(errors):.2f} | "
            f"Acc={100*report[name]['accuracy']:.4f}% | "
            f"F1={np.mean(f1s):.5f}"
        )

    # ========================================================
    # EXACT REPRODUCTION OF R5
    # ========================================================

    expected = [6, 4, 5, 3, 7]

    if report[BASE]["errors"] != expected:
        raise RuntimeError(
            f"R5 mismatch: {report[BASE]['errors']} "
            f"!= {expected}"
        )

    print("\nExact R5 reproduction: OK", flush=True)

    # ========================================================
    # FIGURES
    # ========================================================

    plot_error_figures(
        report, prefix="haste_geom"
    )

    hard = diagnostic_figures(
        X, y, predictions, report,
        choices, nearest, folds,
        params, Dbase, Dq
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    np.savez_compressed(
        OUT/"haste_geom_oof.npz",
        y=y,
        **{
            f"pred_{i}": predictions[name]
            for i, name in enumerate(MODELS)
        }
    )

    output = {
        "protocol": "Train-only nested CV, outer 5x5, inner 3",
        "reference": str(REFERENCE_FILE),
        "q_grid": QS,
        "lambda_grid": LAMBDAS,
        "model_order": list(MODELS),
        "scheme_dimensions": DIMENSIONS,
        "mds_diagnostics": MDS_DIAGNOSTICS,
        "report": report,
        "selected_q_lambda": choices,
        "r5_parameters": [
            {
                "factors": list(p),
                "C": float(C),
                "gamma": float(gamma)
            }
            for p, C, gamma in params
        ],
        "inner_joint_scores": scores_log,
        "hard_indices": hard,
        "official_test_used": False
    }

    (OUT/"haste_geom_results.json").write_text(
        json.dumps(
            output, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nEXPERIMENT FINISHED")
    print("Saved: results/haste_geom_results.json")
    print("Saved: results/haste_geom_oof.npz")
    print("Saved: figures/haste_geom_01...12.png")


if __name__ == "__main__":
    main()
 