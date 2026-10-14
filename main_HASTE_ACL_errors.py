# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from itertools import combinations
from aeon.datasets import load_classification
from scipy.fft import dct
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


# ============================================================
# HASTE-AL-C 68D — GEOMETRY AND OVO DECISION ANALYSIS
# ============================================================

SEED = 2027

K, H, N_TIME = 24, 2, 241
F_TIME, F_SPEED, F_ARC, F_CURV = 8, 16, 4, 4

W_SPEED, W_ARC, W_CURV = 0.25, 4.0, 2.0
SIGMA, TURNING_SCALE = 0.5, 0.20

C_VALUES = (10.0, 100.0)
GAMMA_VALUES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)

N_INNER, N_OUTER, N_REPEATS = 3, 5, 5

FOCUS_CLASSES = ("7", "9", "10", "18")
FOCUS_PAIRS = (("7", "18"), ("9", "10"), ("10", "18"), ("7", "10"))
TARGET_RIVALS = {619: "10", 682: "18", 1231: "7"}

GRID = np.linspace(0.0, 1.0, N_TIME)
DENOM = 1.0+W_SPEED+W_ARC+W_CURV


# ============================================================
# DATA — OFFICIAL TRAIN SPLIT ONLY
# ============================================================

X_train, y_train = load_classification(
    "CharacterTrajectories", split="train", load_equal_length=False
)

X_train, y_train = list(X_train), np.asarray(y_train)
N = len(y_train)

assert N > max(TARGET_RIVALS)

print(f"Train={N} | Classes={len(np.unique(y_train))}")
print(f"Outer={N_REPEATS}x{N_OUTER} | Inner={N_INNER}")


# ============================================================
# ANGULAR EMBEDDING
# ============================================================

def quantize_angles(theta):

    symbols = np.floor((theta+np.pi/K)/(2*np.pi/K)).astype(int) % K
    return 2*np.pi*symbols/K


def harmonics(theta):

    return np.stack([fn(h*theta)*np.sqrt(2.0/K)
                     for h in range(1, H+1)
                     for fn in (np.cos, np.sin)], axis=1)


# ============================================================
# FIXED 68-DIMENSIONAL FEATURE EXTRACTION
# ============================================================

def encode_one(case):

    v = np.nan_to_num(np.asarray(case, dtype=np.float64)[:2],
                      nan=0.0, posinf=0.0, neginf=0.0)

    vx, vy = v
    n = v.shape[1]
    speed = np.hypot(vx, vy)

    positive = speed[speed > 1e-10]
    cutoff = max(0.03*np.median(positive), 1e-10) if positive.size else 1e-10

    theta = np.arctan2(vy, vx)
    valid = speed > cutoff

    if valid.any():
        previous = np.where(valid, np.arange(n), -1)
        np.maximum.accumulate(previous, out=previous)
        previous[previous < 0] = np.flatnonzero(valid)[0]
        theta = theta[previous]
    else:
        theta = np.zeros(n)

    # Temporal orientation — 32D

    idx = np.minimum((GRID*n).astype(int), n-1)

    ft = dct(harmonics(quantize_angles(theta[idx])),
             type=2, axis=0, norm="ortho")[:F_TIME].ravel()

    # Relative speed — 16D

    relative_speed = speed[idx]/max(float(speed[idx].mean()), 1e-8)
    fr = dct(relative_speed, type=2, norm="ortho")[:F_SPEED]

    # Normalized arc-length parameterization

    ds = 0.5*(speed[:-1]+speed[1:])
    s = np.r_[0.0, np.cumsum(ds)]

    if s[-1] > 1e-12:
        keep = np.r_[True, np.diff(s) > 1e-12]
        theta_arc = np.interp(GRID*s[-1], s[keep], np.unwrap(theta)[keep])
    else:
        theta_arc = theta[idx]

    # Mixed curvilinear orientation — 16D

    zq = harmonics(quantize_angles(theta_arc))
    zc = harmonics(theta_arc)

    fs = dct(0.5*zq+0.5*zc,
             type=2, axis=0, norm="ortho")[:F_ARC].ravel()

    # Angular turning rate — 4D

    theta_smooth = gaussian_filter1d(
        theta_arc, sigma=SIGMA, mode="nearest"
    )

    turning = np.gradient(theta_smooth, GRID)
    turning = np.tanh(turning/(TURNING_SCALE*(N_TIME-1)))

    fc = dct(turning, type=2, norm="ortho")[:F_CURV]

    return ft, fr, fs, fc


# ============================================================
# PRECOMPUTE FEATURES AND DISTANCES
# ============================================================

print("\nExtracting 68D features...", flush=True)

encoded = [encode_one(x) for x in X_train]

FEATURES = {
    key: np.stack([z[j] for z in encoded]).astype(np.float32)
    for j, key in enumerate(("time", "speed", "arc", "curvature"))
}

assert sum(f.shape[1] for f in FEATURES.values()) == 68

DISTANCES = {
    key: squareform(pdist(X, metric="sqeuclidean")).astype(np.float32)
    for key, X in FEATURES.items()
}

del FEATURES, encoded


# ============================================================
# TRAIN-FOLD DISTANCE NORMALIZATION
# ============================================================

def model_distance(it, iv):

    n, A, B = len(it), None, None

    for key, weight in (
        ("time", 1.0),
        ("speed", W_SPEED),
        ("arc", W_ARC),
        ("curvature", W_CURV)
    ):

        D = DISTANCES[key]

        a = D[np.ix_(it, it)].astype(np.float64)
        b = D[np.ix_(iv, it)].astype(np.float64)

        scale = max(float(a.sum())/(n*(n-1)), 1e-12)

        a, b = weight*a/scale, weight*b/scale
        A = a if A is None else A+a
        B = b if B is None else B+b

    return np.maximum(A/DENOM, 0.0), np.maximum(B/DENOM, 0.0)


# ============================================================
# INNER CV — OPTIMIZE ONLY C AND GAMMA
# ============================================================

def tune_svm(it_outer, seed):

    cv = StratifiedKFold(
        n_splits=N_INNER, shuffle=True, random_state=seed
    )

    folds = []

    for a, b in cv.split(np.zeros(len(it_outer)), y_train[it_outer]):

        it, iv = it_outer[a], it_outer[b]
        A, B = model_distance(it, iv)

        folds.append((it, iv, A, B))

    best = (-np.inf, None, None)

    for gamma in GAMMA_VALUES:

        kernels = [
            (it, iv, np.exp(-gamma*A), np.exp(-gamma*B))
            for it, iv, A, B in folds
        ]

        for C in C_VALUES:

            scores = []

            for it, iv, Ktr, Kval in kernels:

                clf = SVC(
                    C=C, kernel="precomputed",
                    decision_function_shape="ovo"
                )

                clf.fit(Ktr, y_train[it])
                pred = clf.predict(Kval)

                scores.append(f1_score(
                    y_train[iv], pred, average="macro", zero_division=0
                ))

            score = float(np.mean(scores))

            if score > best[0]:
                best = (score, C, gamma)

    return best


# ============================================================
# OVO PAIR SCORES
# Positive score favors the first class in the pair.
# ============================================================

def ovo_details(model, kernel_row):

    classes = list(model.classes_)
    pairs = list(combinations(classes, 2))

    values = np.asarray(
        model.decision_function(kernel_row.reshape(1, -1))
    ).ravel()

    votes = Counter({str(c): 0 for c in classes})
    pair_values = {}

    for (a, b), value in zip(pairs, values):

        votes[str(a if value > 0 else b)] += 1
        pair_values[(str(a), str(b))] = float(value)

    return votes, pair_values


def oriented_score(pair_values, a, b):

    if (str(a), str(b)) in pair_values:
        return pair_values[(str(a), str(b))]

    return -pair_values[(str(b), str(a))]


# ============================================================
# REPEATED NESTED CV — OOF GEOMETRY AND TARGET AUDIT
# ============================================================

outer = RepeatedStratifiedKFold(
    n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED
)

OOF = np.empty((N_REPEATS, N), dtype=y_train.dtype)
RHO = np.full((N_REPEATS, N), np.nan)

SELECTED, CASES = [], []
REF_TRAIN, REF_D = None, None

print("\n"+"="*100)
print("REPEATED NESTED CV")
print("="*100, flush=True)

for iteration, (it, iv) in enumerate(outer.split(np.zeros(N), y_train)):

    repeat, fold = divmod(iteration, N_OUTER)
    inner, C, gamma = tune_svm(it, seed=1000+iteration)

    A, B = model_distance(it, iv)

    if iteration == 0:
        REF_TRAIN, REF_D = it.copy(), A.copy()

    Ktr, Kval = np.exp(-gamma*A), np.exp(-gamma*B)

    svm = SVC(
        C=C, kernel="precomputed",
        decision_function_shape="ovo"
    )

    svm.fit(Ktr, y_train[it])
    pred = svm.predict(Kval)

    OOF[repeat, iv] = pred
    SELECTED.append((C, gamma))

    # Local geometric separation for held-out observations

    for row, sample in enumerate(iv):

        same_mask = y_train[it] == y_train[sample]

        d_same = B[row, same_mask].min()
        d_rival = B[row, ~same_mask].min()

        RHO[repeat, sample] = d_rival/max(d_same, 1e-12)

    # Detailed OVO diagnostics for difficult trajectories

    for target, rival in TARGET_RIVALS.items():

        where = np.flatnonzero(iv == target)

        if not where.size:
            continue

        row = int(where[0])
        true = str(y_train[target])

        same = np.flatnonzero(y_train[it] == y_train[target])
        other = np.flatnonzero(y_train[it] != y_train[target])

        same_pos = int(same[np.argmin(B[row, same])])
        rival_pos = int(other[np.argmin(B[row, other])])

        votes, pair_values = ovo_details(svm, Kval[row])

        top_neighbors = np.argsort(
            B[row], kind="stable"
        )[:7]

        CASES.append({
            "target": target,
            "repeat": repeat+1,
            "true": true,
            "pred": str(pred[row]),
            "rho": float(RHO[repeat, target]),
            "same_id": int(it[same_pos]),
            "rival_id": int(it[rival_pos]),
            "rival_class": str(y_train[it[rival_pos]]),
            "top7": [
                (int(it[j]), str(y_train[it[j]]))
                for j in top_neighbors
            ],
            "votes": votes,
            "signed_true_vs_rival": oriented_score(
                pair_values, true, rival
            )
        })

    acc = accuracy_score(y_train[iv], pred)
    f1 = f1_score(y_train[iv], pred, average="macro", zero_division=0)

    print(
        f"Repeat {repeat+1}/{N_REPEATS} | Fold {fold+1}/{N_OUTER} | "
        f"Acc={acc:.4f} | F1={f1:.4f} | Inner={inner:.4f} | "
        f"C={C:g} | gamma={gamma:g}",
        flush=True
    )


# ============================================================
# AGGREGATED RESULTS
# ============================================================

accs = np.array([
    accuracy_score(y_train, OOF[r])
    for r in range(N_REPEATS)
])

f1s = np.array([
    f1_score(y_train, OOF[r], average="macro", zero_division=0)
    for r in range(N_REPEATS)
])

errors = [
    int(np.sum(OOF[r] != y_train))
    for r in range(N_REPEATS)
]

correct = OOF == y_train[None, :]

print("\n"+"="*100)
print("AGGREGATED RESULTS")
print("="*100)

print(
    f"SVM-68D | Acc={accs.mean():.4f} ± {accs.std(ddof=1):.4f} | "
    f"F1={f1s.mean():.4f} ± {f1s.std(ddof=1):.4f} | Errors={errors}"
)

print("SVM params:", Counter(SELECTED).most_common())

confusions = Counter(
    (str(y_train[i]), str(OOF[r, i]))
    for r in range(N_REPEATS)
    for i in range(N)
    if OOF[r, i] != y_train[i]
)

print("Main confusions:", confusions.most_common(15))
print("Reproducibility:", "OK" if errors == [3, 7, 7, 8, 9] else "DIFFERENT")


# ============================================================
# HELD-OUT LOCAL GEOMETRY
# ============================================================

print("\n"+"="*100)
print("HELD-OUT NEIGHBOR RATIOS")
print("="*100)

print(
    f"Correct samples | Median rho={np.median(RHO[correct]):.3f} | "
    f"Fraction rho<1={np.mean(RHO[correct] < 1):.3f}"
)

print(
    f"Wrong samples   | Median rho={np.median(RHO[~correct]):.3f} | "
    f"Fraction rho<1={np.mean(RHO[~correct] < 1):.3f}"
)

for cls in FOCUS_CLASSES:

    mask = np.asarray([str(y) == cls for y in y_train])

    rr = RHO[:, mask]
    cc = correct[:, mask]

    print(
        f"Class {cls:>2s} | Errors={int((~cc).sum()):2d} | "
        f"Median rho={np.median(rr):.3f} | "
        f"Fraction rho<1={np.mean(rr < 1):.3f}"
    )


# ============================================================
# TARGET-LEVEL OVO AND NEIGHBOR AUDIT
# ============================================================

print("\n"+"="*100)
print("TARGET-LEVEL OVO AND NEAREST-NEIGHBOR AUDIT")
print("="*100)

for target, rival in TARGET_RIVALS.items():

    rec = sorted(
        [z for z in CASES if z["target"] == target],
        key=lambda z: z["repeat"]
    )

    print(
        f"\nTarget #{target} | Class {rec[0]['true']} | "
        f"Rival class {rival}"
    )

    print("Predictions:", Counter(z["pred"] for z in rec))

    for z in rec:

        focus_votes = {
            c: z["votes"][c]
            for c in FOCUS_CLASSES
        }

        nn_votes = dict(Counter(
            label for _, label in z["top7"]
        ))

        print(
            f"R{z['repeat']} | Pred={z['pred']:>2s} | "
            f"rho={z['rho']:.3f} | "
            f"OVO(true vs {rival})={z['signed_true_vs_rival']:+.4f} | "
            f"Votes={focus_votes} | "
            f"Same=#{z['same_id']} | "
            f"Rival=#{z['rival_id']} ({z['rival_class']}) | "
            f"7-NN={nn_votes}"
        )


# ============================================================
# TRAJECTORY VISUALIZATION HELPERS
# ============================================================

def reconstructed_path(case):

    v = np.nan_to_num(
        np.asarray(case, dtype=np.float64)[:2],
        nan=0.0, posinf=0.0, neginf=0.0
    )

    xy = np.cumsum(v.T, axis=0)
    xy -= xy[0]

    length = np.sum(
        np.linalg.norm(np.diff(xy, axis=0), axis=1)
    )

    return xy/max(float(length), 1e-12)


def plot_path(ax, index, title):

    xy = reconstructed_path(X_train[index])

    ax.plot(xy[:, 0], xy[:, 1], linewidth=1.5)
    ax.scatter(xy[0, 0], xy[0, 1], s=23, marker="o")
    ax.scatter(xy[-1, 0], xy[-1, 1], s=23, marker="x")

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


# ============================================================
# FIGURE 1 — HARD CASES AND TRAIN-ONLY NEIGHBORS
# ============================================================

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for row, target in enumerate(TARGET_RIVALS):

    rec = next(
        z for z in CASES
        if z["target"] == target and z["repeat"] == 1
    )

    plot_path(
        axes[row, 0], target,
        f"Target #{target} | True {rec['true']} | SVM {rec['pred']}"
    )

    plot_path(
        axes[row, 1], rec["same_id"],
        f"Same-class neighbor #{rec['same_id']}"
    )

    plot_path(
        axes[row, 2], rec["rival_id"],
        f"Rival #{rec['rival_id']} | Class {rec['rival_class']}"
    )

fig.suptitle("Hard cases and training-only nearest neighbors", fontsize=13)
fig.tight_layout()
plt.show()


# ============================================================
# FIGURE 2 — CLASS MEDOIDS AND MORPHOLOGICAL VARIABILITY
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

for ax, cls in zip(axes.flat, FOCUS_CLASSES):

    pos = np.flatnonzero(
        np.asarray([str(y) == cls for y in y_train[REF_TRAIN]])
    )

    Da = REF_D[np.ix_(pos, pos)]
    center = int(pos[np.argmin(Da.mean(axis=1))])

    ordered = pos[np.argsort(REF_D[center, pos])]

    selection = ordered[
        np.linspace(0, len(ordered)-1, min(7, len(ordered)), dtype=int)
    ]

    for j in selection:

        xy = reconstructed_path(X_train[int(REF_TRAIN[j])])
        ax.plot(xy[:, 0], xy[:, 1], linewidth=0.9, alpha=0.45)

    medoid_id = int(REF_TRAIN[center])
    xy = reconstructed_path(X_train[medoid_id])

    ax.plot(
        xy[:, 0], xy[:, 1],
        linewidth=2.5, label=f"Medoid #{medoid_id}"
    )

    ax.set_title(f"Class {cls} — training-fold examples")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)

fig.suptitle(
    "Class geometry — medoids and within-class variability",
    fontsize=13
)

fig.tight_layout()
plt.show()


# ============================================================
# FIGURE 3 — WITHIN-CLASS AND BETWEEN-CLASS DISTANCES
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (a, b) in zip(axes.flat, FOCUS_PAIRS):

    ia = np.flatnonzero(
        np.asarray([str(y) == a for y in y_train[REF_TRAIN]])
    )

    ib = np.flatnonzero(
        np.asarray([str(y) == b for y in y_train[REF_TRAIN]])
    )

    Da = REF_D[np.ix_(ia, ia)]
    Db = REF_D[np.ix_(ib, ib)]

    same_a = Da[np.triu_indices(len(ia), k=1)]
    same_b = Db[np.triu_indices(len(ib), k=1)]
    cross = REF_D[np.ix_(ia, ib)].ravel()

    ceiling = np.quantile(
        np.concatenate([same_a, same_b, cross]), 0.98
    )

    bins = np.linspace(0, max(float(ceiling), 1e-9), 36)

    for values, label in (
        (same_a, f"{a}-{a}"),
        (same_b, f"{b}-{b}"),
        (cross, f"{a}-{b}")
    ):

        ax.hist(
            values[values <= ceiling], bins=bins,
            density=True, histtype="step",
            linewidth=1.4, label=label
        )

    ax.set_title(f"Classes {a} vs {b}")
    ax.set_xlabel("Normalized squared distance D")
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

fig.suptitle(
    "Within-class versus between-class geometry — first training fold",
    fontsize=12
)

fig.tight_layout()
plt.show()


# ============================================================
# FIGURE 4 — HELD-OUT LOCAL GEOMETRIC SEPARATION
# ============================================================

fig, ax = plt.subplots(figsize=(9, 4.5))

data = [
    np.log10(np.maximum(
        RHO[:, np.asarray([str(y) == cls for y in y_train])].ravel(),
        1e-12
    ))
    for cls in FOCUS_CLASSES
]

ax.boxplot(data, showfliers=False)

for j, cls in enumerate(FOCUS_CLASSES, start=1):

    mask = np.asarray([str(y) == cls for y in y_train])

    wrong_ratios = RHO[:, mask][~correct[:, mask]]

    ax.scatter(
        np.full(len(wrong_ratios), j+0.12),
        np.log10(np.maximum(wrong_ratios, 1e-12)),
        marker="x", s=34,
        label="SVM errors" if j == 1 else None
    )

ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)

ax.set_xticks(
    range(1, len(FOCUS_CLASSES)+1),
    [f"Class {c}" for c in FOCUS_CLASSES]
)

ax.set_ylabel("log10 (nearest rival D / nearest same-class D)")
ax.set_title("Held-out geometric separation — 5 repeated outer folds")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="best")

fig.tight_layout()
plt.show()


# ============================================================
# FIGURE 5 — PAIRWISE OVO DECISIONS FOR HARD CASES
# ============================================================

fig, ax = plt.subplots(figsize=(9, 4.5))

for target, rival in TARGET_RIVALS.items():

    rec = sorted(
        [z for z in CASES if z["target"] == target],
        key=lambda z: z["repeat"]
    )

    ax.plot(
        [z["repeat"] for z in rec],
        [z["signed_true_vs_rival"] for z in rec],
        marker="o", linewidth=1.4,
        label=f"#{target}: true {rec[0]['true']} vs {rival}"
    )

ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)

ax.set_xticks(range(1, N_REPEATS+1))
ax.set_xlabel("Outer CV repeat")
ax.set_ylabel("Signed OVO score — positive favors true class")
ax.set_title("Pairwise SVM decisions on persistent errors")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9)

fig.tight_layout()
plt.show()
