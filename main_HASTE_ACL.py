# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from aeon.datasets import load_classification
from scipy.fft import dct
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


# ============================================================
# HASTE-AL-C — LOW-SMOOTHING AND GEOMETRIC DIAGNOSTICS
# ============================================================

SEED = 2027

K, H, N_TIME = 24, 2, 241
F_TIME, F_SPEED, F_ARC, F_CURV = 8, 16, 4, 4

W_SPEED, W_ARC, W_CURV = 0.25, 4.0, 2.0
TURNING_SCALE = 0.20

SIGMAS = (1.5, 0.75, 0.5, 0.25, 0.0)
CONFIGS = {f"Sigma-{s:.2f}": s for s in SIGMAS}

C_VALUES = (10.0, 100.0)
GAMMA_VALUES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)

N_INNER, N_OUTER, N_REPEATS = 3, 5, 5

TARGET_IDS = (619, 682, 1231)
DIAG_SIGMA = 0.5
PLOT_DIAGNOSTICS = True

GRID = np.linspace(0.0, 1.0, N_TIME)
DENOM = 1.0+W_SPEED+W_ARC+W_CURV


# ============================================================
# DATA — OFFICIAL TRAIN SPLIT ONLY
# ============================================================

X_train, y_train = load_classification(
    "CharacterTrajectories", split="train", load_equal_length=False
)

X_train = list(X_train)
y_train = np.asarray(y_train)
N = len(y_train)

assert max(TARGET_IDS) < N

print(f"Train={N} | Classes={len(np.unique(y_train))}")
print(f"Outer CV={N_REPEATS} x {N_OUTER} | Inner CV={N_INNER}")
print(f"Sigmas={SIGMAS}")


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


def curvature_signal(theta_arc, sigma):

    if sigma == 0:
        theta_smooth = theta_arc
    else:
        theta_smooth = gaussian_filter1d(
            theta_arc, sigma=sigma, mode="nearest"
        )

    turning = np.gradient(theta_smooth, GRID)
    return np.tanh(turning/(TURNING_SCALE*(N_TIME-1)))


# ============================================================
# FEATURE EXTRACTION
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

    # Curvature — 4D for each sigma

    fc = {
        sigma: dct(curvature_signal(theta_arc, sigma),
                   type=2, norm="ortho")[:F_CURV]
        for sigma in SIGMAS
    }

    return ft, fr, fs, fc, theta_arc


# ============================================================
# PRECOMPUTE FEATURES AND DISTANCES
# ============================================================

print("\nExtracting features...", flush=True)

encoded = [encode_one(x) for x in X_train]

FEATURES = {
    "time": np.stack([e[0] for e in encoded]).astype(np.float32),
    "speed": np.stack([e[1] for e in encoded]).astype(np.float32),
    "arc": np.stack([e[2] for e in encoded]).astype(np.float32)
}

for sigma in SIGMAS:
    FEATURES[sigma] = np.stack([e[3][sigma] for e in encoded]).astype(np.float32)

THETA_TARGETS = {i: encoded[i][4].copy() for i in TARGET_IDS}
del encoded

print("Feature dimensions:", {str(k): v.shape[1] for k, v in FEATURES.items()})
print("\nComputing pairwise distances...", flush=True)

DISTANCES = {
    key: squareform(pdist(X, metric="sqeuclidean")).astype(np.float32)
    for key, X in FEATURES.items()
}

del FEATURES


# ============================================================
# TRAIN-FOLD DISTANCE NORMALIZATION
# ============================================================

def fold_blocks(it, iv):

    blocks = {}
    n = len(it)

    for key, D in DISTANCES.items():

        A = D[np.ix_(it, it)].astype(np.float64)
        B = D[np.ix_(iv, it)].astype(np.float64)

        scale = max(float(A.sum())/(n*(n-1)), 1e-12)
        blocks[key] = (A/scale, B/scale)

    return blocks


def model_distance(blocks, sigma):

    At, Bt = blocks["time"]
    Ar, Br = blocks["speed"]
    As, Bs = blocks["arc"]
    Ac, Bc = blocks[sigma]

    A = (At+W_SPEED*Ar+W_ARC*As+W_CURV*Ac)/DENOM
    B = (Bt+W_SPEED*Br+W_ARC*Bs+W_CURV*Bc)/DENOM

    return np.maximum(A, 0.0), np.maximum(B, 0.0)


# ============================================================
# INNER CV — OPTIMIZE ONLY C AND GAMMA
# ============================================================

def tune_models(train_ids, seed):

    cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    folds = []

    for a, b in cv.split(np.zeros(len(train_ids)), y_train[train_ids]):
        it, iv = train_ids[a], train_ids[b]
        folds.append((it, iv, fold_blocks(it, iv)))

    selected = {}

    for name, sigma in CONFIGS.items():

        pairs = [(it, iv, *model_distance(blocks, sigma))
                 for it, iv, blocks in folds]

        best = (-np.inf, None, None)

        for gamma in GAMMA_VALUES:

            kernels = [(it, iv, np.exp(-gamma*A), np.exp(-gamma*B))
                       for it, iv, A, B in pairs]

            for C in C_VALUES:

                scores = []

                for it, iv, Ktr, Kval in kernels:

                    model = SVC(C=C, kernel="precomputed")
                    model.fit(Ktr, y_train[it])
                    pred = model.predict(Kval)

                    scores.append(f1_score(
                        y_train[iv], pred, average="macro", zero_division=0
                    ))

                score = float(np.mean(scores))

                if score > best[0]:
                    best = (score, C, gamma)

        selected[name] = best

    return selected


# ============================================================
# TRAIN-ONLY NEAREST-NEIGHBOR DIAGNOSTIC
# ============================================================

def neighbor_diagnostic(target, it, iv, B, pred, repeat, sigma):

    pos = int(np.flatnonzero(iv == target)[0])
    distances = B[pos]
    labels = y_train[it]
    true_label = y_train[target]

    same = np.flatnonzero(labels == true_label)
    other = np.flatnonzero(labels != true_label)

    a = same[np.argmin(distances[same])]
    b = other[np.argmin(distances[other])]

    return {
        "index": target,
        "repeat": repeat,
        "sigma": sigma,
        "true": str(true_label),
        "pred": str(pred[pos]),
        "same_id": int(it[a]),
        "other_id": int(it[b]),
        "other_class": str(labels[b]),
        "d_same": float(distances[a]),
        "d_other": float(distances[b])
    }


# ============================================================
# REPEATED NESTED CROSS-VALIDATION
# ============================================================

outer = RepeatedStratifiedKFold(
    n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED
)

OOF = {name: np.empty((N_REPEATS, N), dtype=y_train.dtype) for name in CONFIGS}
SELECTIONS = {name: [] for name in CONFIGS}
DIAGNOSTICS = []

print("\n"+"="*100)
print("REPEATED NESTED CROSS-VALIDATION")
print("="*100, flush=True)

for iteration, (it, iv) in enumerate(outer.split(np.zeros(N), y_train)):

    repeat, fold = divmod(iteration, N_OUTER)
    selected = tune_models(it, seed=1000+iteration)
    blocks = fold_blocks(it, iv)

    print(f"\nRepeat {repeat+1}/{N_REPEATS} | Fold {fold+1}/{N_OUTER}", flush=True)

    for name, sigma in CONFIGS.items():

        inner_f1, C, gamma = selected[name]
        A, B = model_distance(blocks, sigma)

        model = SVC(C=C, kernel="precomputed")
        model.fit(np.exp(-gamma*A), y_train[it])
        pred = model.predict(np.exp(-gamma*B))

        OOF[name][repeat, iv] = pred
        SELECTIONS[name].append((C, gamma))

        acc = accuracy_score(y_train[iv], pred)
        f1 = f1_score(y_train[iv], pred, average="macro", zero_division=0)

        print(f"{name:12s} | Acc={acc:.4f} | F1={f1:.4f} | "
              f"Inner={inner_f1:.4f} | C={C:g} | gamma={gamma:g}", flush=True)

        if sigma in (1.5, 0.5, 0.0):
            for target in TARGET_IDS:
                if target in iv:
                    DIAGNOSTICS.append(
                        neighbor_diagnostic(target, it, iv, B, pred, repeat, sigma)
                    )


# ============================================================
# AGGREGATED RESULTS
# ============================================================

RESULTS = {}

print("\n"+"="*100)
print("AGGREGATED RESULTS")
print("="*100)

for name in CONFIGS:

    accuracies = np.array([
        accuracy_score(y_train, OOF[name][r]) for r in range(N_REPEATS)
    ])

    f1s = np.array([
        f1_score(y_train, OOF[name][r], average="macro", zero_division=0)
        for r in range(N_REPEATS)
    ])

    errors = [int(np.sum(OOF[name][r] != y_train)) for r in range(N_REPEATS)]

    RESULTS[name] = {"accuracy": accuracies, "f1": f1s, "errors": errors}

    print(f"{name:12s} | Acc={accuracies.mean():.4f} ± {accuracies.std(ddof=1):.4f} | "
          f"F1={f1s.mean():.4f} ± {f1s.std(ddof=1):.4f} | Errors={errors}")


# ============================================================
# PAIRED COMPARISONS
# ============================================================

print("\n"+"="*100)
print("PAIRED COMPARISONS VS SIGMA=0.50")
print("="*100)

reference = OOF["Sigma-0.50"]

for name in CONFIGS:

    if name == "Sigma-0.50":
        continue

    print(f"\nSigma-0.50 vs {name}")

    for r in range(N_REPEATS):

        ref = reference[r] == y_train
        alt = OOF[name][r] == y_train

        corrected = int(np.sum(~ref & alt))
        introduced = int(np.sum(ref & ~alt))

        n = corrected+introduced
        p = binomtest(min(corrected, introduced), n, 0.5).pvalue if n else 1.0

        print(f"Repeat {r+1} | Corrected={corrected} | "
              f"Introduced={introduced} | McNemar p={p:.4f}")


# ============================================================
# CONFUSION ANALYSIS
# ============================================================

print("\n"+"="*100)
print("MOST FREQUENT CONFUSIONS")
print("="*100)

for name in CONFIGS:

    confusions = Counter(
        (str(y_train[i]), str(OOF[name][r, i]))
        for r in range(N_REPEATS) for i in range(N)
        if OOF[name][r, i] != y_train[i]
    )

    print(f"\n{name}: {confusions.most_common(12)}")


# ============================================================
# PERSISTENT ERRORS
# ============================================================

print("\n"+"="*100)
print("PERSISTENT ERRORS")
print("="*100)

COUNTS = {
    name: np.sum(OOF[name] != y_train[None, :], axis=0)
    for name in CONFIGS
}

total = np.sum(np.stack(list(COUNTS.values())), axis=0)
order = np.argsort(-total, kind="stable")

for i in order[:15]:

    details = " | ".join(
        f"{name}={COUNTS[name][i]}/5" for name in CONFIGS
    )

    print(f"Index={i:4d} | Class={y_train[i]} | Total={total[i]}/25 | {details}")


# ============================================================
# HARD-CASE GEOMETRY
# ============================================================

print("\n"+"="*100)
print("HARD-CASE GEOMETRY — TRAIN-ONLY NEIGHBORS")
print("="*100)

for target in TARGET_IDS:

    print(f"\nIndex={target} | True class={y_train[target]}")

    for sigma in (1.5, 0.5, 0.0):

        records = sorted(
            [d for d in DIAGNOSTICS if d["index"] == target and d["sigma"] == sigma],
            key=lambda d: d["repeat"]
        )

        predictions = Counter(d["pred"] for d in records)
        rival_classes = Counter(d["other_class"] for d in records)

        ratios = [
            d["d_other"]/max(d["d_same"], 1e-12)
            for d in records
        ]

        print(f"Sigma={sigma:.2f} | Predictions={dict(predictions)} | "
              f"Nearest rival classes={dict(rival_classes)} | "
              f"Median rival/true distance={np.median(ratios):.3f}")

        for d in records:
            print(f"  R{d['repeat']+1} | Pred={d['pred']:>2s} | "
                  f"Same=#{d['same_id']:4d} ({d['d_same']:.3f}) | "
                  f"Rival=#{d['other_id']:4d}, class {d['other_class']:>2s} "
                  f"({d['d_other']:.3f})")


# ============================================================
# GEOMETRIC VISUALIZATIONS
# ============================================================

def reconstruct_path(case):

    v = np.nan_to_num(np.asarray(case, dtype=np.float64)[:2],
                      nan=0.0, posinf=0.0, neginf=0.0)

    xy = np.cumsum(v.T, axis=0)
    xy -= xy[0]

    length = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))

    if length > 1e-12:
        xy /= length

    return xy


def draw_path(ax, index, title):

    xy = reconstruct_path(X_train[index])

    ax.plot(xy[:, 0], xy[:, 1], linewidth=1.3)
    ax.scatter(xy[0, 0], xy[0, 1], s=25, marker="o")
    ax.scatter(xy[-1, 0], xy[-1, 1], s=25, marker="x")

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


if PLOT_DIAGNOSTICS:

    # Trajectory versus nearest true-class and rival-class neighbors.
    # Neighbors originate exclusively from the first outer training fold.

    fig, axes = plt.subplots(len(TARGET_IDS), 3, figsize=(12, 9))

    for row, target in enumerate(TARGET_IDS):

        record = next(d for d in DIAGNOSTICS
                      if d["index"] == target
                      and d["sigma"] == DIAG_SIGMA
                      and d["repeat"] == 0)

        draw_path(axes[row, 0], target,
                  f"Target #{target} | Class {y_train[target]} | Pred {record['pred']}")

        draw_path(axes[row, 1], record["same_id"],
                  f"Same class #{record['same_id']} | D={record['d_same']:.3f}")

        draw_path(axes[row, 2], record["other_id"],
                  f"Rival #{record['other_id']} | Class {record['other_class']} | "
                  f"D={record['d_other']:.3f}")

    fig.suptitle(f"Hard-case geometry — sigma={DIAG_SIGMA}", fontsize=13)
    fig.tight_layout()
    plt.show()

    # Angular turning profiles at different smoothing scales.

    fig, axes = plt.subplots(1, len(TARGET_IDS), figsize=(15, 3.8))

    for ax, target in zip(axes, TARGET_IDS):

        theta_arc = THETA_TARGETS[target]

        for sigma in (1.5, 0.5, 0.0):
            ax.plot(GRID, curvature_signal(theta_arc, sigma),
                    linewidth=1.0, label=f"Sigma={sigma:g}")

        ax.set_title(f"Index {target} | Class {y_train[target]}")
        ax.set_xlabel("Normalized arc length")
        ax.set_ylabel("Normalized turning rate")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)

    axes[-1].legend(fontsize=9)
    fig.tight_layout()
    plt.show()


# ============================================================
# REPRODUCIBILITY CONTROL
# ============================================================

print("\n"+"="*100)
print("REPRODUCIBILITY")
print("="*100)

expected = {
    "Sigma-1.50": [10, 12, 6, 10, 10],
    "Sigma-0.75": [6, 10, 6, 10, 10],
    "Sigma-0.50": [3, 7, 7, 8, 9]
}

for name, errors in expected.items():
    status = "OK" if RESULTS[name]["errors"] == errors else "DIFFERENT"
    print(f"{name}: {status}")
