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


# ================================================================
# 1. CONFIGURATION
# ================================================================

SEED = 2027

K, H, N_TIME = 24, 2, 241
F_TIME, F_SPEED, F_ARC, F_CURV = 8, 16, 4, 4

W_SPEED, W_ARC = 0.25, 4.0
CURV_SMOOTHING, TURNING_SCALE = 1.5, 0.20

C_VALUES = (10.0, 100.0)
GAMMA_VALUES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)

N_INNER, N_OUTER, N_REPEATS = 3, 5, 5

# Each configuration = (angular blending lambda, curvature weight)
CONFIGS = {
    "Baseline": (0.0, 0.0),
    "Curvature-Q": (0.0, 1.0),
    "Curvature-Mixed": (0.5, 2.0),
    "Curvature-Continuous": (1.0, 2.0)
}

METHODS = tuple(CONFIGS)
LAMBDAS = (0.0, 0.5, 1.0)

grid = np.linspace(0.0, 1.0, N_TIME)


# ================================================================
# 2. DATASET — OFFICIAL TRAIN SPLIT ONLY
# ================================================================

X_train, y_train = load_classification(
    "CharacterTrajectories", split="train", load_equal_length=False
)

cases = list(X_train)
y_train = np.asarray(y_train)
n_train = len(y_train)

print(f"Train={n_train} | Classes={len(np.unique(y_train))}")
print(f"Outer CV={N_REPEATS} x {N_OUTER} | Inner CV={N_INNER}")
print(f"Configurations={len(CONFIGS)}")


# ================================================================
# 3. ANGULAR REPRESENTATION
# ================================================================

def quantize_angles(theta):
    symbols = np.floor((theta+np.pi/K)/(2*np.pi/K)).astype(int) % K
    return 2*np.pi*symbols/K


def harmonic_embedding(theta):
    return np.stack([fn(h*theta)*np.sqrt(2.0/K)
                     for h in range(1, H+1)
                     for fn in (np.cos, np.sin)], axis=1)


# ================================================================
# 4. FEATURE EXTRACTION
# ================================================================

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

    # ------------------------------------------------------------
    # Temporal angular representation — 32D
    # ------------------------------------------------------------

    idx = np.minimum((grid*n).astype(int), n-1)
    z_time = harmonic_embedding(quantize_angles(theta[idx]))

    ft = dct(z_time, type=2, axis=0, norm="ortho")[:F_TIME].ravel()

    # ------------------------------------------------------------
    # Relative speed — 16D
    # ------------------------------------------------------------

    relative_speed = speed[idx]/max(float(speed[idx].mean()), 1e-8)
    fr = dct(relative_speed, type=2, norm="ortho")[:F_SPEED]

    # ------------------------------------------------------------
    # Arc-length parameterization
    # ------------------------------------------------------------

    ds = 0.5*(speed[:-1]+speed[1:])
    s = np.r_[0.0, np.cumsum(ds)]

    if s[-1] > 1e-12:
        targets = grid*s[-1]
        keep = np.r_[True, np.diff(s) > 1e-12]
        theta_arc = np.interp(targets, s[keep], np.unwrap(theta)[keep])
    else:
        theta_arc = theta[idx]

    # ------------------------------------------------------------
    # Quantized and continuous arc-length representations — 16D
    # ------------------------------------------------------------

    z_quantized = harmonic_embedding(quantize_angles(theta_arc))
    z_continuous = harmonic_embedding(theta_arc)

    arc_features = {}

    for lam in LAMBDAS:
        z = (1.0-lam)*z_quantized+lam*z_continuous
        arc_features[lam] = dct(z, type=2, axis=0, norm="ortho")[:F_ARC].ravel()

    # ------------------------------------------------------------
    # Regularized turning-rate representation — 4D
    # ------------------------------------------------------------

    theta_smooth = gaussian_filter1d(
        theta_arc, sigma=CURV_SMOOTHING, mode="nearest"
    )

    turning = np.gradient(theta_smooth, grid)
    turning = np.tanh(turning/(TURNING_SCALE*(N_TIME-1)))

    curvature = dct(turning, type=2, norm="ortho")[:F_CURV]

    return ft, fr, arc_features, curvature


# ================================================================
# 5. PRECOMPUTE ALL FEATURES
# ================================================================

print("\nExtracting features...", flush=True)

encoded = [encode_one(case) for case in cases]

FEATURES = {
    "time": np.stack([e[0] for e in encoded]).astype(np.float32),
    "speed": np.stack([e[1] for e in encoded]).astype(np.float32),
    "curvature": np.stack([e[3] for e in encoded]).astype(np.float32)
}

for lam in LAMBDAS:
    FEATURES[("arc", lam)] = np.stack(
        [e[2][lam] for e in encoded]
    ).astype(np.float32)

print("\nFEATURE DIMENSIONS")
for key, value in FEATURES.items():
    print(f"{str(key):20s} | Dim={value.shape[1]}")


# ================================================================
# 6. PRECOMPUTED SQUARED DISTANCES
# ================================================================

print("\nComputing pairwise distances...", flush=True)

DISTANCES = {
    key: squareform(pdist(X, metric="sqeuclidean")).astype(np.float32)
    for key, X in FEATURES.items()
}


def fold_blocks(it, iv):

    blocks = {}
    n = len(it)

    for key, D in DISTANCES.items():
        A = D[np.ix_(it, it)].astype(np.float64)
        B = D[np.ix_(iv, it)].astype(np.float64)

        scale = max(float(A.sum())/(n*(n-1)), 1e-12)
        blocks[key] = (A/scale, B/scale)

    return blocks


# ================================================================
# 7. MODEL DISTANCES
# ================================================================

def model_distances(blocks, config):

    lam, w_curv = config

    At, Bt = blocks["time"]
    Ar, Br = blocks["speed"]
    Aa, Ba = blocks[("arc", lam)]

    A = At+W_SPEED*Ar+W_ARC*Aa
    B = Bt+W_SPEED*Br+W_ARC*Ba

    denominator = 1.0+W_SPEED+W_ARC

    if w_curv > 0:
        Ac, Bc = blocks["curvature"]
        A = A+w_curv*Ac
        B = B+w_curv*Bc
        denominator += w_curv

    return np.maximum(A/denominator, 0.0), np.maximum(B/denominator, 0.0)


# ================================================================
# 8. INNER CV — ONLY C AND GAMMA ARE OPTIMIZED
# ================================================================

def tune_models(train_ids, seed):

    cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)

    prepared = []

    for a, b in cv.split(np.zeros(len(train_ids)), y_train[train_ids]):
        it, iv = train_ids[a], train_ids[b]
        prepared.append((it, iv, fold_blocks(it, iv)))

    selected = {}

    for name, config in CONFIGS.items():

        pairs = [
            (it, iv, *model_distances(blocks, config))
            for it, iv, blocks in prepared
        ]

        best = (-np.inf, None, None)

        for gamma in GAMMA_VALUES:

            kernels = [
                (it, iv, np.exp(-gamma*A), np.exp(-gamma*B))
                for it, iv, A, B in pairs
            ]

            for C in C_VALUES:

                scores = []

                for it, iv, Ktr, Kval in kernels:

                    model = SVC(C=C, kernel="precomputed")
                    model.fit(Ktr, y_train[it])

                    prediction = model.predict(Kval)

                    scores.append(
                        f1_score(y_train[iv], prediction,
                                 average="macro", zero_division=0)
                    )

                score = float(np.mean(scores))

                if score > best[0]:
                    best = (score, C, gamma)

        selected[name] = best

    return selected


# ================================================================
# 9. REPEATED NESTED CROSS-VALIDATION
# ================================================================

outer_cv = RepeatedStratifiedKFold(
    n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED
)

OOF = {
    name: np.empty((N_REPEATS, n_train), dtype=y_train.dtype)
    for name in METHODS
}

SELECTIONS = {name: [] for name in METHODS}

print("\n" + "="*95)
print("REPEATED NESTED CROSS-VALIDATION")
print("="*95, flush=True)

for iteration, (it, iv) in enumerate(
    outer_cv.split(np.zeros(n_train), y_train)
):

    repeat, fold = divmod(iteration, N_OUTER)

    selected = tune_models(it, seed=1000+iteration)
    blocks = fold_blocks(it, iv)

    print(f"\nRepeat {repeat+1}/{N_REPEATS} | Fold {fold+1}/{N_OUTER}", flush=True)

    for name in METHODS:

        inner_f1, C, gamma = selected[name]
        A, B = model_distances(blocks, CONFIGS[name])

        model = SVC(C=C, kernel="precomputed")
        model.fit(np.exp(-gamma*A), y_train[it])

        prediction = model.predict(np.exp(-gamma*B))

        OOF[name][repeat, iv] = prediction
        SELECTIONS[name].append((C, gamma))

        acc = accuracy_score(y_train[iv], prediction)
        f1 = f1_score(y_train[iv], prediction, average="macro", zero_division=0)

        print(f"{name:22s} | Acc={acc:.4f} | F1={f1:.4f} | "
              f"Inner={inner_f1:.4f} | C={C:g} | gamma={gamma:g}", flush=True)


# ================================================================
# 10. AGGREGATED RESULTS
# ================================================================

RESULTS = {}

print("\n" + "="*95)
print("AGGREGATED RESULTS — 5 x 5 NESTED CV")
print("="*95)

for name in METHODS:

    accuracies = np.array([
        accuracy_score(y_train, OOF[name][r])
        for r in range(N_REPEATS)
    ])

    f1_scores = np.array([
        f1_score(y_train, OOF[name][r], average="macro", zero_division=0)
        for r in range(N_REPEATS)
    ])

    errors = [
        int(np.sum(OOF[name][r] != y_train))
        for r in range(N_REPEATS)
    ]

    RESULTS[name] = {
        "accuracy": accuracies,
        "f1": f1_scores,
        "errors": errors
    }

    print(f"{name:22s} | Acc={accuracies.mean():.4f} ± {accuracies.std(ddof=1):.4f} | "
          f"F1={f1_scores.mean():.4f} ± {f1_scores.std(ddof=1):.4f} | Errors={errors}")


# ================================================================
# 11. PAIRED COMPARISONS
# ================================================================

print("\n" + "="*95)
print("PAIRED COMPARISONS")
print("="*95)


def paired_comparison(reference, alternative):

    print(f"\n{reference} vs {alternative}")

    for r in range(N_REPEATS):

        correct_ref = OOF[reference][r] == y_train
        correct_alt = OOF[alternative][r] == y_train

        ref_only = int(np.sum(correct_ref & ~correct_alt))
        alt_only = int(np.sum(~correct_ref & correct_alt))

        n = ref_only+alt_only
        p = binomtest(min(ref_only, alt_only), n, 0.5).pvalue if n else 1.0

        print(f"Repeat {r+1} | {reference} only={ref_only} | "
              f"{alternative} only={alt_only} | McNemar p={p:.4f}")


for alternative in METHODS[1:]:
    paired_comparison("Baseline", alternative)

paired_comparison("Curvature-Q", "Curvature-Continuous")
paired_comparison("Curvature-Mixed", "Curvature-Continuous")


# ================================================================
# 12. SVM HYPERPARAMETER FREQUENCIES
# ================================================================

print("\n" + "="*95)
print("SVM PARAMETER SELECTION")
print("="*95)

for name in METHODS:
    print(f"\n{name}")
    for parameters, count in Counter(SELECTIONS[name]).most_common():
        print(f"  C={parameters[0]:g} | gamma={parameters[1]:g} | {count}/{N_OUTER*N_REPEATS}")


# ================================================================
# 13. CORRECTED AND INTRODUCED ERRORS
# ================================================================

print("\n" + "="*95)
print("CLASS-WISE ERROR ANALYSIS")
print("="*95)

for alternative in METHODS[1:]:

    corrected, introduced = [], []

    for r in range(N_REPEATS):

        base_correct = OOF["Baseline"][r] == y_train
        alt_correct = OOF[alternative][r] == y_train

        corrected.extend(y_train[~base_correct & alt_correct].tolist())
        introduced.extend(y_train[base_correct & ~alt_correct].tolist())

    print(f"\n{alternative}")
    print("Corrected :", Counter(corrected).most_common())
    print("Introduced:", Counter(introduced).most_common())


# ================================================================
# 14. REPEAT-WISE ACCURACY FIGURE
# ================================================================

fig, ax = plt.subplots(figsize=(8, 4.3))

for name in METHODS:
    ax.plot(np.arange(1, N_REPEATS+1), RESULTS[name]["accuracy"],
            marker="o", markersize=3.5, linewidth=1.2, label=name)

ax.set_xlabel("CV repetition")
ax.set_ylabel("Out-of-fold accuracy")
ax.set_xticks(np.arange(1, N_REPEATS+1))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.show()
plt.close(fig)
