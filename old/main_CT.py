# -*- coding: utf-8 -*-


import time
import numpy as np
import matplotlib.pyplot as plt

from aeon.datasets import load_classification
from scipy.fft import dct
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


# ================================================================
# 1. EXPERIMENTAL CONFIGURATION
# ================================================================

SEED = 42
T, DT = 12.0, 0.05
N_TIME = int(round(T/DT))+1

K_VALUES = (8, 12, 16, 24, 32)
H_VALUES = (1, 2, 3, 4)
F_VALUES = (8, 16, 32, 64)
ALPHAS = (-1.0, -0.5, 0.0, 0.5, 1.0)

C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
GAMMA_GRID = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

N_SCREEN = 650
TOP_SCREEN = 5

# Smooth monotone temporal perturbations
AUGMENTATIONS = {"none": (), "mild": (-0.04, 0.04), "moderate": (-0.08, 0.08)}

rng = np.random.default_rng(SEED)
clock = time.perf_counter()


# ================================================================
# 2. DATASET
# ================================================================

X_train, y_train = load_classification("CharacterTrajectories", split="train", load_equal_length=False)
X_test, y_test = load_classification("CharacterTrajectories", split="test", load_equal_length=False)

cases = list(X_train)+list(X_test)
y_train, y_test = np.asarray(y_train), np.asarray(y_test)

n_train, n_test = len(X_train), len(X_test)
n_total = len(cases)

print(f"Train: {n_train} | Test: {n_test} | Classes: {len(np.unique(y_train))}", flush=True)


# ================================================================
# 3. ANGULAR SIGNALS
# ================================================================

def continuous_angles(case):

    vx, vy = np.asarray(case, dtype=float)[:2]
    speed = np.hypot(vx, vy)

    positive = speed[speed > 1e-10]
    threshold = 0.03*np.median(positive) if len(positive) else 0.0
    valid = speed > max(threshold, 1e-10)

    angles = np.arctan2(vy, vx)

    if valid.any():
        previous = np.where(valid, np.arange(len(speed)), -1)
        np.maximum.accumulate(previous, out=previous)
        previous[previous < 0] = np.flatnonzero(valid)[0]
        angles = angles[previous]
    else:
        angles = np.zeros_like(angles)

    return angles


angles = [continuous_angles(x) for x in cases]
time_grid = np.linspace(0, 1, N_TIME)


def make_symbol_grid(K, ids=None, warp=0.0):

    if ids is None:
        ids = np.arange(n_total)

    u = time_grid+warp*np.sin(2*np.pi*time_grid)
    result = np.empty((len(ids), N_TIME), dtype=np.int16)

    for row, i in enumerate(ids):

        a = angles[i]
        symbols = np.floor((a+np.pi/K)/(2*np.pi/K)).astype(np.int16) % K
        indices = np.clip((u*len(symbols)).astype(int), 0, len(symbols)-1)

        result[row] = symbols[indices]

    return result


symbol_grids = {K: make_symbol_grid(K) for K in K_VALUES}


# ================================================================
# 4. ANGULAR HARMONIC EMBEDDING
# ================================================================

def harmonic_coordinates(symbols, K, H):

    theta = (2*np.pi/K)*symbols.astype(np.float32)
    coordinates = []

    for h in range(1, H+1):

        if K % 2 == 0 and h == K//2:
            coordinates.append(np.cos(h*theta)/np.sqrt(K))
        else:
            factor = np.sqrt(2.0/K)
            coordinates.append(factor*np.cos(h*theta))
            coordinates.append(factor*np.sin(h*theta))

    return np.stack(coordinates, axis=2).astype(np.float32)


def dct_coefficients(symbols, K, H):

    X = harmonic_coordinates(symbols, K, H)
    return dct(X, type=2, axis=1, norm="ortho")


def spectral_features(coefficients, F, alpha):

    weights = (1+np.arange(F, dtype=np.float32))**alpha
    weighted = coefficients[:, :F, :]*weights[None, :, None]

    return weighted.reshape(len(coefficients), -1).astype(np.float32)


# ================================================================
# 5. RBF SVM AND CROSS-VALIDATION
# ================================================================

def make_splits(y):

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))


full_splits = make_splits(y_train)


def normalized_distances(F, it, iv):

    A = squareform(pdist(F[it], metric="sqeuclidean")).astype(np.float32)
    B = cdist(F[iv], F[it], metric="sqeuclidean").astype(np.float32)

    scale = max(float(A.sum())/(len(it)*(len(it)-1)), 1e-12)
    return A/scale, B/scale


def tune_svm(F, y, splits, C_values, gamma_values):

    distances = [(it, iv, *normalized_distances(F, it, iv)) for it, iv in splits]
    best = (-np.inf, None, None)

    for gamma in gamma_values:

        kernels = [(it, iv, np.exp(-gamma*A), np.exp(-gamma*B))
                   for it, iv, A, B in distances]

        for C in C_values:

            scores = []

            for it, iv, A, B in kernels:

                clf = SVC(C=C, kernel="precomputed")
                clf.fit(A, y[it])

                pred = clf.predict(B)
                scores.append(f1_score(y[iv], pred, average="macro", zero_division=0))

            score = float(np.mean(scores))

            if score > best[0]:
                best = (score, C, gamma)

    return best


# ================================================================
# 6. BASELINE — EXPANDED SVM SEARCH
# ================================================================

print("\n" + "="*65)
print("BASELINE — K=8, H=1, F=16, alpha=0")
print("="*65, flush=True)

base_coefficients = dct_coefficients(symbol_grids[8][:n_train], 8, 1)
base_features = spectral_features(base_coefficients, 16, 0.0)

base_score, base_C, base_gamma = tune_svm(
    base_features, y_train, full_splits, C_GRID, GAMMA_GRID
)

print(f"CV Macro-F1={base_score:.4f} | C={base_C:g} | gamma={base_gamma:g}", flush=True)


# ================================================================
# 7. SCREENING — K, H, F AND ALPHA
# ================================================================

screen_ids, _ = train_test_split(
    np.arange(n_train), train_size=N_SCREEN, stratify=y_train, random_state=SEED
)

screen_y = y_train[screen_ids]
screen_splits = make_splits(screen_y)

screening = []

print("\n" + "="*65)
print("SCREENING — ANGULAR HARMONICS AND WEIGHTED DCT")
print("="*65, flush=True)

for K in K_VALUES:

    symbols = symbol_grids[K][screen_ids]

    for H in H_VALUES:

        coefficients = dct_coefficients(symbols, K, H)
        local_best = (-np.inf, None)

        for F in F_VALUES:
            for alpha in ALPHAS:

                features = spectral_features(coefficients, F, alpha)

                score, _, _ = tune_svm(
                    features, screen_y, screen_splits, (base_C,), (base_gamma,)
                )

                screening.append((score, K, H, F, alpha))

                if score > local_best[0]:
                    local_best = (score, (F, alpha, features.shape[1]))

        score, (F, alpha, dim) = local_best

        print(f"K={K:2d} | H={H} | Best F={F:2d} | alpha={alpha:+.1f} | "
              f"Dim={dim:3d} | CV F1={score:.4f}", flush=True)

screening.sort(reverse=True)

print("\nTOP SCREENING")
for row in screening[:12]:
    print(row)


# ================================================================
# 8. SELECT FINALISTS
# ================================================================

finalists = {(K, H, F, alpha) for _, K, H, F, alpha in screening[:TOP_SCREEN]}

# Ensure the best candidate of each quantization resolution
for K in K_VALUES:
    row = next(r for r in screening if r[1] == K)
    finalists.add(tuple(row[1:]))

# Ensure the best candidate of each harmonic order
for H in H_VALUES:
    row = next(r for r in screening if r[2] == H)
    finalists.add(tuple(row[1:]))

# Keep the previous simple representation
finalists.add((8, 1, 16, 0.0))
finalists = sorted(finalists)

print(f"\nFull-training finalists: {len(finalists)}")


# ================================================================
# 9. FULL TRAINING CROSS-VALIDATION
# ================================================================

print("\n" + "="*75)
print("FULL TRAINING CROSS-VALIDATION")
print("="*75, flush=True)

full_results = []

for K, H, F, alpha in finalists:

    coefficients = dct_coefficients(symbol_grids[K][:n_train], K, H)
    features = spectral_features(coefficients, F, alpha)

    score, C, gamma = tune_svm(
        features, y_train, full_splits, C_GRID, GAMMA_GRID
    )

    full_results.append((score, K, H, F, alpha, C, gamma, features.shape[1]))

    print(f"K={K:2d} | H={H} | F={F:2d} | alpha={alpha:+.1f} | "
          f"Dim={features.shape[1]:3d} | F1={score:.4f} | "
          f"C={C:g} | gamma={gamma:g}", flush=True)

full_results.sort(reverse=True)

print("\nTOP FULL TRAINING CV")
for row in full_results[:12]:
    print(row)


# ================================================================
# 10. BEST CONFIGURATION
# ================================================================

cv_f1, K_best, H_best, F_best, alpha_best, C_best, gamma_best, dim_best = full_results[0]

print("\n" + "="*75)
print("SELECTED REPRESENTATION")
print("="*75)

print(f"K={K_best} | H={H_best} | F={F_best} | alpha={alpha_best:+.1f}")
print(f"Dimension={dim_best} | C={C_best:g} | gamma={gamma_best:g}")
print(f"CV Macro-F1={cv_f1:.4f}")


# ================================================================
# 11. TEMPORAL AUGMENTATION — TRAINING ONLY
# ================================================================

coefficients = dct_coefficients(symbol_grids[K_best], K_best, H_best)
F_all = spectral_features(coefficients, F_best, alpha_best)

F_train, F_test = F_all[:n_train], F_all[n_train:]

augmented = {}

for amplitude in sorted({a for values in AUGMENTATIONS.values() for a in values}):

    grid = make_symbol_grid(K_best, ids=np.arange(n_train), warp=amplitude)
    coeffs = dct_coefficients(grid, K_best, H_best)
    augmented[amplitude] = spectral_features(coeffs, F_best, alpha_best)


def tune_augmentation(amplitudes, C_values, gamma_values):

    fold_data = []

    for it, iv in full_splits:

        Xtr = np.concatenate([F_train[it]]+[augmented[a][it] for a in amplitudes], axis=0)
        Ytr = np.tile(y_train[it], len(amplitudes)+1)

        A = squareform(pdist(Xtr, metric="sqeuclidean")).astype(np.float32)
        B = cdist(F_train[iv], Xtr, metric="sqeuclidean").astype(np.float32)

        scale = max(float(A.sum())/(len(Xtr)*(len(Xtr)-1)), 1e-12)
        fold_data.append((Ytr, y_train[iv], A/scale, B/scale))

    best = (-np.inf, None, None)

    for gamma in gamma_values:

        kernels = [(Ytr, Yval, np.exp(-gamma*A), np.exp(-gamma*B))
                   for Ytr, Yval, A, B in fold_data]

        for C in C_values:

            scores = []

            for Ytr, Yval, A, B in kernels:
                clf = SVC(C=C, kernel="precomputed").fit(A, Ytr)
                scores.append(f1_score(Yval, clf.predict(B), average="macro", zero_division=0))

            score = float(np.mean(scores))

            if score > best[0]:
                best = (score, C, gamma)

    return best


aug_results = [(cv_f1, "none", C_best, gamma_best)]

print("\n" + "="*75)
print("TEMPORAL AUGMENTATION")
print("="*75, flush=True)

for name in ("mild", "moderate"):

    amplitudes = AUGMENTATIONS[name]

    score, C, gamma = tune_augmentation(
        amplitudes,
        (C_best, 10*C_best),
        (gamma_best/2, gamma_best, 2*gamma_best)
    )

    aug_results.append((score, name, C, gamma))

    print(f"{name:8s} | CV Macro-F1={score:.4f} | C={C:g} | gamma={gamma:g}", flush=True)

aug_results.sort(reverse=True)
best_aug_score, best_aug, C_final, gamma_final = aug_results[0]

print(f"\nSelected augmentation: {best_aug} | CV Macro-F1={best_aug_score:.4f}")


# ================================================================
# 12. OFFICIAL TEST — SINGLE FINAL EVALUATION
# ================================================================

amplitudes = AUGMENTATIONS[best_aug]

Xtr = np.concatenate([F_train]+[augmented[a] for a in amplitudes], axis=0)
Ytr = np.tile(y_train, len(amplitudes)+1)

A = squareform(pdist(Xtr, metric="sqeuclidean")).astype(np.float32)
B = cdist(F_test, Xtr, metric="sqeuclidean").astype(np.float32)

scale = max(float(A.sum())/(len(Xtr)*(len(Xtr)-1)), 1e-12)

clf = SVC(C=C_final, kernel="precomputed")
clf.fit(np.exp(-gamma_final*A/scale), Ytr)

pred = clf.predict(np.exp(-gamma_final*B/scale))

accuracy = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)

print("\n" + "="*75)
print("OFFICIAL TEST")
print("="*75)

print(f"K={K_best} | H={H_best} | F={F_best} | alpha={alpha_best:+.1f}")
print(f"Dimension={dim_best} | Augmentation={best_aug}")
print(f"Accuracy={accuracy:.4f} | Macro-F1={macro_f1:.4f}")
print(f"Total runtime: {(time.perf_counter()-clock)/60:.1f} min")


# ================================================================
# 13. FIGURE — QUANTIZATION VS HARMONIC ORDER
# ================================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

for H in H_VALUES:

    scores = [max(r[0] for r in screening if r[1] == K and r[2] == H)
              for K in K_VALUES]

    ax.plot(K_VALUES, scores, marker="o", markersize=3.5,
            linewidth=1.3, label=f"H={H}")

ax.set(xlabel="Number of angular symbols", ylabel="Screening CV Macro-F1",
       xticks=K_VALUES)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)


# ================================================================
# 14. FIGURE — DIMENSION VS CLASSIFICATION PERFORMANCE
# ================================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

for H in H_VALUES:

    rows = sorted([r for r in full_results if r[2] == H], key=lambda r: r[7])

    if rows:
        ax.plot([r[7] for r in rows], [r[0] for r in rows],
                marker="o", markersize=3.5, linewidth=1.2, label=f"H={H}")

ax.set(xlabel="Representation dimension", ylabel="Full training CV Macro-F1")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)




# ================================================================
# QUANTIZED VS CONTINUOUS ANGULAR REPRESENTATIONS
# ================================================================

from scipy.stats import binomtest

K = 24

CONFIGS = [
    (1, 8, 0.0),       # 16 coefficients
    (2, 8, 0.0),       # 32 coefficients - compact
    (2, 16, 0.0),      # 64 coefficients
    (4, 16, -0.5)      # 128 coefficients - previous best
]

# Continuous angles sampled on exactly the same temporal grid
continuous_grid = np.stack([
    a[np.minimum((time_grid*len(a)).astype(int), len(a)-1)]
    for a in angles
]).astype(np.float32)


def continuous_coordinates(theta, K, H):

    factor = np.sqrt(2.0/K)

    return np.stack([
        f(h*theta)*factor
        for h in range(1, H+1)
        for f in (np.cos, np.sin)
    ], axis=2).astype(np.float32)


def make_features(source, H, F, alpha):

    if source == "quantized":
        coefficients = dct_coefficients(symbol_grids[K], K, H)

    elif source == "continuous":
        X = continuous_coordinates(continuous_grid, K, H)
        coefficients = dct(X, type=2, axis=1, norm="ortho")

    else:
        raise ValueError(source)

    return spectral_features(coefficients, F, alpha)


def evaluate_configuration(F_all, C, gamma):

    Xtr, Xte = F_all[:n_train], F_all[n_train:]

    A = squareform(pdist(Xtr, metric="sqeuclidean"))
    B = cdist(Xte, Xtr, metric="sqeuclidean")

    scale = max(A.sum()/(n_train*(n_train-1)), 1e-12)

    clf = SVC(C=C, kernel="precomputed")
    clf.fit(np.exp(-gamma*A/scale), y_train)

    pred = clf.predict(np.exp(-gamma*B/scale))

    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro"), pred


# ================================================================
# 1. FULL TRAINING CROSS-VALIDATION
# ================================================================

results = {}
predictions = {}

print("\n" + "="*85)
print("QUANTIZED VS CONTINUOUS — CROSS-VALIDATION")
print("="*85)

for source in ("quantized", "continuous"):

    for H, F, alpha in CONFIGS:

        features = make_features(source, H, F, alpha)

        score, C, gamma = tune_svm(
            features[:n_train], y_train, full_splits, C_GRID, GAMMA_GRID
        )

        results[source, H, F, alpha] = (score, C, gamma, features.shape[1])

        print(f"{source:11s} | H={H} | F={F:2d} | alpha={alpha:+.1f} | "
              f"Dim={features.shape[1]:3d} | CV F1={score:.4f} | "
              f"C={C:g} | gamma={gamma:g}", flush=True)


# ================================================================
# 2. TEST — COMPACT 32D AND EXTENDED 128D
# ================================================================

print("\n" + "="*85)
print("OFFICIAL TEST — MATCHED REPRESENTATIONS")
print("="*85)

test_rows = []

for H, F, alpha in ((2, 8, 0.0), (4, 16, -0.5)):

    for source in ("quantized", "continuous"):

        score, C, gamma, dim = results[source, H, F, alpha]
        features = make_features(source, H, F, alpha)

        acc, f1, pred = evaluate_configuration(features, C, gamma)
        predictions[source, H, F, alpha] = pred

        test_rows.append((source, H, F, dim, score, acc, f1))

        print(f"{source:11s} | Dim={dim:3d} | CV F1={score:.4f} | "
              f"Test Acc={acc:.4f} | Test F1={f1:.4f}", flush=True)


# ================================================================
# 3. PAIRED COMPARISON OF CLASSIFICATION ERRORS
# ================================================================

print("\n" + "="*85)
print("PAIRED ERROR COMPARISON")
print("="*85)

for H, F, alpha in ((2, 8, 0.0), (4, 16, -0.5)):

    pred_q = predictions["quantized", H, F, alpha]
    pred_c = predictions["continuous", H, F, alpha]

    correct_q = pred_q == y_test
    correct_c = pred_c == y_test

    q_only = np.sum(correct_q & ~correct_c)
    c_only = np.sum(correct_c & ~correct_q)

    p_value = binomtest(int(min(q_only, c_only)), int(q_only+c_only), 0.5).pvalue if q_only+c_only else 1.0

    print(f"Dimension={2*H*F}")
    print(f"  Correct only with quantization: {q_only}")
    print(f"  Correct only with continuous angles: {c_only}")
    print(f"  Exact McNemar p-value: {p_value:.4f}")


# ================================================================
# 4. FIGURE — CROSS-VALIDATION RESULTS
# ================================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

for source in ("quantized", "continuous"):

    dimensions = []
    scores = []

    for H, F, alpha in CONFIGS:
        score, _, _, dim = results[source, H, F, alpha]
        dimensions.append(dim)
        scores.append(score)

    ax.plot(dimensions, scores, marker="o", markersize=3.5,
            linewidth=1.4, label=source.capitalize())

ax.set(xlabel="Representation dimension", ylabel="CV Macro-F1",
       xticks=[16, 32, 64, 128])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)


# ================================================================
# 5. FIGURE — OFFICIAL TEST ACCURACY
# ================================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

labels = [f"{source} ({dim}D)" for source, H, F, dim, cv, acc, f1 in test_rows]
accuracies = [100*row[5] for row in test_rows]

ax.barh(labels[::-1], accuracies[::-1])
ax.axvline(97.77, color="gray", linestyle="--", linewidth=1, label="Previous best")

ax.set(xlabel="Test accuracy (%)", xlim=(90, 100))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.2)
ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)

