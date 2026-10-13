# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from aeon.datasets import load_classification
from scipy.fft import dct
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


# ================================================================
# 1. CONFIGURATION
# ================================================================

SEED = 42
K, H, N_TIME = 24, 2, 241
F_ANGLE, F_SPEED = 8, 16
WEIGHT_SPEED = 0.25

F_SPATIAL = (4, 8, 16)
SPATIAL_MODES = ("bbox", "path_length", "center_only")
SPATIAL_WEIGHTS = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)

N_SCREEN, TOP_SCREEN = 650, 8

FULL_C = (1.0, 10.0, 100.0, 1000.0)
FULL_GAMMA = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)

RUN_OFFICIAL_TEST = True


# ================================================================
# 2. DATASET
# ================================================================

X_train, y_train = load_classification("CharacterTrajectories", split="train", load_equal_length=False)
X_test, y_test = load_classification("CharacterTrajectories", split="test", load_equal_length=False)

cases = list(X_train) + list(X_test)

y_train, y_test = np.asarray(y_train), np.asarray(y_test)
n_train, n_test = len(y_train), len(y_test)
N = len(cases)

time_grid = np.linspace(0, 1, N_TIME)

print(f"Train={n_train} | Test={n_test} | Classes={len(np.unique(y_train))}")


# ================================================================
# 3. ANGULAR AND SPEED REPRESENTATIONS
# ================================================================

def extract_signals(case):

    a = np.nan_to_num(np.asarray(case, dtype=np.float64)[:2], nan=0., posinf=0., neginf=0.)
    vx, vy = a

    speed = np.hypot(vx, vy)
    positive = speed[speed > 1e-10]
    threshold = 0.03*np.median(positive) if len(positive) else 0.
    valid = speed > max(threshold, 1e-10)

    theta = np.arctan2(vy, vx)

    if valid.any():
        previous = np.where(valid, np.arange(len(speed)), -1)
        np.maximum.accumulate(previous, out=previous)
        previous[previous < 0] = np.flatnonzero(valid)[0]
        theta = theta[previous]
    else:
        theta = np.zeros_like(theta)

    idx = np.minimum((time_grid*len(speed)).astype(int), len(speed)-1)

    theta = theta[idx]
    speed_sampled = speed[idx]
    relative_speed = speed_sampled/max(float(speed_sampled.mean()), 1e-8)

    symbols = np.floor((theta+np.pi/K)/(2*np.pi/K)).astype(int) % K
    angular_positions = 2*np.pi*symbols/K

    Z = np.stack([
        fn(h*angular_positions)*np.sqrt(2.0/K)
        for h in range(1, H+1)
        for fn in (np.cos, np.sin)
    ], axis=1)

    return Z.astype(np.float32), relative_speed.astype(np.float32)


def dct_features(X, nf, skip_dc=False):

    coeff = dct(X, type=2, axis=1, norm="ortho")
    start = 1 if skip_dc else 0

    return coeff[:, start:start+nf, :].reshape(len(X), -1).astype(np.float32)


angular, velocity = [], []

for case in cases:
    z, v = extract_signals(case)
    angular.append(z)
    velocity.append(v)

angular = np.stack(angular)
velocity = np.stack(velocity)

ANGLE = dct_features(angular, F_ANGLE)
SPEED = dct_features(velocity[:, :, None], F_SPEED)

print(f"Reference HASTE: {ANGLE.shape[1]} + {SPEED.shape[1]} = 48 coefficients")


# ================================================================
# 4. RECONSTRUCT SPATIAL TRAJECTORIES
# ================================================================

def reconstruct_position(case):

    v = np.nan_to_num(np.asarray(case, dtype=np.float64)[:2].T,
                      nan=0., posinf=0., neginf=0.)

    # Numerical integration over normalized time
    position = np.cumsum(v, axis=0)/len(v)

    # Interpolate to the common observation grid
    t = np.linspace(0, 1, len(position))
    xy = np.column_stack([
        np.interp(time_grid, t, position[:, 0]),
        np.interp(time_grid, t, position[:, 1])
    ])

    # Translation invariance
    xy -= xy.mean(axis=0, keepdims=True)

    return xy


positions = np.stack([reconstruct_position(case) for case in cases])


# ================================================================
# 5. SPATIAL NORMALIZATIONS
# ================================================================

def normalize_positions(X, mode):

    X = X.copy()

    if mode == "bbox":
        width = np.ptp(X, axis=1)
        scale = np.maximum(width.max(axis=1), 1e-8)

    elif mode == "path_length":
        increments = np.diff(X, axis=1)
        scale = np.maximum(np.linalg.norm(increments, axis=2).sum(axis=1), 1e-8)

    elif mode == "center_only":
        scale = np.ones(len(X))

    else:
        raise ValueError(mode)

    return (X/scale[:, None, None]).astype(np.float32)


SPATIAL = {}

for mode in SPATIAL_MODES:

    X = normalize_positions(positions, mode)

    for nf in F_SPATIAL:
        SPATIAL[(mode, nf)] = dct_features(X, nf, skip_dc=True)

print("Spatial dimensions:", {str(k): v.shape[1] for k, v in SPATIAL.items()})


# ================================================================
# 6. PRECOMPUTE SQUARED DISTANCES
# ================================================================

FEATURES = {"angle": ANGLE, "speed": SPEED}

for key, value in SPATIAL.items():
    FEATURES[key] = value

DISTANCES = {
    key: squareform(pdist(F[:n_train], metric="sqeuclidean")).astype(np.float32)
    for key, F in FEATURES.items()
}


def normalized_fold_distance(D, it, iv):

    A = D[np.ix_(it, it)]
    B = D[np.ix_(iv, it)]

    scale = max(float(A.sum())/(len(it)*(len(it)-1)), 1e-12)

    return A/scale, B/scale


# ================================================================
# 7. FOLD-SPECIFIC KERNEL COMPONENTS
# ================================================================

def prepare_folds(ids, n_splits, seed, spatial_key=None):

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for local_it, local_iv in cv.split(np.zeros(len(ids)), y_train[ids]):

        it, iv = ids[local_it], ids[local_iv]

        Aa, Ba = normalized_fold_distance(DISTANCES["angle"], it, iv)
        Ar, Br = normalized_fold_distance(DISTANCES["speed"], it, iv)

        if spatial_key is not None:
            Ap, Bp = normalized_fold_distance(DISTANCES[spatial_key], it, iv)
        else:
            Ap, Bp = None, None

        folds.append((it, iv, Aa, Ba, Ar, Br, Ap, Bp))

    return folds


# ================================================================
# 8. SVM CROSS-VALIDATION
# ================================================================

def optimize_folds(folds, spatial_weight, C_values, gamma_values):

    prepared = []

    for it, iv, Aa, Ba, Ar, Br, Ap, Bp in folds:

        denominator = 1+WEIGHT_SPEED+spatial_weight

        A = Aa+WEIGHT_SPEED*Ar
        B = Ba+WEIGHT_SPEED*Br

        if spatial_weight > 0:
            A = A+spatial_weight*Ap
            B = B+spatial_weight*Bp

        prepared.append((it, iv, A/denominator, B/denominator))

    best = (-np.inf, None, None)

    for gamma in gamma_values:

        kernels = [
            (it, iv, np.exp(-gamma*A), np.exp(-gamma*B))
            for it, iv, A, B in prepared
        ]

        for C in C_values:

            scores = []

            for it, iv, Ktr, Kval in kernels:

                clf = SVC(C=C, kernel="precomputed")
                clf.fit(Ktr, y_train[it])

                pred = clf.predict(Kval)
                scores.append(f1_score(y_train[iv], pred, average="macro", zero_division=0))

            score = float(np.mean(scores))

            if score > best[0]:
                best = (score, C, gamma)

    return best


# ================================================================
# 9. SCREENING SPATIAL CONFIGURATIONS
# ================================================================

screen_ids, _ = train_test_split(
    np.arange(n_train), train_size=N_SCREEN, stratify=y_train, random_state=SEED
)

screening = []

print("\n" + "="*85)
print("SPATIAL SCREENING")
print("="*85, flush=True)

for mode in SPATIAL_MODES:

    for nf in F_SPATIAL:

        key = (mode, nf)
        folds = prepare_folds(screen_ids, n_splits=3, seed=42, spatial_key=key)

        for wp in SPATIAL_WEIGHTS:

            score, _, _ = optimize_folds(folds, wp, C_values=(10.,), gamma_values=(1.,))
            screening.append((score, key, wp))

        local = max((r for r in screening if r[1] == key), key=lambda r: r[0])

        print(f"{mode:13s} | Spatial F={nf:2d} | Dim={48+2*nf:2d} | "
              f"Best weight={local[2]:g} | Screen F1={local[0]:.4f}", flush=True)


screening.sort(key=lambda x: x[0], reverse=True)

print("\nTOP SCREENING")
for row in screening[:12]:
    print(row)


# ================================================================
# 10. SELECT FINALISTS
# ================================================================

finalists = [(key, wp) for score, key, wp in screening[:TOP_SCREEN]]

for mode in SPATIAL_MODES:
    row = next(r for r in screening if r[1][0] == mode)
    finalists.append((row[1], row[2]))

for nf in F_SPATIAL:
    row = next(r for r in screening if r[1][1] == nf)
    finalists.append((row[1], row[2]))

finalists = list(dict.fromkeys(finalists))

print(f"\nFinalists: {len(finalists)}")


# ================================================================
# 11. FIVE-FOLD CROSS-VALIDATION
# ================================================================

train_ids = np.arange(n_train)
base_folds = prepare_folds(train_ids, n_splits=5, seed=123)

reference_cv = optimize_folds(base_folds, 0.0, FULL_C, FULL_GAMMA)

print("\n" + "="*85)
print("FULL TRAINING CROSS-VALIDATION")
print("="*85)

print(f"HASTE 48D | F1={reference_cv[0]:.4f} | "
      f"C={reference_cv[1]:g} | gamma={reference_cv[2]:g}", flush=True)


final_results = []

for key, wp in finalists:

    folds = prepare_folds(train_ids, n_splits=5, seed=123, spatial_key=key)

    score, C, gamma = optimize_folds(folds, wp, FULL_C, FULL_GAMMA)
    dimension = 48+SPATIAL[key].shape[1]

    final_results.append((score, key, wp, C, gamma, dimension))

    print(f"{str(key):25s} | Dim={dimension:2d} | wp={wp:g} | "
          f"F1={score:.4f} | C={C:g} | gamma={gamma:g}", flush=True)


final_results.sort(key=lambda r: r[0], reverse=True)

print("\nTOP SPATIAL MODELS")
for row in final_results[:10]:
    print(row)


# ================================================================
# 12. SELECT BEST SPATIAL MODEL
# ================================================================

best_f1, best_key, best_wp, best_C, best_gamma, best_dim = final_results[0]

print("\n" + "="*85)
print("BEST SPATIAL CONFIGURATION")
print("="*85)

print(f"Spatial normalization: {best_key[0]}")
print(f"Spatial DCT frequencies: {best_key[1]}")
print(f"Spatial weight: {best_wp:g}")
print(f"Dimension: {best_dim}")

print(f"HASTE CV F1:   {reference_cv[0]:.4f}")
print(f"HASTE-S CV F1: {best_f1:.4f}")
print(f"Difference:    {best_f1-reference_cv[0]:+.4f}")


# ================================================================
# 13. OFFICIAL TEST EVALUATION
# ================================================================

def test_model(spatial_key, wp, C, gamma):

    n = n_train
    total = 1+WEIGHT_SPEED+wp

    A = np.zeros((n, n), dtype=np.float32)
    B = np.zeros((n_test, n), dtype=np.float32)

    components = [("angle", 1.0), ("speed", WEIGHT_SPEED)]

    if spatial_key is not None and wp > 0:
        components.append((spatial_key, wp))

    for key, weight in components:

        D = DISTANCES[key]
        F = FEATURES[key]

        scale = max(float(D.sum())/(n*(n-1)), 1e-12)

        A += weight*D/scale
        B += weight*cdist(F[n:], F[:n], metric="sqeuclidean").astype(np.float32)/scale

    A /= total
    B /= total

    clf = SVC(C=C, kernel="precomputed")
    clf.fit(np.exp(-gamma*A), y_train)

    return clf.predict(np.exp(-gamma*B))


if RUN_OFFICIAL_TEST:

    # Historical model with its previously selected parameters
    pred_reference = test_model(None, 0.0, C=10.0, gamma=1.0)

    # New model selected exclusively from training CV
    pred_spatial = test_model(best_key, best_wp, best_C, best_gamma)

    reference_acc = accuracy_score(y_test, pred_reference)
    spatial_acc = accuracy_score(y_test, pred_spatial)

    reference_f1 = f1_score(y_test, pred_reference, average="macro", zero_division=0)
    spatial_f1 = f1_score(y_test, pred_spatial, average="macro", zero_division=0)

    reference_errors = set(np.flatnonzero(pred_reference != y_test))
    spatial_errors = set(np.flatnonzero(pred_spatial != y_test))

    corrected = sorted(reference_errors-spatial_errors)
    new_errors = sorted(spatial_errors-reference_errors)

    discordant = len(corrected)+len(new_errors)
    p_value = binomtest(min(len(corrected), len(new_errors)), discordant, 0.5).pvalue if discordant else 1.0

    print("\n" + "="*85)
    print("OFFICIAL TEST — HASTE VS HASTE-S")
    print("="*85)

    print(f"HASTE   | Dim=48        | Accuracy={reference_acc:.4f} | Macro-F1={reference_f1:.4f} | Errors={len(reference_errors)}")
    print(f"HASTE-S | Dim={best_dim:2d}        | Accuracy={spatial_acc:.4f} | Macro-F1={spatial_f1:.4f} | Errors={len(spatial_errors)}")

    print("\nCorrected errors:", corrected)
    print("New errors:", new_errors)
    print(f"Exploratory McNemar p-value: {p_value:.4f}")

    if len(reference_errors) != 11:
        print("WARNING: Historical HASTE model did not reproduce the previous 11 errors.")


# ================================================================
# 14. FIGURE — SPATIAL CONTRIBUTION
# ================================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

for mode in SPATIAL_MODES:

    rows = [r for r in final_results if r[1][0] == mode]
    rows.sort(key=lambda r: r[5])

    if rows:
        ax.scatter([r[5] for r in rows], [r[0] for r in rows],
                   s=30, label=mode)

ax.axhline(reference_cv[0], color="gray", linestyle="--",
           linewidth=1, label="HASTE 48D")

ax.set(xlabel="Representation dimension", ylabel="5-fold CV Macro-F1")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)
