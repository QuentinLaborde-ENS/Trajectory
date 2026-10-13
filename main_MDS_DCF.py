# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from aeon.datasets import load_classification
from scipy.fft import dct
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


# ================================================================
# 1. DATA AND PARAMETERS
# ================================================================

SEED, K, H, F_ANGLE, F_SCALAR, F_LOCAL = 42, 24, 2, 8, 16, 4
N_TIME = 241

WEIGHTS = (0.25, 0.5, 1.0, 2.0)
COARSE_C, COARSE_GAMMA = (10., 100.), (0.25, 1.)
FULL_C = (1., 10., 100., 1000.)
FULL_GAMMA = (0.0625, 0.125, 0.25, 0.5, 1., 2., 4.)

EVALUATE_EACH_FAMILY_ON_TEST = False

X_train, y_train = load_classification("CharacterTrajectories", split="train", load_equal_length=False)
X_test, y_test = load_classification("CharacterTrajectories", split="test", load_equal_length=False)

cases = list(X_train) + list(X_test)
y_train, y_test = np.asarray(y_train), np.asarray(y_test)
n_train, n_test = len(X_train), len(X_test)

print(f"Train={n_train} | Test={n_test} | Classes={len(np.unique(y_train))}", flush=True)


# ================================================================
# 2. COMMON TIME GRID
# ================================================================

def original_signals(case):

    a = np.nan_to_num(np.asarray(case, dtype=np.float64), nan=0., posinf=0., neginf=0.)
    vx, vy, third = a[:3]
    speed = np.hypot(vx, vy)

    positive = speed[speed > 1e-10]
    threshold = 0.03*np.median(positive) if positive.size else 0.0
    valid = speed > max(threshold, 1e-10)

    theta = np.arctan2(vy, vx)

    if valid.any():
        previous = np.where(valid, np.arange(len(speed)), -1)
        np.maximum.accumulate(previous, out=previous)
        previous[previous < 0] = np.flatnonzero(valid)[0]
        theta = theta[previous]
    else:
        theta = np.zeros_like(theta)

    idx = np.minimum((np.linspace(0, 1, N_TIME)*len(speed)).astype(int), len(speed)-1)

    return theta[idx], speed[idx], third[idx]


signals = [original_signals(x) for x in cases]

theta = np.stack([s[0] for s in signals]).astype(np.float32)
speed = np.stack([s[1] for s in signals]).astype(np.float32)
third = np.stack([s[2] for s in signals]).astype(np.float32)


# ================================================================
# 3. HARMONIC SYMBOLIC EMBEDDING
# ================================================================

symbols = np.floor((theta+np.pi/K)/(2*np.pi/K)).astype(np.int16) % K
angle = (2*np.pi/K)*symbols

Z = np.stack([
    fn(h*angle)*np.sqrt(2./K)
    for h in range(1, H+1)
    for fn in (np.cos, np.sin)
], axis=2).astype(np.float32)


def dct_features(x, nf):
    return dct(x, type=2, axis=1, norm="ortho")[:, :nf, :].reshape(len(x), -1).astype(np.float32)


def scalar_dct(x, nf):
    return dct_features(x[:, :, None], nf)


# ================================================================
# 4. FEATURE BLOCKS
# ================================================================

mid = N_TIME//2

blocks = {
    "angle": dct_features(Z, F_ANGLE),

    "local": np.concatenate([
        dct_features(Z[:, :mid], F_LOCAL),
        dct_features(Z[:, mid:], F_LOCAL)
    ], axis=1),

    "speed_raw": scalar_dct(speed, F_SCALAR),
    "speed_log": scalar_dct(np.log1p(speed), F_SCALAR),
    "speed_relative": scalar_dct(speed/np.maximum(speed.mean(axis=1, keepdims=True), 1e-8), F_SCALAR),

    "third_raw": scalar_dct(third, F_SCALAR),
    "third_center": scalar_dct(third-third.mean(axis=1, keepdims=True), F_SCALAR),
    "third_zscore": scalar_dct(
        (third-third.mean(axis=1, keepdims=True))/np.maximum(third.std(axis=1, keepdims=True), 1e-8),
        F_SCALAR
    )
}

print("Feature blocks:", {k: v.shape[1] for k, v in blocks.items()}, flush=True)


# ================================================================
# 5. TRAINING-ONLY DISTANCES
# ================================================================

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
splits = list(cv.split(np.zeros(n_train), y_train))

train_sq = {
    name: squareform(pdist(F[:n_train], metric="sqeuclidean")).astype(np.float32)
    for name, F in blocks.items()
}

folds = []

for it, iv in splits:

    distances = {}

    for name, D in train_sq.items():

        A = D[np.ix_(it, it)]
        scale = max(float(A.sum())/(len(it)*(len(it)-1)), 1e-12)

        distances[name] = (A/scale, D[np.ix_(iv, it)]/scale)

    folds.append((it, iv, distances))


# ================================================================
# 6. WEIGHTED KERNEL FUSION
# ================================================================

def combine_distances(recipe, cache):

    denominator = sum(recipe.values())

    A = sum(w*cache[key][0] for key, w in recipe.items())/denominator
    B = sum(w*cache[key][1] for key, w in recipe.items())/denominator

    return A, B


def cv_tune(recipe, Cs, gammas):

    prepared = [(it, iv, *combine_distances(recipe, data)) for it, iv, data in folds]
    best = (-np.inf, None, None)

    for gamma in gammas:

        kernels = [
            (it, iv, np.exp(-gamma*A), np.exp(-gamma*B))
            for it, iv, A, B in prepared
        ]

        for C in Cs:

            scores = []

            for it, iv, A, B in kernels:

                prediction = SVC(C=C, kernel="precomputed").fit(A, y_train[it]).predict(B)
                scores.append(f1_score(y_train[iv], prediction, average="macro", zero_division=0))

            score = float(np.mean(scores))

            if score > best[0]:
                best = (score, C, gamma)

    return best


# ================================================================
# 7. SCREENING FEATURE TRANSFORMATIONS AND WEIGHTS
# ================================================================

def screen(name, recipes):

    scores = [(cv_tune(recipe, COARSE_C, COARSE_GAMMA)[0], recipe) for recipe in recipes]
    score, recipe = max(scores, key=lambda x: x[0])

    print(f"{name:28s} | Screen F1={score:.4f} | Weights={recipe}", flush=True)

    return recipe


print("\nCOARSE SEARCH", flush=True)

selected = {"Angle only": {"angle": 1.0}}

selected["Angle + speed"] = screen(
    "Angle + speed",
    [{"angle": 1.0, f"speed_{form}": w}
     for form in ("raw", "log", "relative") for w in WEIGHTS]
)

selected["Angle + third channel"] = screen(
    "Angle + third channel",
    [{"angle": 1.0, f"third_{form}": w}
     for form in ("raw", "center", "zscore") for w in WEIGHTS]
)

selected["Angle + local DCT"] = screen(
    "Angle + local DCT",
    [{"angle": 1.0, "local": w} for w in WEIGHTS]
)

speed_key = next(k for k in selected["Angle + speed"] if k.startswith("speed_"))
third_key = next(k for k in selected["Angle + third channel"] if k.startswith("third_"))

selected["Angle + speed + third"] = screen(
    "Angle + speed + third",
    [{"angle": 1., speed_key: s, third_key: p}
     for s in WEIGHTS for p in WEIGHTS]
)

selected["Angle + speed + local"] = screen(
    "Angle + speed + local",
    [{"angle": 1., speed_key: s, "local": l}
     for s in WEIGHTS for l in WEIGHTS]
)

selected["Angle + third + local"] = screen(
    "Angle + third + local",
    [{"angle": 1., third_key: p, "local": l}
     for p in WEIGHTS for l in WEIGHTS]
)

selected["All four blocks"] = screen(
    "All four blocks",
    [{"angle": 1., speed_key: s, third_key: p, "local": l}
     for s in WEIGHTS for p in WEIGHTS for l in WEIGHTS]
)


# ================================================================
# 8. FULL CROSS-VALIDATION
# ================================================================

print("\nFULL CROSS-VALIDATION", flush=True)

refined = []

for name, recipe in selected.items():

    score, C, gamma = cv_tune(recipe, FULL_C, FULL_GAMMA)
    dim = sum(blocks[k].shape[1] for k in recipe)

    refined.append((score, name, recipe, C, gamma, dim))

    print(f"{name:28s} | Dim={dim:3d} | CV F1={score:.4f} | "
          f"C={C:g} | gamma={gamma:g}", flush=True)

refined.sort(key=lambda r: r[0], reverse=True)

print("\nSELECTED:", refined[0], flush=True)


# ================================================================
# 9. OFFICIAL TEST
# ================================================================

def test_model(recipe, C, gamma):

    n = n_train
    total = sum(recipe.values())

    A = np.zeros((n, n), dtype=np.float64)
    B = np.zeros((n_test, n), dtype=np.float64)

    for key, w in recipe.items():

        D = train_sq[key]
        scale = max(float(D.sum())/(n*(n-1)), 1e-12)

        A += w*D/scale
        B += w*cdist(blocks[key][n:], blocks[key][:n], metric="sqeuclidean")/scale

    A /= total
    B /= total

    clf = SVC(C=C, kernel="precomputed").fit(np.exp(-gamma*A), y_train)
    pred = clf.predict(np.exp(-gamma*B))

    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro", zero_division=0)


print("\nOFFICIAL TEST", flush=True)

rows = refined if EVALUATE_EACH_FAMILY_ON_TEST else refined[:1]

for cv_f1, name, recipe, C, gamma, dim in rows:

    acc, f1 = test_model(recipe, C, gamma)

    print(f"{name:28s} | Dim={dim:3d} | CV F1={cv_f1:.4f} | "
          f"Test Acc={acc:.4f} | Test Macro-F1={f1:.4f}", flush=True)


# ================================================================
# 10. FIGURE — CV PERFORMANCE BY FEATURE FAMILY
# ================================================================

ordered = list(selected)
values = {name: next(r[0] for r in refined if r[1] == name) for name in ordered}

fig, ax = plt.subplots(figsize=(9, 4.8))

ax.barh(ordered[::-1], [values[k] for k in ordered[::-1]])
ax.axvline(values["Angle only"], linewidth=1, linestyle="--",
           color="gray", label="Angle-only baseline")

ax.set(xlabel="Cross-validated Macro-F1")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.2)
ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)
