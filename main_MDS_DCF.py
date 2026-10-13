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


# ================================================================
# SPEED ABLATION — GLOBAL DCT, LOCAL DCT, SYMBOLIC QUANTIZATION
# ================================================================

from sklearn.model_selection import StratifiedKFold

FR_VALUES = (4, 8, 12, 16, 24, 32)
SPEED_WEIGHTS = (0.25, 0.5, 1.0, 2.0)
LOCAL_WEIGHTS = (0.25, 0.5, 1.0)
LOCAL_CONFIGS = ((2, 4), (2, 8), (4, 4), (4, 8))

COARSE_C = (10.0, 100.0)
COARSE_GAMMA = (0.125, 0.25, 0.5, 1.0)
FINAL_C = (1.0, 10.0, 100.0, 1000.0)
FINAL_GAMMA = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)

relative_speed = speed/np.maximum(speed.mean(axis=1, keepdims=True), 1e-8)
angle_features = blocks["angle"]

print("\nReference — Angle + speed relative, 48D: CV F1=0.9877 | Test Acc=0.9889")


# ================================================================
# 1. SPEED ENCODING
# ================================================================

def encode_speed(mode, q, encoding, fit_ids):

    if mode == "continuous":
        return relative_speed[:, :, None].astype(np.float32)

    # Fit quantization thresholds on training observations only
    edges = np.quantile(relative_speed[fit_ids].ravel(), np.arange(1, q)/q)
    codes = np.searchsorted(edges, relative_speed, side="right")

    if encoding == "ordinal":
        X = (codes/(q-1)).astype(np.float32)
        return X[:, :, None]

    if encoding == "onehot":
        return np.eye(q, dtype=np.float32)[codes]

    raise ValueError(encoding)


def global_dct(X, nf):
    return dct(X, type=2, axis=1, norm="ortho")[:, :nf, :].reshape(len(X), -1)


def local_dct(X, n_segments, nf):

    parts = np.array_split(np.arange(X.shape[1]), n_segments)

    return np.concatenate([
        global_dct(X[:, indices, :], nf) for indices in parts
    ], axis=1)


# ================================================================
# 2. FOLD-SPECIFIC DISTANCE NORMALIZATION
# ================================================================

def normalized_pair(F, it, iv):

    A = squareform(pdist(F[it], metric="sqeuclidean")).astype(np.float32)
    B = cdist(F[iv], F[it], metric="sqeuclidean").astype(np.float32)

    scale = max(float(A.sum())/(len(it)*(len(it)-1)), 1e-12)
    return A/scale, B/scale


def prepare_folds(config, splits):

    mode, q, encoding, fr, ns, fl = config
    result = []

    for it, iv in splits:

        X = encode_speed(mode, q, encoding, it)
        Fg = global_dct(X, fr)

        Aa, Ba = normalized_pair(angle_features, it, iv)
        Ag, Bg = normalized_pair(Fg, it, iv)

        if ns > 0:
            Fl = local_dct(X, ns, fl)
            Al, Bl = normalized_pair(Fl, it, iv)
        else:
            Al = np.zeros_like(Aa)
            Bl = np.zeros_like(Ba)

        result.append((it, iv, Aa, Ba, Ag, Bg, Al, Bl))

    return result


# ================================================================
# 3. WEIGHTED KERNEL OPTIMIZATION
# ================================================================

def optimize_model(prepared, weights, Cs, gammas):

    best = (-np.inf, None, None, None, None)

    for wr, wl in weights:

        denominator = 1.0+wr+wl
        matrices = []

        for it, iv, Aa, Ba, Ag, Bg, Al, Bl in prepared:

            A = (Aa+wr*Ag+wl*Al)/denominator
            B = (Ba+wr*Bg+wl*Bl)/denominator
            matrices.append((it, iv, A, B))

        for gamma in gammas:

            kernels = [(it, iv, np.exp(-gamma*A), np.exp(-gamma*B))
                       for it, iv, A, B in matrices]

            for C in Cs:

                scores = []

                for it, iv, A, B in kernels:

                    clf = SVC(C=C, kernel="precomputed").fit(A, y_train[it])
                    pred = clf.predict(B)

                    scores.append(f1_score(y_train[iv], pred, average="macro", zero_division=0))

                score = float(np.mean(scores))

                if score > best[0]:
                    best = (score, wr, wl, C, gamma)

    return best


screen_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
screen_splits = list(screen_cv.split(np.zeros(n_train), y_train))

final_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
final_splits = list(final_cv.split(np.zeros(n_train), y_train))


# ================================================================
# 4. EXPERIMENT A — GLOBAL SPEED DCT AND QUANTIZATION
# ================================================================

GLOBAL_TYPES = [
    ("continuous", 0, "continuous"),
    ("quantized", 4, "ordinal"),
    ("quantized", 4, "onehot"),
    ("quantized", 8, "ordinal"),
    ("quantized", 8, "onehot")
]

global_results = []

print("\n" + "="*85)
print("GLOBAL SPEED DCT — FREQUENCY AND SYMBOLIC QUANTIZATION")
print("="*85, flush=True)

for mode, q, encoding in GLOBAL_TYPES:

    for fr in FR_VALUES:

        config = (mode, q, encoding, fr, 0, 0)
        prepared = prepare_folds(config, screen_splits)

        weights = [(wr, 0.0) for wr in SPEED_WEIGHTS]
        score, wr, wl, C, gamma = optimize_model(prepared, weights, COARSE_C, COARSE_GAMMA)

        global_results.append((score, config, wr, wl, C, gamma))

        dim_speed = fr*(1 if encoding != "onehot" else q)

        print(f"{mode:10s} | q={q:1d} | {encoding:10s} | Fr={fr:2d} | "
              f"Dim={32+dim_speed:3d} | F1={score:.4f} | wr={wr:g}", flush=True)

global_results.sort(reverse=True)

print("\nTOP GLOBAL MODELS")
for row in global_results[:10]:
    print(row)


# ================================================================
# 5. EXPERIMENT B — MULTIRESOLUTION SPEED DCT
# ================================================================

# Select the best continuous speed frequency count
best_continuous = next(r for r in global_results if r[1][0] == "continuous")
fr_best = best_continuous[1][3]
wr_best = best_continuous[2]

local_results = []

print("\n" + "="*85)
print("MULTIRESOLUTION SPEED DCT")
print("="*85, flush=True)

for ns, fl in LOCAL_CONFIGS:

    config = ("continuous", 0, "continuous", fr_best, ns, fl)
    prepared = prepare_folds(config, screen_splits)

    weights = [(wr_best, wl) for wl in LOCAL_WEIGHTS]
    score, wr, wl, C, gamma = optimize_model(prepared, weights, COARSE_C, COARSE_GAMMA)

    local_results.append((score, config, wr, wl, C, gamma))

    dimension = 32+fr_best+ns*fl

    print(f"Segments={ns} | Local F={fl} | Dim={dimension:3d} | "
          f"F1={score:.4f} | wr={wr:g} | wl={wl:g}", flush=True)

local_results.sort(reverse=True)


# ================================================================
# 6. FINALISTS — FIVE-FOLD CROSS-VALIDATION
# ================================================================

# Best two configurations from each global speed representation
finalists = []

for mode, q, encoding in GLOBAL_TYPES:
    candidates = [r for r in global_results if r[1][:3] == (mode, q, encoding)]
    finalists.extend(candidates[:2])

# Two best local speed representations
finalists.extend(local_results[:2])

# Include original best reference
reference = ("continuous", 0, "continuous", 16, 0, 0)
finalists.append((0.0, reference, 0.5, 0.0, 10.0, 0.25))

final_configs = list(dict.fromkeys(row[1] for row in finalists))

final_results = []

print("\n" + "="*85)
print("FINAL FIVE-FOLD CROSS-VALIDATION")
print("="*85, flush=True)

for config in final_configs:

    prepared = prepare_folds(config, final_splits)
    mode, q, encoding, fr, ns, fl = config

    speed_weights = SPEED_WEIGHTS
    local_weights = LOCAL_WEIGHTS if ns > 0 else (0.0,)
    weights = [(wr, wl) for wr in speed_weights for wl in local_weights]

    score, wr, wl, C, gamma = optimize_model(prepared, weights, FINAL_C, FINAL_GAMMA)

    d = 1 if encoding != "onehot" else q
    dimension = 32+(fr+ns*fl)*d

    final_results.append((score, config, wr, wl, C, gamma, dimension))

    print(f"{str(config):49s} | Dim={dimension:3d} | "
          f"F1={score:.4f} | wr={wr:g} | wl={wl:g} | "
          f"C={C:g} | gamma={gamma:g}", flush=True)

final_results.sort(reverse=True)

print("\nBEST FIVE-FOLD CONFIGURATIONS")
for row in final_results[:10]:
    print(row)


# ================================================================
# 7. OFFICIAL TEST — BEST MODEL ONLY
# ================================================================

score, config, wr, wl, C, gamma, dimension = final_results[0]
mode, q, encoding, fr, ns, fl = config

X = encode_speed(mode, q, encoding, np.arange(n_train))
Fg = global_dct(X, fr)
Fl = local_dct(X, ns, fl) if ns > 0 else None

def full_pair(F):

    A = squareform(pdist(F[:n_train], metric="sqeuclidean")).astype(np.float32)
    B = cdist(F[n_train:], F[:n_train], metric="sqeuclidean").astype(np.float32)

    scale = max(float(A.sum())/(n_train*(n_train-1)), 1e-12)
    return A/scale, B/scale


Aa, Ba = full_pair(angle_features)
Ag, Bg = full_pair(Fg)

A = Aa+wr*Ag
B = Ba+wr*Bg

if Fl is not None:
    Al, Bl = full_pair(Fl)
    A += wl*Al
    B += wl*Bl

A /= 1+wr+wl
B /= 1+wr+wl

clf = SVC(C=C, kernel="precomputed").fit(np.exp(-gamma*A), y_train)
pred = clf.predict(np.exp(-gamma*B))

test_accuracy = accuracy_score(y_test, pred)
test_f1 = f1_score(y_test, pred, average="macro", zero_division=0)

print("\n" + "="*85)
print("OFFICIAL TEST — SELECTED MODEL")
print("="*85)
print(f"Configuration: {config}")
print(f"Dimension: {dimension} | wr={wr:g} | wl={wl:g} | C={C:g} | gamma={gamma:g}")
print(f"CV Macro-F1={score:.4f}")
print(f"Test Accuracy={test_accuracy:.4f} | Test Macro-F1={test_f1:.4f}")
print(f"Errors={np.sum(pred != y_test)}/{n_test}")


# ================================================================
# 8. FIGURE — SPEED FREQUENCIES
# ================================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

for mode, q, encoding in GLOBAL_TYPES:

    rows = sorted([r for r in global_results if r[1][:3] == (mode, q, encoding)],
                  key=lambda r: r[1][3])

    label = "Continuous" if mode == "continuous" else f"{q} levels — {encoding}"

    ax.plot([r[1][3] for r in rows], [r[0] for r in rows],
            marker="o", markersize=3, linewidth=1.3, label=label)

ax.set(xlabel="Number of speed DCT frequencies", ylabel="3-fold CV Macro-F1",
       xticks=FR_VALUES)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
plt.close(fig)


# ================================================================
# 9. FIGURE — CV PERFORMANCE VS DIMENSION
# ================================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

for label, predicate in [
    ("Continuous", lambda r: r[1][0] == "continuous"),
    ("4-level symbolic", lambda r: r[1][1] == 4),
    ("8-level symbolic", lambda r: r[1][1] == 8)
]:

    rows = sorted([r for r in final_results if predicate(r)], key=lambda r: r[6])

    if rows:
        ax.scatter([r[6] for r in rows], [r[0] for r in rows],
                   s=28, label=label)

ax.axhline(0.9877, linestyle="--", color="gray", linewidth=1,
           label="Previous 3-fold CV reference")

ax.set(xlabel="Representation dimension", ylabel="5-fold CV Macro-F1")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
plt.close(fig)


# ================================================================
# 1. ANALYSIS OF OFFICIAL TEST ERRORS
# ================================================================

from collections import Counter
from scipy.stats import binomtest
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

errors = np.flatnonzero(pred != y_test)

print("\n" + "="*75)
print(f"ERROR ANALYSIS — {len(errors)} / {len(y_test)}")
print("="*75)

for j in errors:
    print(f"Test #{j:4d} | True={y_test[j]} | Predicted={pred[j]}")

confusions = Counter((str(y_test[j]), str(pred[j])) for j in errors)

print("\nMost frequent confusions:")
for (true, predicted), count in confusions.most_common():
    print(f"{true} -> {predicted}: {count}")


# ================================================================
# 2. RECONSTRUCTED CHARACTER TRAJECTORIES
# ================================================================

ncols = 4
nrows = int(np.ceil(len(errors)/ncols))

if len(errors):

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8*nrows), squeeze=False)

    for ax, j in zip(axes.ravel(), errors):

        case = np.asarray(cases[n_train+j])
        xy = np.cumsum(case[:2].T, axis=0)
        xy -= xy.mean(axis=0)

        scale = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-12)
        xy /= scale

        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.2)
        ax.scatter(xy[0, 0], xy[0, 1], s=24, marker="o", zorder=3)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=24, marker="x", zorder=3)

        ax.set_title(f"#{j} | {y_test[j]} → {pred[j]}", fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")

    for ax in axes.ravel()[len(errors):]:
        ax.axis("off")

    fig.suptitle("Misclassified characters — reconstructed trajectories")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ================================================================
# 3. SPEED PROFILES OF MISCLASSIFIED CHARACTERS
# ================================================================

if len(errors):

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.5*nrows), squeeze=False)

    for ax, j in zip(axes.ravel(), errors):

        r = relative_speed[n_train+j]

        ax.plot(np.linspace(0, 1, len(r)), r, linewidth=1.2)
        ax.set_title(f"#{j} | {y_test[j]} → {pred[j]}", fontsize=10)
        ax.set_xlabel("Normalized time")
        ax.set_ylabel("Relative speed", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)

    for ax in axes.ravel()[len(errors):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ================================================================
# 4. REPEATED NESTED CROSS-VALIDATION
# ================================================================

METHODS = {
    "Continuous 48D": ("continuous", 16),
    "Continuous 40D": ("continuous", 8),
    "Symbolic 40D": ("symbolic8", 8)
}

N_OUTER, N_REPEATS, N_INNER = 5, 2, 3

CV_WEIGHTS = (0.125, 0.25, 0.5, 1.0)
CV_C = (1.0, 10.0, 100.0)
CV_GAMMA = (0.125, 0.25, 0.5, 1.0, 2.0)

ANGLE = np.asarray(blocks["angle"], dtype=np.float32)
SPEED = np.asarray(relative_speed, dtype=np.float32)

outer_cv = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=42)

oof = {
    name: np.empty((N_REPEATS, n_train), dtype=y_train.dtype)
    for name in METHODS
}


# ================================================================
# 5. SPEED FEATURES — FOLD-SPECIFIC QUANTIZATION
# ================================================================

def speed_descriptor(method, fit_ids, apply_ids):

    mode, nf = METHODS[method]
    X = SPEED[apply_ids]

    if mode == "symbolic8":
        edges = np.quantile(SPEED[fit_ids].ravel(), np.arange(1, 8)/8)
        X = np.searchsorted(edges, X, side="right").astype(np.float32)/7.0

    return dct(X, type=2, axis=1, norm="ortho")[:, :nf].astype(np.float32)


def normalized_pair(Xtr, Xval):

    A = squareform(pdist(Xtr, metric="sqeuclidean")).astype(np.float32)
    B = cdist(Xval, Xtr, metric="sqeuclidean").astype(np.float32)

    scale = max(float(A.sum())/(len(Xtr)*(len(Xtr)-1)), 1e-12)
    return A/scale, B/scale


# ================================================================
# 6. INNER CROSS-VALIDATION — HYPERPARAMETER SEARCH
# ================================================================

def optimize_inner(prepared):

    best = (-np.inf, None, None, None)

    for wr in CV_WEIGHTS:

        combined = [
            (it, iv, (Aa+wr*Ar)/(1+wr), (Ba+wr*Br)/(1+wr))
            for it, iv, Aa, Ba, Ar, Br in prepared
        ]

        for gamma in CV_GAMMA:

            kernels = [
                (it, iv, np.exp(-gamma*A), np.exp(-gamma*B))
                for it, iv, A, B in combined
            ]

            for C in CV_C:

                scores = []

                for it, iv, Ktr, Kval in kernels:

                    model = SVC(C=C, kernel="precomputed")
                    model.fit(Ktr, y_train[it])

                    prediction = model.predict(Kval)
                    scores.append(f1_score(y_train[iv], prediction,
                                           average="macro", zero_division=0))

                score = float(np.mean(scores))

                if score > best[0]:
                    best = (score, wr, C, gamma)

    return best


# ================================================================
# 7. OUTER CROSS-VALIDATION — UNBIASED PREDICTIONS
# ================================================================

print("\n" + "="*85)
print("REPEATED NESTED CROSS-VALIDATION")
print("="*85, flush=True)

for iteration, (outer_tr, outer_val) in enumerate(
        outer_cv.split(np.zeros(n_train), y_train)):

    repeat = iteration//N_OUTER
    fold = iteration % N_OUTER

    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                               random_state=1000+iteration)

    inner_splits = list(inner_cv.split(np.zeros(len(outer_tr)), y_train[outer_tr]))
    prepared = {name: [] for name in METHODS}

    # Inner training and validation
    for itr, ival in inner_splits:

        it, iv = outer_tr[itr], outer_tr[ival]
        Aa, Ba = normalized_pair(ANGLE[it], ANGLE[iv])

        for name in METHODS:

            Rtr = speed_descriptor(name, it, it)
            Rval = speed_descriptor(name, it, iv)
            Ar, Br = normalized_pair(Rtr, Rval)

            prepared[name].append((it, iv, Aa, Ba, Ar, Br))

    # Outer training and validation
    Aa, Ba = normalized_pair(ANGLE[outer_tr], ANGLE[outer_val])

    for name in METHODS:

        cv_f1, wr, C, gamma = optimize_inner(prepared[name])

        Rtr = speed_descriptor(name, outer_tr, outer_tr)
        Rval = speed_descriptor(name, outer_tr, outer_val)
        Ar, Br = normalized_pair(Rtr, Rval)

        A = (Aa+wr*Ar)/(1+wr)
        B = (Ba+wr*Br)/(1+wr)

        model = SVC(C=C, kernel="precomputed")
        model.fit(np.exp(-gamma*A), y_train[outer_tr])

        prediction = model.predict(np.exp(-gamma*B))
        oof[name][repeat, outer_val] = prediction

        acc = accuracy_score(y_train[outer_val], prediction)

        print(f"Repeat={repeat+1} | Fold={fold+1} | {name:15s} | "
              f"Acc={acc:.4f} | wr={wr:g} | C={C:g} | gamma={gamma:g}",
              flush=True)


# ================================================================
# 8. REPEATED OUT-OF-FOLD RESULTS
# ================================================================

print("\n" + "="*85)
print("OUT-OF-FOLD RESULTS")
print("="*85)

for name in METHODS:

    accuracies = []
    f1_scores = []

    for repeat in range(N_REPEATS):

        prediction = oof[name][repeat]

        accuracies.append(accuracy_score(y_train, prediction))
        f1_scores.append(f1_score(y_train, prediction, average="macro", zero_division=0))

    print(f"{name:15s} | Accuracy={np.mean(accuracies):.4f} ± {np.std(accuracies):.4f} | "
          f"Macro-F1={np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")


# ================================================================
# 9. PAIRED MODEL COMPARISONS
# ================================================================

print("\n" + "="*85)
print("PAIRED ERROR COMPARISONS")
print("="*85)

reference = "Continuous 48D"

for alternative in ("Continuous 40D", "Symbolic 40D"):

    print(f"\n{reference} vs {alternative}")

    for repeat in range(N_REPEATS):

        correct_ref = oof[reference][repeat] == y_train
        correct_alt = oof[alternative][repeat] == y_train

        ref_only = int(np.sum(correct_ref & ~correct_alt))
        alt_only = int(np.sum(correct_alt & ~correct_ref))

        n_discordant = ref_only+alt_only
        p = binomtest(min(ref_only, alt_only), n_discordant, 0.5).pvalue if n_discordant else 1.0

        print(f"Repeat {repeat+1} | 48D only={ref_only} | "
              f"Alternative only={alt_only} | McNemar p={p:.4f}")


# ================================================================
# 10. FINAL FIGURE
# ================================================================

names = list(METHODS)
means = [
    np.mean([accuracy_score(y_train, oof[name][r]) for r in range(N_REPEATS)])
    for name in names
]

fig, ax = plt.subplots(figsize=(7, 4))

ax.barh(names[::-1], means[::-1])
ax.set(xlabel="Repeated nested CV accuracy", xlim=(0.95, 1.0))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.show()
plt.close(fig)


import numpy as np
import matplotlib.pyplot as plt

from numba import njit, prange
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


# ================================================================
# 1. PARAMETERS AND EXISTING FEATURES
# ================================================================

C_REF, GAMMA_REF, WR_REF = 10.0, 1.0, 0.25

N_ALIGN = 48
DTW_BAND = 8
LANDMARKS_PER_CLASS = 2

BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
C_VALUES = (1.0, 10.0, 100.0)
GAMMA_VALUES = (0.25, 1.0, 4.0)

RUN_OFFICIAL_FUSION_TEST = False

A_FEATURES = np.asarray(blocks["angle"], dtype=np.float32)
R_FEATURES = np.asarray(blocks["speed_relative"], dtype=np.float32)
SPEED = np.asarray(relative_speed, dtype=np.float32)
Z_ALIGN = np.asarray(Z, dtype=np.float32)

assert len(A_FEATURES) == n_train + n_test
assert A_FEATURES.shape[1] == 32 and R_FEATURES.shape[1] == 16
assert Z_ALIGN.shape[2] == 4


# ================================================================
# 2. NORMALIZED SQUARED DISTANCES
# ================================================================

def normalized_pair(Xtr, Xval):

    A = squareform(pdist(Xtr, metric="sqeuclidean")).astype(np.float32)
    B = cdist(Xval, Xtr, metric="sqeuclidean").astype(np.float32)

    scale = max(float(A.sum())/(len(Xtr)*(len(Xtr)-1)), 1e-12)
    return A/scale, B/scale


# ================================================================
# 3. REFIT THE CURRENT 48D MODEL
# ================================================================

Aa, Ba = normalized_pair(A_FEATURES[:n_train], A_FEATURES[n_train:])
Ar, Br = normalized_pair(R_FEATURES[:n_train], R_FEATURES[n_train:])

Dtr = (Aa+WR_REF*Ar)/(1+WR_REF)
Dte = (Ba+WR_REF*Br)/(1+WR_REF)

model_ref = SVC(C=C_REF, kernel="precomputed", decision_function_shape="ovr", break_ties=True)
model_ref.fit(np.exp(-GAMMA_REF*Dtr), y_train)

Kte = np.exp(-GAMMA_REF*Dte)
pred_ref = model_ref.predict(Kte)
scores = model_ref.decision_function(Kte)
classes = model_ref.classes_

errors = np.flatnonzero(pred_ref != y_test)

print("\n" + "="*85)
print("REFERENCE MODEL — ERROR DIAGNOSTICS")
print("="*85)

print(f"Accuracy={accuracy_score(y_test, pred_ref):.4f}")
print(f"Macro-F1={f1_score(y_test, pred_ref, average='macro'):.4f}")
print(f"Errors={len(errors)}/{n_test}")

if "pred" in globals() and not np.array_equal(pred_ref, pred):
    print("WARNING: Refit predictions differ from the previous test predictions.")


# ================================================================
# 4. CLASS SCORES AND NEAREST NEIGHBORS
# ================================================================

error_neighbors = {}

for j in errors:

    true_label, predicted_label = y_test[j], pred_ref[j]

    true_col = np.flatnonzero(classes == true_label)[0]
    predicted_col = np.flatnonzero(classes == predicted_label)[0]

    score_true = scores[j, true_col]
    score_pred = scores[j, predicted_col]
    margin = score_pred-score_true

    order = np.argsort(Dte[j])
    nearest = order[:5]

    true_candidates = np.flatnonzero(y_train == true_label)
    predicted_candidates = np.flatnonzero(y_train == predicted_label)

    nearest_true = true_candidates[np.argmin(Dte[j, true_candidates])]
    nearest_pred = predicted_candidates[np.argmin(Dte[j, predicted_candidates])]

    error_neighbors[j] = (nearest_true, nearest_pred)

    print(f"\nTest #{j} | True={true_label} | Predicted={predicted_label}")
    print(f"  SVM score gap (predicted - true): {margin:+.5f}")
    print(f"  Nearest true-class example:  #{nearest_true} | Distance={Dte[j, nearest_true]:.4f}")
    print(f"  Nearest predicted-class example: #{nearest_pred} | Distance={Dte[j, nearest_pred]:.4f}")
    print("  Five nearest training examples:")

    for k in nearest:
        print(f"    #{k:4d} | Class={y_train[k]} | Distance={Dte[j, k]:.4f}")


# ================================================================
# 5. VISUAL COMPARISON WITH NEAREST NEIGHBORS
# ================================================================

important_pairs = {("7", "10"), ("5", "20"), ("18", "9")}
selected_errors = [
    j for j in errors
    if (str(y_test[j]), str(pred_ref[j])) in important_pairs
]


def reconstructed_xy(case):

    xy = np.cumsum(np.asarray(case, dtype=float)[:2].T, axis=0)
    xy -= xy.mean(axis=0)
    scale = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-12)
    return xy/scale


if selected_errors:

    fig, axes = plt.subplots(len(selected_errors), 3,
                             figsize=(10, 2.3*len(selected_errors)), squeeze=False)

    for row, j in enumerate(selected_errors):

        nearest_true, nearest_pred = error_neighbors[j]

        indices = (n_train+j, nearest_true, nearest_pred)
        titles = (
            f"Test #{j} | True {y_test[j]} → {pred_ref[j]}",
            f"Nearest true class #{nearest_true}",
            f"Nearest predicted class #{nearest_pred}"
        )

        for ax, idx, title in zip(axes[row], indices, titles):

            xy = reconstructed_xy(cases[idx])

            ax.plot(xy[:, 0], xy[:, 1], linewidth=1.2)
            ax.scatter(xy[0, 0], xy[0, 1], s=16)
            ax.scatter(xy[-1, 0], xy[-1, 1], s=22, marker="x")

            ax.set_title(title, fontsize=9)
            ax.set_aspect("equal", adjustable="datalim")
            ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ================================================================
# 6. FAST BANDED DTW
# ================================================================

@njit(fastmath=True)
def dtw_cost(a, b, radius):

    n, m = a.shape[0], b.shape[0]
    previous = np.full(m+1, np.inf)
    current = np.full(m+1, np.inf)
    previous[0] = 0.0

    for i in range(1, n+1):

        current[:] = np.inf
        start = max(1, i-radius)
        stop = min(m, i+radius)

        for j in range(start, stop+1):

            cost = 0.0

            for k in range(a.shape[1]):
                d = a[i-1, k]-b[j-1, k]
                cost += d*d

            current[j] = cost+min(previous[j], previous[j-1], current[j-1])

        previous, current = current, previous

    return previous[m]/n


@njit(parallel=True, fastmath=True)
def dtw_landmarks(X, landmarks, radius):

    out = np.empty((len(X), len(landmarks)), dtype=np.float32)

    for i in prange(len(X)):
        for j in range(len(landmarks)):
            out[i, j] = dtw_cost(X[i], landmarks[j], radius)

    return out


# ================================================================
# 7. TEMPORAL TRAJECTORIES FOR ALIGNMENT
# ================================================================

time_indices = np.linspace(0, Z_ALIGN.shape[1]-1, N_ALIGN).astype(int)


def alignment_paths(ids, fit_ids):

    angular = Z_ALIGN[ids][:, time_indices, :]
    velocity = SPEED[ids][:, time_indices]

    # Normalize angular and speed contributions using training data only
    angular_energy = np.mean(np.sum(Z_ALIGN[fit_ids]**2, axis=2))
    angular_scale = 1.0/np.sqrt(max(float(angular_energy), 1e-12))

    speed_std = max(float(np.std(SPEED[fit_ids])), 1e-8)

    result = np.empty((len(ids), N_ALIGN, 5), dtype=np.float32)
    result[:, :, :4] = angular*angular_scale
    result[:, :, 4] = velocity*(0.5/speed_std)

    return result


def choose_landmarks(train_ids, seed):

    rng = np.random.default_rng(seed)
    selected = []

    for label in np.unique(y_train[train_ids]):

        candidates = train_ids[y_train[train_ids] == label]
        count = min(LANDMARKS_PER_CLASS, len(candidates))

        selected.extend(rng.choice(candidates, count, replace=False))

    return np.asarray(selected, dtype=int)


# ================================================================
# 8. DTW LANDMARK DESCRIPTORS
# ================================================================

def alignment_descriptors(it, iv, seed):

    anchor_ids = choose_landmarks(it, seed)

    Xtr = alignment_paths(it, it)
    Xval = alignment_paths(iv, it)
    Xanchors = alignment_paths(anchor_ids, it)

    Dtr = dtw_landmarks(Xtr, Xanchors, DTW_BAND)
    Dval = dtw_landmarks(Xval, Xanchors, DTW_BAND)

    positive = Dtr[Dtr > 1e-12]
    tau = max(float(np.median(positive)), 1e-12) if positive.size else 1.0

    Ftr = np.exp(-Dtr/tau)
    Fval = np.exp(-Dval/tau)

    return Ftr, Fval


# ================================================================
# 9. BUILD CROSS-VALIDATION KERNELS
# ================================================================

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
splits = list(cv.split(np.zeros(n_train), y_train))

folds = []

print("\n" + "="*85)
print("BUILDING DTW LANDMARK REPRESENTATIONS")
print("="*85, flush=True)

for fold, (it, iv) in enumerate(splits):

    DAtr, DAval = normalized_pair(A_FEATURES[it], A_FEATURES[iv])
    DRtr, DRval = normalized_pair(R_FEATURES[it], R_FEATURES[iv])

    DDtr = (DAtr+WR_REF*DRtr)/(1+WR_REF)
    DDval = (DAval+WR_REF*DRval)/(1+WR_REF)

    Ftr, Fval = alignment_descriptors(it, iv, seed=100+fold)
    DTtr, DTval = normalized_pair(Ftr, Fval)

    folds.append((it, iv, DDtr, DDval, DTtr, DTval))

    print(f"Fold {fold+1}/3 | Landmarks={Ftr.shape[1]} | "
          f"DTW features computed", flush=True)


# ================================================================
# 10. CROSS-VALIDATED KERNEL FUSION
# ================================================================

results = []

print("\n" + "="*85)
print("DCT + DTW KERNEL FUSION")
print("="*85, flush=True)

for beta in BETAS:

    for gamma in GAMMA_VALUES:

        prepared = []

        for it, iv, DDtr, DDval, DTtr, DTval in folds:

            Ktr = beta*np.exp(-gamma*DDtr)+(1-beta)*np.exp(-gamma*DTtr)
            Kval = beta*np.exp(-gamma*DDval)+(1-beta)*np.exp(-gamma*DTval)

            prepared.append((it, iv, Ktr, Kval))

        for C in C_VALUES:

            accuracies, f1_values = [], []

            for it, iv, Ktr, Kval in prepared:

                clf = SVC(C=C, kernel="precomputed").fit(Ktr, y_train[it])
                prediction = clf.predict(Kval)

                accuracies.append(accuracy_score(y_train[iv], prediction))
                f1_values.append(f1_score(y_train[iv], prediction, average="macro"))

            results.append((np.mean(f1_values), np.mean(accuracies), beta, C, gamma))

    best_local = max((r for r in results if r[2] == beta), key=lambda r: r[0])

    print(f"beta={beta:.2f} | CV F1={best_local[0]:.4f} | "
          f"CV Acc={best_local[1]:.4f} | C={best_local[3]:g} | "
          f"gamma={best_local[4]:g}", flush=True)


results.sort(reverse=True)

best_f1, best_acc, best_beta, best_C, best_gamma = results[0]

print("\n" + "="*85)
print("BEST MODEL — TRAINING CROSS-VALIDATION")
print("="*85)

print(f"CV Macro-F1={best_f1:.4f} | Accuracy={best_acc:.4f}")
print(f"beta={best_beta:g} | C={best_C:g} | gamma={best_gamma:g}")

for name, beta_condition in (
    ("DCT only", lambda b: b == 1.0),
    ("DTW only", lambda b: b == 0.0),
    ("Hybrid", lambda b: 0.0 < b < 1.0)
):

    candidates = [r for r in results if beta_condition(r[2])]
    best = max(candidates, key=lambda r: r[0])

    print(f"{name:10s} | CV F1={best[0]:.4f} | CV Acc={best[1]:.4f} | "
          f"beta={best[2]:g} | C={best[3]:g} | gamma={best[4]:g}")


# ================================================================
# 11. OPTIONAL OFFICIAL TEST — CV-SELECTED MODEL ONLY
# ================================================================

if RUN_OFFICIAL_FUSION_TEST:

    all_train = np.arange(n_train)
    all_test = np.arange(n_train, n_train+n_test)

    Ftr, Fte = alignment_descriptors(all_train, all_test, seed=900)
    DTtr, DTte = normalized_pair(Ftr, Fte)

    Ktr = best_beta*np.exp(-best_gamma*Dtr)+(1-best_beta)*np.exp(-best_gamma*DTtr)
    Kte = best_beta*np.exp(-best_gamma*Dte)+(1-best_beta)*np.exp(-best_gamma*DTte)

    classifier = SVC(C=best_C, kernel="precomputed").fit(Ktr, y_train)
    pred_fusion = classifier.predict(Kte)

    errors_fusion = set(np.flatnonzero(pred_fusion != y_test))
    errors_reference = set(errors.tolist())

    print("\n" + "="*85)
    print("OFFICIAL TEST — DCT / DTW FUSION")
    print("="*85)

    print(f"Accuracy={accuracy_score(y_test, pred_fusion):.4f}")
    print(f"Macro-F1={f1_score(y_test, pred_fusion, average='macro'):.4f}")
    print(f"Errors={len(errors_fusion)}/{n_test}")

    print("Corrected errors:", sorted(errors_reference-errors_fusion))
    print("New errors:", sorted(errors_fusion-errors_reference))

