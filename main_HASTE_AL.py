# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from aeon.datasets import load_classification
from scipy.fft import dct
from scipy.spatial.distance import pdist, squareform
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


# ================================================================
# 1. CONFIGURATION
# ================================================================

SEED = 42
K, H, N_TIME = 24, 2, 241
F_ANGLE, F_SPEED, F_ARC = 8, 16, 4
WEIGHT_SPEED = 0.25

ARC_WEIGHTS = (1.0, 2.0, 4.0, 8.0)

C_VALUES = (10.0, 100.0, 1000.0)
GAMMA_VALUES = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

N_OUTER, N_REPEATS, N_INNER = 5, 2, 3

CONFIGS = [("HASTE", 0.0), ("ARC+speed", 0.0)]
CONFIGS += [("HASTE-AL", w) for w in ARC_WEIGHTS]

LABELS = {
    ("HASTE", 0.0): "HASTE",
    ("ARC+speed", 0.0): "ARC+speed",
    **{("HASTE-AL", w): f"AL w={w:g}" for w in ARC_WEIGHTS}
}

METHODS = list(LABELS.values()) + ["HASTE-AL selected"]


# ================================================================
# 2. DATASET — TRAINING SPLIT ONLY
# ================================================================

X_train, y_train = load_classification(
    "CharacterTrajectories", split="train", load_equal_length=False
)

cases = list(X_train)
y_train = np.asarray(y_train)
n_train = len(y_train)

time_grid = np.linspace(0, 1, N_TIME)

print(f"Train={n_train} | Classes={len(np.unique(y_train))}", flush=True)


# ================================================================
# 3. TEMPORAL AND ARC-LENGTH ENCODINGS
# ================================================================

def encode_one(case):

    v = np.nan_to_num(np.asarray(case, dtype=np.float64)[:2],
                      nan=0.0, posinf=0.0, neginf=0.0)

    vx, vy = v
    speed = np.hypot(vx, vy)
    n = len(speed)

    positive = speed[speed > 1e-10]
    threshold = 0.03*np.median(positive) if positive.size else 0.0
    valid = speed > max(threshold, 1e-10)

    theta = np.arctan2(vy, vx)

    if valid.any():
        previous = np.where(valid, np.arange(n), -1)
        np.maximum.accumulate(previous, out=previous)
        previous[previous < 0] = np.flatnonzero(valid)[0]
        theta = theta[previous]
    else:
        theta = np.zeros_like(theta)

    # Temporal sampling — identical to HASTE
    idx_time = np.minimum((time_grid*n).astype(int), n-1)
    r = speed[idx_time]/max(float(speed[idx_time].mean()), 1e-8)

    # Arc-length sampling
    ds = 0.5*(speed[:-1]+speed[1:])
    s = np.r_[0.0, np.cumsum(ds)]

    if s[-1] > 1e-12:

        targets = time_grid*s[-1]
        right = np.clip(np.searchsorted(s, targets), 0, n-1)
        left = np.maximum(right-1, 0)

        idx_arc = np.where(
            np.abs(s[left]-targets) <= np.abs(s[right]-targets),
            left, right
        )

    else:
        idx_arc = idx_time

    # Angular symbolization
    symbols = np.floor(
        (theta+np.pi/K)/(2*np.pi/K)
    ).astype(int) % K

    def harmonics(indices):

        phi = 2*np.pi*symbols[indices]/K

        return np.stack([
            fn(h*phi)*np.sqrt(2.0/K)
            for h in range(1, H+1)
            for fn in (np.cos, np.sin)
        ], axis=1)

    Z_time = harmonics(idx_time)
    Z_arc = harmonics(idx_arc)

    ft = dct(Z_time, type=2, axis=0, norm="ortho")[:F_ANGLE].ravel()
    fa = dct(Z_arc, type=2, axis=0, norm="ortho")[:F_ARC].ravel()
    fr = dct(r, type=2, norm="ortho")[:F_SPEED]

    return ft, fr, fa


encoded = [encode_one(case) for case in cases]

FEATURES = {
    "time": np.asarray([e[0] for e in encoded], dtype=np.float32),
    "speed": np.asarray([e[1] for e in encoded], dtype=np.float32),
    "arc": np.asarray([e[2] for e in encoded], dtype=np.float32)
}

print("\nFEATURE DIMENSIONS")
for key, X in FEATURES.items():
    print(f"{key:6s}: {X.shape[1]}")

print("\nHASTE:     48 coefficients")
print("ARC+speed: 32 coefficients")
print("HASTE-AL: 64 coefficients")


# ================================================================
# 4. PRECOMPUTED SQUARED DISTANCES
# ================================================================

DISTANCES = {
    key: squareform(pdist(X, metric="sqeuclidean"))
    for key, X in FEATURES.items()
}


def fold_blocks(it, iv):

    output = {}

    for key, D in DISTANCES.items():

        A = D[np.ix_(it, it)]
        B = D[np.ix_(iv, it)]

        scale = max(float(A.sum())/(len(it)*(len(it)-1)), 1e-12)
        output[key] = (A/scale, B/scale)

    return output


# ================================================================
# 5. DISTANCE FOR EACH REPRESENTATION
# ================================================================

def model_distance(blocks, config):

    method, weight = config

    At, Bt = blocks["time"]
    Ar, Br = blocks["speed"]
    Aa, Ba = blocks["arc"]

    if method == "HASTE":
        return (
            (At+WEIGHT_SPEED*Ar)/(1+WEIGHT_SPEED),
            (Bt+WEIGHT_SPEED*Br)/(1+WEIGHT_SPEED)
        )

    if method == "ARC+speed":
        return (
            (Aa+WEIGHT_SPEED*Ar)/(1+WEIGHT_SPEED),
            (Ba+WEIGHT_SPEED*Br)/(1+WEIGHT_SPEED)
        )

    if method == "HASTE-AL":
        denominator = 1+WEIGHT_SPEED+weight

        return (
            (At+WEIGHT_SPEED*Ar+weight*Aa)/denominator,
            (Bt+WEIGHT_SPEED*Br+weight*Ba)/denominator
        )

    raise ValueError(config)


# ================================================================
# 6. INNER CROSS-VALIDATION
# ================================================================

def tune_models(train_ids, seed):

    cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    prepared = []

    for a, b in cv.split(np.zeros(len(train_ids)), y_train[train_ids]):

        it, iv = train_ids[a], train_ids[b]
        prepared.append((it, iv, fold_blocks(it, iv)))

    selected = {}

    for config in CONFIGS:

        best = (-np.inf, None, None)

        folded = [
            (it, iv, *model_distance(blocks, config))
            for it, iv, blocks in prepared
        ]

        for gamma in GAMMA_VALUES:

            kernels = [
                (it, iv, np.exp(-gamma*A), np.exp(-gamma*B))
                for it, iv, A, B in folded
            ]

            for C in C_VALUES:

                scores = []

                for it, iv, Ktr, Kval in kernels:

                    model = SVC(C=C, kernel="precomputed")
                    model.fit(Ktr, y_train[it])

                    pred = model.predict(Kval)

                    scores.append(
                        f1_score(y_train[iv], pred, average="macro", zero_division=0)
                    )

                score = float(np.mean(scores))

                if score > best[0]:
                    best = (score, C, gamma)

        selected[config] = best

    return selected


# ================================================================
# 7. OUTER CROSS-VALIDATION
# ================================================================

outer_cv = RepeatedStratifiedKFold(
    n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED
)

oof = {
    name: np.empty((N_REPEATS, n_train), dtype=y_train.dtype)
    for name in METHODS
}

selected_weights = []
selected_parameters = {name: [] for name in METHODS}

print("\n" + "="*90)
print("REPEATED NESTED CROSS-VALIDATION")
print("="*90, flush=True)

for iteration, (it, iv) in enumerate(outer_cv.split(np.zeros(n_train), y_train)):

    repeat = iteration//N_OUTER
    fold = iteration % N_OUTER

    tuned = tune_models(it, seed=1000+iteration)
    blocks = fold_blocks(it, iv)

    outer_predictions = {}

    for config in CONFIGS:

        _, C, gamma = tuned[config]
        A, B = model_distance(blocks, config)

        clf = SVC(C=C, kernel="precomputed")
        clf.fit(np.exp(-gamma*A), y_train[it])

        prediction = clf.predict(np.exp(-gamma*B))
        outer_predictions[config] = prediction

        name = LABELS[config]
        oof[name][repeat, iv] = prediction
        selected_parameters[name].append((C, gamma))

    # Weight selected solely by inner CV
    arc_configs = [cfg for cfg in CONFIGS if cfg[0] == "HASTE-AL"]

    best_arc = max(
        arc_configs,
        key=lambda cfg: (tuned[cfg][0], -cfg[1])
    )

    selected_weights.append(best_arc[1])

    oof["HASTE-AL selected"][repeat, iv] = outer_predictions[best_arc]
    selected_parameters["HASTE-AL selected"].append(
        (best_arc[1], tuned[best_arc][1], tuned[best_arc][2])
    )

    print(f"\nRepeat {repeat+1}/{N_REPEATS} | Fold {fold+1}/{N_OUTER} | "
          f"Selected arc weight={best_arc[1]:g}", flush=True)

    for name in ("HASTE", "ARC+speed", "HASTE-AL selected"):

        prediction = oof[name][repeat, iv]

        acc = accuracy_score(y_train[iv], prediction)
        f1 = f1_score(y_train[iv], prediction, average="macro", zero_division=0)

        print(f"  {name:18s} | Acc={acc:.4f} | F1={f1:.4f}", flush=True)

    weights_report = " | ".join(
        f"w={w:g}: {accuracy_score(y_train[iv], outer_predictions[('HASTE-AL', w)]):.4f}"
        for w in ARC_WEIGHTS
    )

    print("  Fixed weights — " + weights_report, flush=True)


# ================================================================
# 8. AGGREGATED PERFORMANCE
# ================================================================

results = {}

print("\n" + "="*90)
print("AGGREGATED OUT-OF-FOLD RESULTS")
print("="*90)

for name in METHODS:

    accuracies = np.array([
        accuracy_score(y_train, oof[name][r])
        for r in range(N_REPEATS)
    ])

    macro_f1 = np.array([
        f1_score(y_train, oof[name][r], average="macro", zero_division=0)
        for r in range(N_REPEATS)
    ])

    errors = [
        int(np.sum(oof[name][r] != y_train))
        for r in range(N_REPEATS)
    ]

    results[name] = (accuracies, macro_f1)

    print(f"{name:18s} | Acc={accuracies.mean():.4f} ± {accuracies.std(ddof=1):.4f} | "
          f"F1={macro_f1.mean():.4f} ± {macro_f1.std(ddof=1):.4f} | "
          f"Errors={errors}")


# ================================================================
# 9. PAIRED COMPARISONS
# ================================================================

print("\n" + "="*90)
print("PAIRED COMPARISONS")
print("="*90)

for alternative in ("ARC+speed", "HASTE-AL selected"):

    print(f"\nHASTE vs {alternative}")

    for repeat in range(N_REPEATS):

        correct_h = oof["HASTE"][repeat] == y_train
        correct_a = oof[alternative][repeat] == y_train

        haste_only = int(np.sum(correct_h & ~correct_a))
        alternative_only = int(np.sum(correct_a & ~correct_h))

        discordant = haste_only+alternative_only

        p = binomtest(
            min(haste_only, alternative_only), discordant, 0.5
        ).pvalue if discordant else 1.0

        print(f"Repeat {repeat+1} | HASTE only={haste_only} | "
              f"{alternative} only={alternative_only} | McNemar p={p:.4f}")


# ================================================================
# 10. WEIGHT SELECTION FREQUENCY
# ================================================================

print("\n" + "="*90)
print("ARC WEIGHT SELECTION")
print("="*90)

for weight, count in sorted(Counter(selected_weights).items()):
    print(f"Weight={weight:g} | Selected {count}/{N_OUTER*N_REPEATS} folds")

print("\nSELECTED HASTE-AL PARAMETERS")

for params, count in Counter(
    selected_parameters["HASTE-AL selected"]
).most_common():

    print(f"weight={params[0]:g} | C={params[1]:g} | "
          f"gamma={params[2]:g} | Selected {count} folds")


# ================================================================
# 11. CLASS-WISE ERRORS
# ================================================================

print("\n" + "="*90)
print("CLASS-WISE ERRORS")
print("="*90)

for name in ("HASTE", "ARC+speed", "HASTE-AL selected"):

    all_errors = []

    for repeat in range(N_REPEATS):
        wrong = np.flatnonzero(oof[name][repeat] != y_train)
        all_errors.extend(y_train[wrong].tolist())

    print(f"\n{name}")
    for label, count in Counter(all_errors).most_common(8):
        print(f"  Class {label}: {count}")


# ================================================================
# 12. FIGURE — PERFORMANCE VS ARC WEIGHT
# ================================================================

weights = np.asarray(ARC_WEIGHTS)

f1_mean = np.asarray([
    results[f"AL w={w:g}"][1].mean() for w in ARC_WEIGHTS
])

f1_std = np.asarray([
    results[f"AL w={w:g}"][1].std(ddof=1) for w in ARC_WEIGHTS
])

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.errorbar(weights, f1_mean, yerr=f1_std, marker="o",
            markersize=4, linewidth=1.2, capsize=3, label="HASTE-AL")

ax.axhline(results["HASTE"][1].mean(), linestyle="--", linewidth=1,
           label="HASTE — temporal")

ax.axhline(results["ARC+speed"][1].mean(), linestyle=":", linewidth=1.2,
           label="ARC+speed — geometric")

ax.set(xlabel="Arc-length weight", ylabel="Repeated nested CV Macro-F1")
ax.set_xticks(weights)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)


# ================================================================
# 13. FIGURE — TEMPORAL / GEOMETRIC ABLATION
# ================================================================

names = ("HASTE", "ARC+speed", "HASTE-AL selected")

means = [results[name][1].mean() for name in names]
stds = [results[name][1].std(ddof=1) for name in names]

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.bar(np.arange(len(names)), means, yerr=stds, capsize=4)
ax.set_xticks(np.arange(len(names)), names)
ax.set(ylabel="Repeated nested CV Macro-F1")
ax.set_ylim(max(0, min(means)-0.015), 1.0)

ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.show()
plt.close(fig)

for r in range(N_REPEATS):
    h = oof["HASTE"][r] == y_train
    a = oof["AL w=4"][r] == y_train

    h_only = np.flatnonzero(h & ~a)
    a_only = np.flatnonzero(~h & a)
    n = len(h_only)+len(a_only)

    p = binomtest(min(len(h_only), len(a_only)), n, 0.5).pvalue if n else 1.0

    print(f"\nRepeat {r+1} | HASTE only={len(h_only)} | AL only={len(a_only)} | p={p:.4f}")
    print("Corrected classes:", Counter(y_train[a_only]))
    print("New error classes:", Counter(y_train[h_only]))