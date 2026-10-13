# -*- coding: utf-8 -*-
import numpy as np 

from src.embedding import transient_mgdpe
from src.signature import signature_distance, signed_area
from src.visualization import plot_paths_3d, plot_distance, plot_symbol_excursions

from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

  
from itertools import permutations, combinations




def make_events_3(sequence, duration=1.0):
    return [(a, i*duration, duration) for i, a in enumerate(sequence)]


def signature_order2(X):

    dX = np.diff(X, axis=0)
    S1 = np.zeros(X.shape[1])
    S2 = np.zeros((X.shape[1], X.shape[1]))

    for dx in dX:
        S2 += np.outer(S1, dx) + 0.5*np.outer(dx, dx)
        S1 += dx

    return S1, S2


def distance_order2(Sa, Sb):
    return np.sqrt(np.sum((Sa[0]-Sb[0])**2) + np.sum((Sa[1]-Sb[1])**2))


def kendall_distance(seq1, seq2, D, symbols, weighted=True):

    p1 = {a:i for i, a in enumerate(seq1)}
    p2 = {a:i for i, a in enumerate(seq2)}
    idx = {a:i for i, a in enumerate(symbols)}

    return sum((D[idx[a], idx[b]] if weighted else 1.0) for a, b in combinations(seq1, 2) if (p1[a]-p1[b])*(p2[a]-p2[b]) < 0)




symbols = ["A", "B", "C", "D", "E", "F"]

positions = np.array([
    [0.0, 0.0],
    [0.3, 0.1],
    [1.0, 0.1],
    [1.4, 0.4],
    [2.4, 0.0],
    [2.8, 0.6]
])

D = cdist(positions, positions)

sequences = {''.join(p): list(p) for p in permutations("ACE")}
names = list(sequences)


R = np.array([[kendall_distance(sequences[a], sequences[b], D, symbols) for b in names] for a in names])
R_unweighted = np.array([[kendall_distance(sequences[a], sequences[b], D, symbols, weighted=False) for b in names] for a in names])


 

def signature_levels(X, Q=4):
    d = X.shape[1]
    S = [np.array(1.)] + [np.zeros((d,)*q) for q in range(1, Q+1)]

    for dx in np.diff(X, axis=0):
        E = [np.array(1.)]
        for q in range(1, Q+1):
            E.append(np.multiply.outer(E[-1], dx)/q)

        for q in range(Q, 0, -1):
            S[q] += sum(np.multiply.outer(S[q-j], E[j]) for j in range(1, q+1))

    return [s.ravel() for s in S[1:]]


# Compute signatures up to order 4
configs = {"MDS + temporal": "none", "Single spectral filter": "single", "Multiscale": "multiscale"}
names = list(sequences)
tri = np.triu_indices(len(names), 1)
signatures = {}

for label, mode in configs.items():
    signatures[label] = {}

    for name, sequence in sequences.items():
        t, X, _ = transient_mgdpe(make_events_3(sequence), symbols, D, dt=0.01, J=3, tau0=0.25, rho=0.75, sigma=0.7, spectral_mode=mode)
        signatures[label][name] = signature_levels(X[:, :2], Q=4)


# Normalize each signature level by its pooled pairwise RMS distance
scales = {}

for q in (2, 3, 4):
    values = [np.sum((signatures[label][a][q-1]-signatures[label][b][q-1])**2)
              for label in configs for a, b in zip(np.array(names)[tri[0]], np.array(names)[tri[1]])]
    scales[q] = max(np.sqrt(np.mean(values)), 1e-12)


def signature_distance(Sa, Sb, Q):
    return np.sqrt(sum(np.sum((Sa[q-1]-Sb[q-1])**2)/scales[q]**2 for q in range(2, Q+1)))


# Compare orders 2, 3 and 4
matrices = {}

for Q in (2, 3, 4):
    for label in configs:
        S = signatures[label]
        M = np.array([[signature_distance(S[a], S[b], Q) for b in names] for a in names])
        matrices[Q, label] = M

        corr_w = spearmanr(M[tri], R[tri]).statistic
        corr_u = spearmanr(M[tri], R_unweighted[tri]).statistic
        min_mean = M[tri].min()/M[tri].mean()

        print(f"Q={Q} | {label:24s} | Weighted={corr_w:.3f} | Unweighted={corr_u:.3f} | Min/Mean={min_mean:.3f}")


# Plot normalized distance matrices
reference = R/R[tri].mean()
normalized = {key: M/M[tri].mean() for key, M in matrices.items()}
vmax = max(reference.max(), *(M.max() for M in normalized.values()))

fig, axes = plt.subplots(3, 4, figsize=(16, 11))

for row, Q in enumerate((2, 3, 4)):
    for col, label in enumerate(["Weighted Kendall", *configs]):
        ax = axes[row, col]
        M = reference if col == 0 else normalized[Q, label]
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)

        ax.set_xticks(range(len(names)), names, rotation=45)
        ax.set_yticks(range(len(names)), names)

        corr = "" if col == 0 else f"\nSpearman={spearmanr(matrices[Q, label][tri], R[tri]).statistic:.3f}"
        ax.set_title(f"{label}{corr}")

    axes[row, 0].set_ylabel(f"Q={Q}", fontsize=12)

fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label="Normalized distance")
plt.show()
plt.close(fig)


# Correlation evolution across signature orders
fig, ax = plt.subplots(figsize=(6.5, 4))

for label in configs:
    correlations = [spearmanr(matrices[Q, label][tri], R[tri]).statistic for Q in (2, 3, 4)]
    ax.plot([2, 3, 4], correlations, marker="o", linewidth=1.5, markersize=5, label=label)

ax.set(xlabel="Signature order Q", ylabel="Spearman correlation", xticks=[2, 3, 4])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend()
plt.tight_layout()
plt.show()
plt.close(fig)




# # 3. Compare the three geometric representations
# configs = {"MDS + temporal": "none", "Single spectral filter": "single", "Multiscale": "multiscale"}
# matrices = {}
# tri = np.triu_indices(len(names), 1)

# for label, mode in configs.items():

#     signatures = {}

#     for name, sequence in sequences.items():
#         t, X, _ = transient_mgdpe(make_events_3(sequence), symbols, D, dt=0.01, J=3, tau0=0.25, rho=0.75, sigma=0.7, spectral_mode=mode)
#         signatures[name] = signature_order2(X)

#     M = np.array([[distance_order2(signatures[a], signatures[b]) for b in names] for a in names])
#     matrices[label] = M

#     corr_w = spearmanr(M[tri], R[tri])[0]
#     corr_u = spearmanr(M[tri], R_unweighted[tri])[0]
#     values = M[tri]

#     print(f"{label:24s} | Weighted={corr_w:.3f} | Unweighted={corr_u:.3f} | Min/Mean={values.min()/values.mean():.3f}")


# # 4. Distance matrices with common color scale
# all_matrices = {"Weighted Kendall": R, **matrices}
# normalized = {name: M / np.mean(M[tri]) for name, M in all_matrices.items()}
# vmax = max(M.max() for M in normalized.values())

# fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# for ax, (label, M) in zip(axes, normalized.items()):
#     im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)
#     ax.set_xticks(range(len(names)), names, rotation=45)
#     ax.set_yticks(range(len(names)), names)
#     ax.set_title(label)

# fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label="Normalized distance")
# plt.show()
# plt.close(fig)


# # 5. Signature distances versus weighted Kendall distance
# fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

# for ax, (label, M) in zip(axes, matrices.items()):
#     corr = spearmanr(M[tri], R[tri])[0]
#     ax.scatter(R[tri], M[tri], s=30)
#     ax.set(xlabel="Weighted Kendall distance", ylabel="Signature distance", title=f"{label} (Spearman={corr:.3f})")
#     ax.spines[["top", "right"]].set_visible(False)
#     ax.grid(alpha=0.2)

# plt.tight_layout()
# plt.show()
# plt.close(fig)









# def evaluate_symbol_geometry(symbols, D, rho=0.5, duration=1.0, **kwargs):

#     paths = [transient_mgdpe([(a, 0, duration)], symbols, D, rho=rho, **kwargs)[1] for a in symbols]
#     symbolic_distances, geometric_distances = [], []

#     for i in range(len(symbols)):
#         for j in range(i+1, len(symbols)):
#             symbolic_distances.append(D[i, j])
#             geometric_distances.append(np.max(np.linalg.norm(paths[i]-paths[j], axis=1)))

#     correlation, _ = spearmanr(symbolic_distances, geometric_distances)
#     print(f"rho={rho:.2f} | Spearman={correlation:.4f}")

#     plt.figure(figsize=(5, 4))
#     plt.scatter(symbolic_distances, geometric_distances, s=35)
#     plt.xlabel("Symbol dissimilarity")
#     plt.ylabel("Geometric path distance")
#     plt.grid(alpha=0.2)
#     plt.tight_layout()
#     plt.show()

#     return correlation





# sequences = {''.join(p): list(p) for p in permutations("ACE")}




# configs = {"MDS only": (0.0, 1), "Single scale": (0.75, 1), "Multiscale": (0.75, 3)}
# names = list(sequences)
# matrices = {}

# fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# for ax, (label, (rho, J)) in zip(axes, configs.items()):

#     signatures = {}

#     for name, sequence in sequences.items():
#         t, X, _ = transient_mgdpe(make_events_3(sequence), symbols, D, dt=0.01, J=J, tau0=0.25, rho=rho, sigma=0.7)
#         signatures[name] = signature_order2(X)

#     matrix = np.array([[distance_order2(signatures[a], signatures[b]) for b in names] for a in names])
#     matrices[label] = matrix

#     im = ax.imshow(matrix, cmap="viridis", vmin=0)
#     ax.set_xticks(range(len(names)), names, rotation=45)
#     ax.set_yticks(range(len(names)), names)
#     ax.set_title(label)
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()
# plt.clf()



# for label, (rho, J) in configs.items():

#     print(f"\n{label}")

#     for name, sequence in sequences.items():
#         t, X, _ = transient_mgdpe(make_events_3(sequence), symbols, D, dt=0.01, J=J, tau0=0.25, rho=rho, sigma=0.7)
#         area = signed_area(X)
#         print(f"{name} | Area={area:+.6f}")

# for label, M in matrices.items():
#     values = M[np.triu_indices_from(M, 1)]
#     print(f"{label:15s} | Min/Mean={values.min()/values.mean():.4f} | Max/Mean={values.max()/values.mean():.4f}")





# fig, ax = plt.subplots(figsize=(7, 3.5))
# ax.scatter(positions[:, 0], positions[:, 1], s=55)

# for a, (x, y) in zip(symbols, positions):
#     ax.annotate(a, (x, y), xytext=(5, 5), textcoords="offset points")

# ax.set(xlabel="Dimension 1", ylabel="Dimension 2", title="Original symbolic geometry")
# ax.set_aspect("equal")
# ax.spines[["top", "right"]].set_visible(False)
# plt.tight_layout()
# plt.show()



 
# for rho in [0, 0.25, 0.5, 0.75, 0.95]:
#     areas = []
#     for a in symbols:
#         t, X, _ = transient_mgdpe([(a, 0, 1)], symbols, D, dt=0.01, J=3, tau0=0.25, rho=rho, sigma=0.7)
#         areas.append(signed_area(X))
#     print(f"rho={rho:.2f} | Areas={np.round(areas, 5)}")



# plot_symbol_excursions(symbols, D)



