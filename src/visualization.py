# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np 

from src.embedding import transient_mgdpe



def plot_paths_3d(paths, labels=None, pcs=(0, 1, 3)):

    labels = labels or [f"Sequence {i+1}" for i in range(len(paths))]
    all_points = np.vstack(paths)
    center = all_points.mean(axis=0)
    _, S, Vt = np.linalg.svd(all_points-center, full_matrices=False)

    basis = Vt[list(pcs)].T
    explained = S**2 / np.sum(S**2)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i, (X, label) in enumerate(zip(paths, labels)):
        P = (X-center) @ basis
        line, = ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=1.5, linestyle="--" if i == 0 else "-", label=label)
        ax.scatter(*P[0], color=line.get_color(), s=35, marker="o")
        ax.scatter(*P[-1], color=line.get_color(), s=45, marker="x")

    for axis, pc in zip([ax.set_xlabel, ax.set_ylabel, ax.set_zlabel], pcs):
        axis(f"PC{pc+1} ({explained[pc]*100:.1f}%)")

    ax.set_title("Transient M-GDPE trajectories")
    ax.view_init(elev=25, azim=-55)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return basis, center





def plot_distance(paths, t, onsets):
      
    fig, ax = plt.subplots(figsize=(10, 4))
    
    for name, X in paths.items():
        if name == "Reference": continue
        distance = np.linalg.norm(paths["Reference"] - X, axis=1)
        ax.plot(t, distance, linewidth=1.5, label=name)
        print(f"{name:20s} max={distance.max():.4f}   integral={np.trapz(distance, t):.4f}")
    
    ax.axvspan(onsets[7], onsets[9], color="gray", alpha=0.10, label="Modified region")
    ax.set(xlabel="Time", ylabel=r"$\|X_{\mathrm{ref}}(t)-X(t)\|_2$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.show()




def plot_symbol_excursions(symbols, D, rhos=(0, 0.5, 0.95), duration=1.0, equal_aspect=False):

    fig, axes = plt.subplots(1, len(rhos), figsize=(5*len(rhos), 5.5))
    axes = np.atleast_1d(axes)
    colors = plt.cm.tab10(np.linspace(0, 1, len(symbols)))

    for ax, rho in zip(axes, rhos):
        for a, color in zip(symbols, colors):
            t, X, _ = transient_mgdpe([(a, 0, duration)], symbols, D, dt=0.01, J=3, tau0=0.25, rho=rho, sigma=0.7)
            ax.plot(X[:, 0], X[:, 1], color=color, linewidth=1.5, label=a)
            k = np.argmax(np.linalg.norm(X[:, :2], axis=1))
            ax.scatter(X[k, 0], X[k, 1], color=color, s=15)

        ax.scatter(0, 0, color="black", marker="+", s=40)
        ax.set_title(rf"$\rho={rho}$")
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.set_aspect("equal" if equal_aspect else "auto")
        ax.set_box_aspect(1)
        ax.margins(0.12)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].legend(title="Symbol", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


