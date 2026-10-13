# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def complete_gaps(events, gap="∅"):

    events = sorted([(str(a), float(t), float(d)) for a, t, d in events], key=lambda z: z[1])
    out, end = [], 0.0

    for a, t, d in events:
        if d <= 0 or t < end-1e-10:
            raise ValueError("Durations must be positive and occurrences cannot overlap.")
        if t > end+1e-10:
            out.append((gap, end, t-end))
        out.append((a, t, d))
        end = t+d

    return out


def mds_embedding(D, p=None):

    D = np.asarray(D, dtype=float)
    k = D.shape[0]

    if D.shape != (k, k) or not np.all(np.isfinite(D)) or not np.allclose(D, D.T) or np.any(D < 0) or not np.allclose(np.diag(D), 0):
        raise ValueError("D must be finite, symmetric, non-negative and have a zero diagonal.")

    H = np.eye(k)-np.ones((k, k))/k
    B = -0.5*H @ (D**2) @ H

    values, U = np.linalg.eigh(B)
    order = np.argsort(values)[::-1]
    values, U = values[order], U[:, order]

    r = int(np.sum(values > max(1e-12, 1e-10*max(values[0], 0))))
    if p is not None: r = min(r, int(p))
    if r < 1: raise ValueError("No positive MDS dimensions found.")

    coordinates = U[:, :r]*np.sqrt(values[:r])
    return coordinates, values


def graph_laplacian(D, sigma=None):

    D = np.asarray(D, dtype=float)
    k = D.shape[0]

    positive = D[np.triu_indices(k, 1)]
    positive = positive[positive > 0]

    sigma = float(sigma if sigma is not None else np.median(positive) if len(positive) else 1.0)
    if sigma <= 0: raise ValueError("sigma must be positive.")

    W = np.zeros((k+1, k+1))
    W[:k, :k] = np.exp(-D**2/(2*sigma**2))
    np.fill_diagonal(W, 0)

    return np.diag(W.sum(axis=1))-W


def transient_mgdpe(events, symbols, D, dt=0.01, J=3, tau0=0.18, rho=0.5, sigma=None, tail=None, p=None, gap_scale=None, spectral_mode="multiscale"):

    if dt <= 0 or tau0 <= 0 or J < 1 or not 0 <= rho < 1:
        raise ValueError("Require dt>0, tau0>0, J>=1 and 0<=rho<1.")
    if len(set(symbols)) != len(symbols) or "∅" in symbols:
        raise ValueError("symbols must be unique and exclude the gap symbol.")

    D = np.asarray(D, dtype=float)
    if D.shape != (len(symbols), len(symbols)):
        raise ValueError("D must match the number and order of symbols.")

    events = complete_gaps(events)
    if not events: raise ValueError("At least one event is required.")

    index = {a:i for i, a in enumerate([*symbols, "∅"])}
    if any(a not in index for a, _, _ in events):
        raise ValueError("Unknown symbol in events.")

    # 1. MDS: geometric positions of ordinary symbols.
    coordinates, mds_values = mds_embedding(D, p=p)
    k = len(symbols)

    positive = D[D > 0]
    gap_scale = float(gap_scale if gap_scale is not None else np.median(positive) if len(positive) else 1.0)

    P = np.zeros((k+1, coordinates.shape[1]+1))
    P[:k, :coordinates.shape[1]] = coordinates
    P[-1, -1] = gap_scale

    # 2. Laplacian and multiscale spectral filters.
    L = graph_laplacian(D, sigma)
    eigenvalues, U = np.linalg.eigh(L)

    s0 = np.log(2)/eigenvalues[-1] if eigenvalues[-1] > 1e-12 else 1.0
    scales = s0*2.0**np.arange(J)
    taus = tau0*2.0**np.arange(J)
    weights = np.full(J, 1.0/J)

    
    if spectral_mode == "single":
        scales[:] = s0
    elif spectral_mode == "none":
        rho = 0.0
    elif spectral_mode != "multiscale":
        raise ValueError("spectral_mode must be 'none', 'single' or 'multiscale'.")


    # 3. Temporal discretization.
    T = max(t+d for a, t, d in events)
    tail = 10*taus[-1] if tail is None else float(tail)
    t = np.arange(0, T+tail+dt/2, dt)

    # 4. Smooth excitations in the original symbolic space.
    v = np.zeros((len(t), k+1))

    for a, onset, duration in events:
        u = (t-onset)/duration
        mask = (u >= 0) & (u <= 1)
        v[mask, index[a]] += 1-np.cos(2*np.pi*u[mask])

    # 5. Transient multiscale kernel in the Laplacian eigenbasis.
    kernel = np.zeros((k+1, len(t)))

    for s, tau, w in zip(scales, taus, weights):
        psi = 4*(np.exp(-s*eigenvalues)-np.exp(-2*s*eigenvalues))
        h = np.e*(t/tau)*np.exp(-t/tau)
        kernel += w*(1-rho*psi)[:, None]*h[None, :]

    # 6. Causal convolution in spectral coordinates.
    v_hat = v @ U
    x_hat = np.column_stack([fftconvolve(v_hat[:, r], kernel[r], mode="full")[:len(t)]*dt for r in range(k+1)])

    # 7. Projection of the diffused trajectory into the MDS space.
    X = x_hat @ (U.T @ P)

    config = {"embedding":P, "L":L, "mds_eigenvalues":mds_values, "scales":scales, "taus":taus, "events":events}
    return t, X, config





