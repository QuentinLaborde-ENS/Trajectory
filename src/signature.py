# -*- coding: utf-8 -*-

import numpy as np 



def path_signature(X, Q=3):

    d = X.shape[1]
    levels = [np.array(1.0)] + [np.zeros((d,)*q) for q in range(1, Q+1)]

    for dx in np.diff(X, axis=0):
        powers = [np.array(1.0)]

        for q in range(1, Q+1):
            powers.append(np.multiply.outer(powers[-1], dx)/q)

        for q in range(Q, 0, -1):
            levels[q] += sum(np.multiply.outer(levels[q-k], powers[k]) for k in range(1, q+1))

    return levels[1:]


def signature_distance(X1, X2, Q=3, weights=None):

    S1, S2 = path_signature(X1, Q), path_signature(X2, Q)
    weights = np.ones(Q) if weights is None else np.asarray(weights)

    return np.sqrt(sum(w*np.sum((a-b)**2) for w, a, b in zip(weights, S1, S2)))


def signed_area(X):
    return 0.5*np.sum(X[:-1, 0]*X[1:, 1]-X[1:, 0]*X[:-1, 1])




