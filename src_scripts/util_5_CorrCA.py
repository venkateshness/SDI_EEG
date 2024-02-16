import numpy as np
from scipy.linalg import eigh
from timeit import default_timer

# Shoutout and a big thanks to https://github.com/pa-ak/ISC-Inter-Subject-Correlations
def train_cca(data):
    """Run Correlated Component Analysis on your training data.
    Parameters:
    ----------
    data : dict
        Dictionary with keys are names of conditions and values are numpy
        arrays structured like (subjects, channels, samples).
        The number of channels must be the same between all conditions!
    Returns:
    -------
    W : np.array
        Columns are spatial filters. They are sorted in descending order, it means that first column-vector maximize
        correlation the most.
    ISC : np.array
        Inter-subject correlation sorted in descending order
    """

    start = default_timer()

    C = len(data.keys())
    print(f"train_cca - calculations started. There are {C} conditions")

    gamma = 0.1
    Rw, Rb = 0, 0
    for cond in data.values():
        (
            N,
            D,
            T,
        ) = cond.shape
        print(f"Condition has {N} subjects, {D} sensors and {T} samples")
        cond = cond.reshape(D * N, T)

        # Rij
        Rij = np.swapaxes(np.reshape(np.cov(cond), (N, D, N, D)), 1, 2)

        # Rw
        Rw = Rw + np.mean([Rij[i, i, :, :] for i in range(0, N)], axis=0)

        # Rb
        Rb = Rb + np.mean(
            [Rij[i, j, :, :] for i in range(0, N) for j in range(0, N) if i != j],
            axis=0,
        )
    # Divide by number of condition
    Rw, Rb = Rw / C, Rb / C

    # Regularization
    Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # ISCs and Ws
    [ISC, W] = eigh(Rb, Rw_reg)  # Eigen values, W matrix

    # Make descending order
    ISC, W = ISC[::-1], W[:, ::-1]
    # print(ISC[0])
    stop = default_timer()

    print(f"Elapsed time: {round(stop - start)} seconds.")
    return W, ISC


def apply_cca(X, W, fs):
    """Applying precomputed spatial filters to your data.
    Parameters:
    ----------
    X : ndarray
        3-D numpy array structured like (subject, channel, sample)
    W : ndarray
        Spatial filters.
    fs : int
        Frequency sampling.
    Returns:
    -------
    ISC : ndarray
        Inter-subject correlations values are sorted in descending order.
    ISC_persecond : ndarray
        Inter-subject correlations values per second where first row is the most correlated.
    ISC_bysubject : ndarray
        Description goes here.
    A : ndarray
        Scalp projections of ISC.
    """

    start = default_timer()
    print("apply_cca - calculations started")

    N, D, T = X.shape
    # gamma = 0.1
    window_sec = 8
    X = X.reshape(D * N, T)

    # Rij
    Rij = np.swapaxes(np.reshape(np.cov(X), (N, D, N, D)), 1, 2)

    # Rw
    Rw = np.mean([Rij[i, i, :, :] for i in range(0, N)], axis=0)
    # Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # Rb
    Rb = np.mean(
        [Rij[i, j, :, :] for i in range(0, N) for j in range(0, N) if i != j], axis=0
    )

    # ISCs
    ISC = np.sort(
        np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)
    )[::-1]

    # Scalp projections
    A = np.linalg.solve(Rw @ W, np.transpose(W) @ Rw @ W)  # a, b.
    ISC_persecond = np.empty((D, int(T / fs)))
    window_i = 0

    for t in range(0, T, fs):

        Xt = X[:, t : t + window_sec * fs]  # [subj 1, subj 2........subj 10]

        # the covariance
        Rij = np.cov(Xt)  # 910, 910for all the subjects
        # <----10 subjects---->
        #  [sub1, sub2... sub10 ] sub 1
        #  [sub1, sub2... sub10 ] sub 2
        #  [sub1, sub2... sub10 ] ..
        #  [sub1, sub2... sub10 ] ..
        #   [sub1, sub2... sub10 ] sub 10

        Rw = np.mean(
            [
                Rij[i : i + D, i : i + D]  # Correlation diagonally (itself)
                for i in range(0, D * N, D)
            ],
            axis=0,
        )

        Rb = np.mean(
            [
                Rij[i : i + D, j : j + D]
                for i in range(0, D * N, D)
                for j in range(0, D * N, D)
                if i != j
            ],
            axis=0,
        )  # Correlation with other subjects

        ISC_persecond[:, window_i] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(
            np.transpose(W) @ Rw @ W
        )
        window_i += 1

    stop = default_timer()
    # print(f'Elapsed time: {round(stop - start)} seconds.')

    return ISC, ISC_persecond, A  # , A