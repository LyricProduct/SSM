import numpy as np
import math


def DataProjection(X, r, type="NormalProj"):
    Xp = None
    D, N = X.shape
    if r == 0:
        Xp = X
    else:
        if type == "PCA":
            isEcon = False
            if D > N:
                isEcon = True
            U, S, V = np.linalg.svd(X.T, full_matrices=isEcon)
            Xp = U[:, 0:r].T
        if type == "NormalProj":
            normP = (1.0 / math.sqrt(r)) * np.random.randn(r * D, 1)
            PrN = normP.reshape(r, D, order="F")
            Xp = np.matmul(PrN, X)
        if type == "BernoulliProj":
            bp = np.random.rand(r * D, 1)
            Bp = (1.0 / math.sqrt(r)) * (bp >= 0.5) - (1.0 / math.sqrt(r)) * (bp < 0.5)
            PrB = Bp.reshape(r, D, order="F")
            Xp = np.matmul(PrB, X)
    return Xp


if __name__ == "__main__":
    pass
