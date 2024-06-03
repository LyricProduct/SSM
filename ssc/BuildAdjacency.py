import numpy as np


def BuildAdjacency(CMat, K):
    CMat = CMat.astype(float)
    CKSym = None
    N, _ = CMat.shape
    CAbs = np.absolute(CMat).astype(float)
    for i in range(0, N):
        c = CAbs[:, i]
        PInd = np.flip(np.argsort(c), 0)
        CAbs[:, i] = CAbs[:, i] / float(np.absolute(c[PInd[0]]))
    CSym = np.add(CAbs, CAbs.T).astype(float)
    if K != 0:
        Ind = np.flip(np.argsort(CSym, axis=0), 0)
        CK = np.zeros([N, N]).astype(float)
        for i in range(0, N):
            for j in range(0, K):
                CK[Ind[j, i], i] = CSym[Ind[j, i], i] / float(
                    np.absolute(CSym[Ind[0, i], i])
                )
        CKSym = np.add(CK, CK.T)
    else:
        CKSym = CSym
    return CKSym


if __name__ == "__main__":
    pass
