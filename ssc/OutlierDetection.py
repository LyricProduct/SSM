import numpy as np


def OutlierDetection(CMat, n):
    _, N = CMat.shape
    OutlierIndx = list()
    FailCnt = 0
    Fail = False

    for i in range(0, N):
        c = CMat[:, i]
        if np.sum(np.isnan(c)) >= 1:
            OutlierIndx.append(i)
            FailCnt += 1
    CMatC = CMat.astype(float)
    CMatC[OutlierIndx, :] = np.nan
    CMatC[:, OutlierIndx] = np.nan
    OutlierIndx = OutlierIndx

    if FailCnt > (N - n):
        CMatC = np.nan
        Fail = True
    return CMatC, OutlierIndx, Fail


if __name__ == "__main__":
    pass
