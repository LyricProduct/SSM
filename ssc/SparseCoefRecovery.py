import numpy as np
import cvxpy as cvx


def SparseCoefRecovery(Xp, cst=0, Opt="Lasso", lmbda=0.001):
    D, N = Xp.shape
    CMat = np.zeros([N, N])
    for i in range(0, N):
        y = Xp[:, i]
        if i == 0:
            Y = Xp[:, i + 1 :]
        elif i > 0 and i < N - 1:
            Y = np.concatenate((Xp[:, 0:i], Xp[:, i + 1 : N]), axis=1)
        else:
            Y = Xp[:, 0 : N - 1]

        if cst == 1:
            if Opt == "Lasso":
                c = cvx.Variable(shape=(N - 1), name="c")
                obj = cvx.Minimize(
                    cvx.square(cvx.norm2(Y @ c - y)) + lmbda * cvx.norm(c, 1)
                )
                constraint = [cvx.sum(c) == 1]
                prob = cvx.Problem(obj, constraint)
                prob.solve(solver=cvx.SCS)
            elif Opt == "L1Perfect":
                c = cvx.Variable(N - 1, 1)
                obj = cvx.Minimize(cvx.norm(c, 1))
                constraint = [Y * c == y, cvx.sum(c) == 1]
                prob = cvx.Problem(obj, constraint)
                prob.solve()
            elif Opt == "L1Noise":
                c = cvx.Variable(N - 1, 1)
                obj = cvx.Minimize(cvx.norm(c, 1))
                constraint = [(Y * c - y) <= lmbda, cvx.sum(c) == 1]
                prob = cvx.Problem(obj, constraint)
                prob.solve()
            elif Opt == "L1ED":
                c = cvx.Variable(N - 1 + D, 1)
                obj = cvx.Minimize(cvx.norm(c, 1))
                constraint = [
                    np.concatenate((Y, np.identity(D)), axis=1) * c == y,
                    cvx.sum(c[0 : N - 1]) == 1,
                ]
                prob = cvx.Problem(obj, constraint)
                prob.solve()
        else:
            if Opt == "Lasso":
                c = cvx.Variable(shape=(N - 1), name="c")
                obj = cvx.Minimize(
                    cvx.square(cvx.norm2(Y @ c - y)) + lmbda * cvx.norm(c, 1)
                )
                prob = cvx.Problem(obj)
                prob.solve(solver=cvx.ECOS)
            elif Opt == "L1Perfect":
                # c = cvx.Variable(N - 1, 1)
                c = cvx.Variable(shape=(N - 1), name="c")
                obj = cvx.Minimize(cvx.norm(c, 1))
                constraint = [Y * c == y]
                prob = cvx.Problem(obj, constraint)
                prob.solve(solver="Clarabel")
            elif Opt == "L1Noise":
                # c = cvx.Variable(N - 1, 1)
                c = cvx.Variable(shape=(N - 1), name="c")
                obj = cvx.Minimize(cvx.norm(c, 1))
                constraint = [(Y * c - y) <= lmbda]
                prob = cvx.Problem(obj, constraint)
                prob.solve(solver="Clarabel")
            elif Opt == "L1ED":
                c = cvx.Variable(N - 1 + D, 1)
                obj = cvx.Minimize(cvx.norm(c, 1))
                constraint = [np.concatenate((Y, np.identity(D)), axis=1) * c == y]
                prob = cvx.Problem(obj, constraint)
                prob.solve()

        if i == 0:
            CMat[0, 0] = 0
            CMat[1:N, 0] = c.value[0 : N - 1]
        elif i > 0 and i < N - 1:
            CMat[0:i, i] = c.value[0:i]
            CMat[i, i] = 0
            CMat[i + 1 : N, i] = c.value[i : N - 1]
        else:
            CMat[0 : N - 1, N - 1] = c.value[0 : N - 1]
            CMat[N - 1, N - 1] = 0
    return CMat


if __name__ == "__main__":
    pass
