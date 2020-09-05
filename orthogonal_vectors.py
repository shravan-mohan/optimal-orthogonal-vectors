import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

def checkDegeneracy(A):
    """
    This function checks if the set of vectors has a zero vector,
    or two vectors which are linearly dependent.
    :param A: Set of vectors horizontally stacked.
    :return: True or False
    """
    for k in range(A.shape[1]):
        if (np.linalg.norm(A[:, k]) <= 1e-6):
            print('Do not include zero vectors!')
            return True
    for k in range(A.shape[1]-1):
        for l in range(k+1,A.shape[1]):
            for m in range(A.shape[0]):
                if(A[m,l]!=0):
                    if(np.linalg.norm(A[:,k] - (A[m,k]/A[m,l])*A[:,l])<=1e-50):
                        return True
                    else:
                        break
                else:
                    continue
    return False


def getNearestOthogonalVectors(A=np.array([[-0.36404949, -0.73010094, -0.29211371],
                                            [-0.18887306, -1.4690241 ,  0.52940094],
                                            [ 0.04957185,  1.67519598, -1.60157291]]),
                               solver='CVXOPT'):

    """
    This function computes an orthogonal set of vectors which have the
    minimum of Euclidean distance squared measure.
    :param A: Horizontal stacking of input vectors
    :param solver: One of the CVXPY solvers. Default is set to 'CVXOPT'.
    :return: Optimally adjusted orthogonal vectors (horizontally
    stacked into  a matrix). Also generates a plot in case of 3 3D vectors
    being the input.
    """

    if(A.shape[0]<A.shape[1]):
        print('Enter a square or a tall matrix only!')
        return -1
    if(checkDegeneracy(A)):
        print('You have entered a matrix either with two vectors being linearly '
              'dependent or has a zero vector!')
        return -1

    Y = A.T
    Q = cvx.Variable((A.shape[0], A.shape[1]))
    D = cvx.Variable(A.shape[1])
    cf = 2 * cvx.trace(Y @ Q) - cvx.sum(D)
    constraints = [cvx.vstack((cvx.hstack((cvx.diag(D), Q.T)), cvx.hstack((Q, np.eye(A.shape[0]))))) >> 0]
    prob = cvx.Problem(cvx.Maximize(cf), constraints)
    prob.solve(solver=solver)
    print(prob.status)

    if (prob.status == 'infeasible'):
        print('Something went wrong with the solver!')
        return -1
    elif (prob.status == 'optimal_inaccurate'):
        print('The solution is numerically inaccurate. Try using another solver!')

    if(A.shape[0]==3 and A.shape[1]==3):

        p0 = A[:,0]
        p1 = A[:,1]
        p2 = A[:,2]

        origin = [0,0,0]
        X, Y, Z = zip(origin,origin,origin)
        U, V, W = zip(p0,p1,p2)

        fig = plt.figure()
        plt.xlim(-2*np.max(np.abs(U)),2*np.max(np.abs(U)))
        plt.ylim(-2*np.max(np.abs(U)),2*np.max(np.abs(U)))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-np.max(np.abs(U)),np.max(np.abs(U)))
        ax.set_ylim(-np.max(np.abs(V)),np.max(np.abs(V)))
        ax.set_zlim(-np.max(np.abs(W)),np.max(np.abs(W)))
        ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.1)
        plt.show()

        p0 = list(Q.value[:,0])
        p1 = list(Q.value[:,1])
        p2 = list(Q.value[:,2])

        origin = [0,0,0]
        X, Y, Z = zip(origin,origin,origin)
        U, V, W = zip(p0,p1,p2)

        ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.1,color='r')

    return Q.value