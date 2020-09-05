# Optimal Orthogonal Vectors
This function computes an orthogonal set of vectors which have the minimum of Euclidean distance squared measure. 
The problem is posed as a convex optimization routine using Schur complement, and the fact that an optimal solution 
to a linear SDP lies on the boundary. 

# Use
getNearestOthogonalVectors(A=np.array([[-0.36404949, -0.73010094, -0.29211371],
                                            [-0.18887306, -1.4690241 ,  0.52940094],
                                            [ 0.04957185,  1.67519598, -1.60157291]]),
                               solver='CVXOPT')
                               
 # Package Requirements
 1) Numpy
 2) CVXPY
 3) MatPlotLib

