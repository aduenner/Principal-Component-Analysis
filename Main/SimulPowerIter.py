__author__ = "Andrew Duenner"
__version__ = "1.0.1"
__maintainer__ = "Andrew Duenner"
__status__ = "VDevelopment"
__license__ = "MIT"

"""SimulPowerIter.py: Iterative Method for Calculating Multiple Eigenvalues of Matrix

Sources:
	Stopping Criterion:
		Arbenz, P. (2012). Numerical Methods for Solving Large Scale Eigenvalue
		Problems. Unpublished Manuscript, Computer Science Department,
		Eidgenössische Technische Hochschule Zürich, Zürich, Switzerland.
		http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter8.pdf

	Modified Gram Schmidt:
		Verschelde, J. (2013). MCS 507 Mathematical, Statistical and Scientific Software.
		Unpublished Manuscript, Department of Mathematics, Statistics and Computer Science
		University of Illinois at Chicago, Chicago, Illinois.
		http://homepages.math.uic.edu/~jan/mcs507f13/

	Simultaneous Iteration:
		Trefethen, Lloyd N., and David Bau III. Numerical linear algebra. Vol. 50. Siam, 1997.
"""

import numpy as np

def MGS(A):
    """
    Perform QR Factorization on A using Modified Gram Schmidt, Return Q and R
    Parameters:
        A - numpy.Array(m,n)
    Outputs:
        Q - numpy.Array(m,n) A=Q*R
        R - numpy.Array(n,n) Upper Triangular R such that A=Q*R
    """
    m=A.shape[0]
    n=A.shape[1] #n is number of columns in A
    Q=np.zeros((m,n))
    R=np.zeros((n,n))
    for k in range(0,n):
        R[k,k]=np.sqrt(np.dot(A[:, k], A[:, k]))
        Q[:,k]=A[:,k]/R[k,k]
        for j in range (k+1,n):
            R[k,j] = np.dot(np.matrix.getH(Q[:,k]),A[:,j])
            A[:,j]=A[:,j]-np.dot(R[k,j],Q[:,k])
    return Q,R

def SimulIter(A,neigs=1,maxiters=100,tol=1e-6):
    """
    Perform simultaneous power iteration on A and return eigenvectors Q,
    eigenvalues R, residual ErrOut
    Parameters:
        A        - Array(m,n) - Input matrix
        neigs    - Int64      - Number of Eigenvalues to calculate (1)
        maxiters - Int        - Maximum number of iterations (100)
        tol      - Double     - Tolerance threshold to end for loop early (1e-6)
    Outputs:
        Q    - Array(m,neigs) - Eigenvectors of the (neigs)th largest eigenvalues
        R    - Array(neigs,)  - Eigenvalues

"""
    Q=np.random.rand(A.shape[0],neigs)
    ErrOut=np.zeros((maxiters,))
    for k in range(maxiters):
        Z=A@Q
        Qold=Q
        Q,R=MGS(Z)

        Err=(np.eye(A.shape[0])-np.dot(Q,np.matrix.getH(Q)))@Qold
        Err=np.linalg.norm(Err)
        ErrOut[k]=Err
        if Err<=tol:
            break
    return Q,np.diag(R),ErrOut
