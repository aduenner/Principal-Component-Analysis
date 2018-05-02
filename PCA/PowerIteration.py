import numpy as np


A=np.random.rand(50,50)
#A=;
m=A.shape[1] #m = number of cols in A

nummults=4
nvals=2

r=np.zeros(nvals)

evsabs = np.absolute(np.linalg.eigvals(A))

def GramSchmidt(V):
	n=V.shape[0]
	k=V.shape[1]
	U = np.zeros((n,k))
	U[:,0] = np.divide(V[:,0],np.sqrt(np.dot(np.transpose(V[:,0]),V[:,0])))
	
	for i in range (1,k):
		U[:,i] = V[:,i]
		for j in range(0,i-1):
			UTUij = np.dot(np.transpose(U[:,i]),U[:,j])
			UTUjj = np.dot(np.transpose(U[:,j]),U[:,j])
			U[:,i] = U[:,i] - np.dot(np.divide(UTUij,UTUjj),U[:,j])
		UTUii = np.dot(np.transpose(U[:,i]),U[:,i])
		U[:,i] = U[:,i]/np.sqrt(UTUii)
	return U
	
for j in range(1,nummults-1):
	for k in range (1,nummults):
		v = np.random.rand(m,1)
		v = v / np.linalg.norm(A)
		w = np.dot(A,v)
		v = w / np.linalg.norm(w) #normalize w and replace v with w
	
		r[j-1] = np.dot(np.transpose(v),np.dot(A,v))
		A = GramSchmidt(A)	
		

evid=np.argsort(evsabs)
evs = evsabs[evid[m-nvals:m]]