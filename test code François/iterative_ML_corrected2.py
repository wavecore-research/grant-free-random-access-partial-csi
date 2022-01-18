import matplotlib.pyplot as plt
import numpy as np
#import inputs
#import pickle
#from run_cg import run


plt.close('all')


def ML_value(gamma_hat):
    r_out=M*np.log(np.linalg.det(C_inverse)) - M*T*np.log(np.pi) - np.trace((y - s @ np.diag(gamma_hat[:,0]) @ g ).T.conj() @ C_inverse @ (y - s @ np.diag(gamma_hat[:,0]) @ g ))
    return np.real(r_out)

K=40 # Number of single-antenna users
M=64 # Number of receive antennas
T=10 # Preamble length
p_TX=1
SNR_dB=20
SNR=10**(SNR_dB/10)

## Preamble generation and user activity
s=np.random.normal(0,1/np.sqrt(2),(T,K))+1j*np.random.normal(0,1/np.sqrt(2),(T,K))
a=np.random.binomial(n=1,p=0.5,size=(K,1))
phi=np.random.uniform(0,2*np.pi,size=(K,1))
rho=np.ones((K,1))*p_TX
gamma= np.sqrt(rho) * a * np.exp(1j*phi)

## Channel generation
#lambda_k=np.random.uniform(0,1,(K,1))
lambda_k=np.zeros((K,1))+0.2
lambda_compl_k=np.sqrt(1-lambda_k**2)
#lambda_compl_k=np.ones((K,1))*0.7

g=np.diag(lambda_compl_k[:,0])@(np.random.normal(0,1/np.sqrt(2),(K,M))+1j*np.random.normal(0,1/np.sqrt(2),(K,M)))
epsilon=np.random.normal(0,1/np.sqrt(2),(K,M))+1j*np.random.normal(0,1/np.sqrt(2),(K,M))

h=g+np.diag(lambda_k[:,0]) @ epsilon

## Received preamble
sigma2=p_TX/SNR
w=(np.random.normal(0,1/np.sqrt(2),(T,M))+1j*np.random.normal(0,1/np.sqrt(2),(T,M)))*np.sqrt(sigma2)
y= s @ np.diag(gamma[:,0]) @ h + w



# Estimate based on prior csi, assuming all lambdas are zeros, cf. conf. paper
y_tilde=np.reshape(y.T,(M*T,1))
D=np.diag(s.reshape(K*T)) @ np.kron(np.ones((T,1)), np.identity(K))
Gamma=np.zeros((M*T,K),dtype=complex)
for index_m in range(M):
    Gamma[0+index_m*T:T+index_m*T,:]= s @ np.diag(g[:,index_m])
gamma_hat_prior_CSI=np.linalg.inv(Gamma.conj().T @ Gamma) @ Gamma.conj().T @ y_tilde



## Estimator based on partial CSI and iterative ML
not_converged=1
iter_number=0
iter_max=K*10

root_store_partial=np.zeros((iter_max,3),dtype=complex)
determinant=np.zeros(iter_max,dtype=complex)

#gamma_hat_0=np.zeros((K,1),dtype=complex) # Initialization to zero
gamma_hat_0=gamma_hat_prior_CSI #♣ Initialization thanks to prior CSI

gamma_hat=gamma_hat_0.copy()
k_prime=0
C_inverse=np.linalg.inv( s @ np.diag(abs(gamma_hat[:,0])**2*lambda_k[:,0]**2) @ s.T.conj() +sigma2*np.identity(T))
#print('Initial value of cost function: '+ str(ML_value(gamma_hat)))

while not_converged:
    temp=gamma_hat[:,0]
    temp[k_prime]=0
    y_m_k_prime= y - s @ np.diag(temp) @ g
    C_minus_k_prime_inverse=np.linalg.inv( s @ np.diag(abs(temp)**2*lambda_k[:,0]**2) @ s.T.conj() +sigma2*np.identity(T))
    
    temp=s[:,k_prime].T.conj() @ C_minus_k_prime_inverse @ s[:,k_prime]
    temp2 = y_m_k_prime.T.conj() @ C_minus_k_prime_inverse @ s[:,k_prime]
    alpha=float(np.real(lambda_k[k_prime]**2 * np.sum( abs(temp2)**2 ) - temp * np.sum(abs(g[k_prime,:])**2)))
    beta=float(2 * abs(g[k_prime,:] @ temp2))
    delta=float(np.real(temp  * lambda_k[k_prime]**2))
    
    # Amplitude optimization
    a=-2*M*delta**2
    b=-beta*delta
    c=-2*M*delta+2*alpha
    d=beta
    root_store_partial[iter_number,:]=np.roots([a, b, c, d])
    # check for best solution being positive and real
    sol = -1
    for r in root_store_partial[iter_number,:]:
        if np.isreal(r) and r>=0:
            sol = r if r > sol else sol
    if sol == -1:
        raise ValueError("No solution for poly. found.")
    
    r_k_prime_hat=sol
#    r_k_prime_hat =np.max(np.real(root_store[iter_number,:]))
    
    # Phase optimization
    phi_k_prime_hat=np.angle(s[:,k_prime].T.conj() @ C_inverse @ (y_m_k_prime @ g[k_prime,:].T.conj()) )
    gamma_hat_k_prime=r_k_prime_hat*np.exp(1j*phi_k_prime_hat)
    
    gamma_hat[k_prime]=gamma_hat_k_prime
    
#    C_inverse=C_minus_k_prime_inverse - (C_minus_k_prime_inverse_s_k_prime @ C_minus_k_prime_inverse_s_k_prime.conj().T) * (r_k_prime_hat**2*lambda_k[k_prime]**2)/(1+  s[k_prime,:].conj() @ C_minus_k_prime_inverse_s_k_prime * r_k_prime_hat**2*lambda_k[k_prime]**2)    
    C_inverse=np.linalg.inv( s @ np.diag(abs(gamma_hat[:,0])**2*lambda_k[:,0]**2) @ s.T.conj() +sigma2*np.identity(T))
    k_prime=np.mod(k_prime+1,K)
    
#    print('Iteration number: ' + str(iter_number) + ', value of cost function: '+ str(ML_value(gamma_hat)))
    iter_number += 1  
    if (iter_number > iter_max-1):
        not_converged=0
print('Value of cost function just using partial CSI: '+ str(ML_value(gamma_hat)))
gamma_hat_partial_CSI=gamma_hat.copy()

C_inverse=np.linalg.inv( s @ np.diag(abs(gamma_hat_prior_CSI[:,0])**2*lambda_k[:,0]**2) @ s.T.conj() +sigma2*np.identity(T))
print('Value of cost function just using only prior CSI: '+ str(ML_value(gamma_hat_prior_CSI)))



# Estimator based on no CSI and iterative ML (as Caire)
not_converged=1
iter_number=0
g=g*0
gamma_hat_0=np.zeros((K,1),dtype=complex) # Initialization based on 


lambda_k=np.ones((K,1))
gamma_hat=gamma_hat_0.copy()
k_prime=0
C_inverse=np.linalg.inv( s @ np.diag(abs(gamma_hat[:,0])**2*lambda_k[:,0]**2) @ s.T.conj() +sigma2*np.identity(T))
#print('Initial value of cost function: '+ str(ML_value(gamma_hat)))

root_store=np.zeros((iter_max,3),dtype=complex)
while not_converged:
    temp=gamma_hat[:,0]
    temp[k_prime]=0
    y_m_k_prime= y
    C_minus_k_prime_inverse=np.linalg.inv( s @ np.diag(abs(temp)**2*lambda_k[:,0]**2) @ s.T.conj() +sigma2*np.identity(T))
    
    temp=s[:,k_prime].T.conj() @ C_minus_k_prime_inverse @ s[:,k_prime]
    temp2 = y_m_k_prime.T.conj() @ C_minus_k_prime_inverse @ s[:,k_prime]
    alpha=float(np.real(lambda_k[k_prime]**2 * np.sum( abs(temp2)**2 )))
    beta=float(0)
    delta=float(np.real(temp  * lambda_k[k_prime]**2))
    
    # Amplitude optimization
    a=-2*M*delta**2
    b=-beta*delta
    c=-2*M*delta+2*alpha
    d=beta
    root_store[iter_number,:]=np.roots([a, b, c, d])
    # check for best solution being positive and real
    sol = -1
    for r in root_store[iter_number,:]:
        if np.isreal(r) and r>=0:
            sol = r if r > sol else sol
    if sol == -1:
        raise ValueError("No solution for poly. found.")
    
    r_k_prime_hat=sol
#    r_k_prime_hat =np.max(np.real(root_store[iter_number,:]))
    
    # Phase optimization
    phi_k_prime_hat=np.angle(s[:,k_prime].T.conj() @ C_inverse @ (y_m_k_prime @ g[k_prime,:].T.conj()) )
    gamma_hat_k_prime=r_k_prime_hat*np.exp(1j*phi_k_prime_hat)
    
    gamma_hat[k_prime]=gamma_hat_k_prime
    
#    C_inverse=C_minus_k_prime_inverse - (C_minus_k_prime_inverse_s_k_prime @ C_minus_k_prime_inverse_s_k_prime.conj().T) * (r_k_prime_hat**2*lambda_k[k_prime]**2)/(1+  s[k_prime,:].conj() @ C_minus_k_prime_inverse_s_k_prime * r_k_prime_hat**2*lambda_k[k_prime]**2)    
    C_inverse=np.linalg.inv( s @ np.diag(abs(gamma_hat[:,0])**2*lambda_k[:,0]**2) @ s.T.conj() +sigma2*np.identity(T))
    k_prime=np.mod(k_prime+1,K)
    
#    print('Iteration number: ' + str(iter_number) + ', value of cost function: '+ str(ML_value(gamma_hat)))
    iter_number += 1  
    if (iter_number > iter_max-1):
        not_converged=0
print('Value of cost function using no CSI: '+ str(ML_value(gamma_hat)))
gamma_hat_no_CSI=gamma_hat.copy()

plt.figure()
plt.subplot(4, 1, 1)
plt.stem(np.abs(gamma),use_line_collection=True)
plt.title('gamma')
plt.subplot(4, 1, 2)
plt.stem(np.abs(gamma_hat_prior_CSI),use_line_collection=True)
plt.title('gamma_hat_prior_CSI')
plt.subplot(4, 1, 3)
plt.stem(np.abs(gamma_hat_partial_CSI),use_line_collection=True)
plt.title('gamma_hat_partial_CSI')
plt.subplot(4, 1, 4)
plt.stem(np.abs(gamma_hat_no_CSI),use_line_collection=True)
plt.title('gamma_hat_no_CSI')


print('MSE just using prior CSI: '+ str(10*np.log10(np.average(abs(abs(gamma)-abs(gamma_hat_prior_CSI))**2))))
print('MSE using partial CSI: '+ str(10*np.log10(np.average(abs(abs(gamma)-abs(gamma_hat_partial_CSI))**2))))
print('MSE using no CSI: '+ str(10*np.log10(np.average(abs(abs(gamma)-abs(gamma_hat_no_CSI))**2))))

