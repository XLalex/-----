import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.stats import norm
import time

def compute_sigma(n_steps,T):
    sigma = np.zeros(n_steps)
    dt = T/n_steps
    for i in range(n_steps):
        sigma[i] = 0.11+0.16*i*dt
    return sigma
def simulate_gbm_paths(S0,mu,T,n_steps,n_paths):
    np.random.seed(0)
    Z = np.zeros([n_paths,n_steps+1])
    dt = T/n_steps
    Z[:,0] = np.log(S0)
    times = np.linspace(0,T,n_steps+1)
    epsilon = np.random.randn(n_paths,n_steps)
    sigma = compute_sigma(n_steps,T)
    for i in range(n_steps):
        Z[:,i+1] = Z[:,i]+(mu-1/2*sigma[i]**2)*dt+sigma[i]*np.sqrt(dt)*epsilon[:,i]
    S = np.exp(Z)
    return S,times
def simulate_em_paths(S0,mu,T,n_steps,n_paths):
    S = np.zeros([n_paths,n_steps+1])
    dt = T/n_steps
    S[:,0] = S0
    times = np.linspace(0,T,n_steps+1)
    epsilon = np.random.randn(n_paths,n_steps)
    sigma = compute_sigma(n_steps,T)
    for i in range(n_steps):
        S[:,i+1] = S[:,i]+S[:,i]*(mu*dt+sigma[i]*np.sqrt(dt)*epsilon[:,i])
    return S,times
def compute_sigma_for_option(n_steps,n_paths,T):
    sigma = np.zeros([n_paths,n_steps])
    dt = T/n_steps
    for i in range(n_steps):
        sigma[:,i] = np.sqrt((1/(T-i*dt))*(0.0121*(T-i*dt)+(0.0176/T)*(T**2-(i*dt)**2)+(0.0256/(3*T**2))*(T**3-(i*dt)**3)))
    return sigma
def compute_put_option_price(S,K,T,n_steps,n_paths,r):
    sigma = compute_sigma_for_option(n_steps,n_paths,T)
    dt = T/n_steps
    d1 = np.zeros([n_paths,n_steps])
    d2 = np.zeros([n_paths,n_steps])
    price = np.zeros([n_paths,n_steps+1])
    delta = np.zeros([n_paths,n_steps])
    for i in range(n_steps):
        d1[:,i] = (1/(sigma[:,i]*np.sqrt(T-i*dt)))*(np.log(S[:,i]/K)+(r+sigma[:,i]**2/2)*(T-i*dt))
        d2[:,i] = (1/(sigma[:,i]*np.sqrt(T-i*dt)))*(np.log(S[:,i]/K)+(r-sigma[:,i]**2/2)*(T-i*dt))
        price[:,i] = K*np.exp(-r*(T-i*dt))*norm.cdf(-d2[:,i])-norm.cdf(-d1[:,i])*S[:,i]
        delta[:,i] = norm.cdf(d1[:,i])-1
    for j in range(n_paths):
        price[j,-1] = max(0,K-S[j,-1])
    return price,delta
def compute_b(S0,mu,K1,K2,T,n_steps,n_paths,r,theta,b0,q1,q2,q3):
    S,times = simulate_gbm_paths(S0,mu,T,n_steps,n_paths)
    put_price1,delta1 = compute_put_option_price(S,K1,T,n_steps,n_paths,r)
    put_price2,delta2 = compute_put_option_price(S,K2,T,n_steps,n_paths,r)
    dt = T/n_steps
    b = np.zeros([n_paths,n_steps+1])
    b[:,0] = b0
    a = np.zeros([n_paths,n_steps])
    a[:,0] = b[:,0]-(q1+q2*(delta1[:,0]+delta2[:,0]))*S[:,0]-q3*(put_price1[:,0]+put_price2[:,0])-abs(q1+q2*(delta1[:,0]+delta2[:,0]))*theta*S[:,0]
    b[:,1] = a[:,0]*np.exp(r*dt)+(q1+q2*(delta1[:,0]+delta2[:,0]))*S[:,1]+q3*(put_price1[:,1]+put_price2[:,1])-abs(q1+q2*(delta1[:,0]+delta2[:,0]))*theta*S[:,0]
    for i in range(1,n_steps):
        a[:,i] = b[:,i]-(q1+q2*(delta1[:,i]+delta2[:,i]))*S[:,i]-q3*(put_price1[:,i]+put_price2[:,i])-abs(q2*(delta1[:,i]+delta2[:,i]-delta1[:,i-1]-delta2[:,i-1]))*theta*S[:,i]
        b[:,i+1] = a[:,i]*np.exp(r*dt)+(q1+q2*(delta1[:,i]+delta2[:,i]))*S[:,i+1]+q3*(put_price1[:,i+1]+put_price2[:,i+1])-abs(q2*(delta1[:,i]+delta2[:,i]-delta1[:,i-1]-delta2[:,i-1]))*theta*S[:,i]
    return b
def compute_expected_utility(S0,mu,K1,K2,T,n_steps,n_paths,r,theta,b0,q1,q2,q3,lamb):
    b = compute_b(S0,mu,K1,K2,T,n_steps,n_paths,r,theta,b0,q1,q2,q3)
    n_paths = b.shape[0]
    b0 = b[0,0]
    utility = -1*np.exp(-lamb*b[:,-1])
    expected_utility = (1/n_paths)*np.sum(utility)
    return expected_utility
def objective(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    return (1/1.41)*np.log(-1*compute_expected_utility(S0,mu,K1,K2,T,n_steps,n_paths,r,theta,b0,q1,q2,q3,lamb))
def constraint_function(q):
    return b0-(q[0]+q[1]*delta0)*S0-q[2]*P0-abs(q[0]+q[1]*delta0)*theta*S0
constraintDic = {'type':'ineq','fun':constraint_function}

K1 = 167
K2 = 115
b0 = 141.00
S0 = 137.00
mu = 0.10
r = 0.02
T = 1.70
theta = 0.01
n_steps = 32
n_paths = 100000
lamb = 0.01

#compute the put option prices at time 0
sigma0 =np.sqrt((1/T)*(0.11*T+(0.08/T)*T**2))
d1_1 = (1/(sigma0*np.sqrt(T)))*(np.log(S0/K1)+(r+sigma0**2/2)*T)
d2_1 = (1/(sigma0*np.sqrt(T)))*(np.log(S0/K1)+(r-sigma0**2/2)*T)
d1_2 = (1/(sigma0*np.sqrt(T)))*(np.log(S0/K2)+(r+sigma0**2/2)*T)
d2_2 = (1/(sigma0*np.sqrt(T)))*(np.log(S0/K2)+(r-sigma0**2/2)*T)
delta0 = (norm.cdf(d1_1)-1)+(norm.cdf(d1_2)-1)
P0 = (K1*np.exp(-r*T)*norm.cdf(-d2_1)-norm.cdf(-d1_1)*S0)+(K2*np.exp(-r*T)*norm.cdf(-d2_2)-norm.cdf(-d1_2)*S0)

# phi1 = (-1/1.41)*np.log(-1*compute_expected_utility(S0,mu,K1,K2,T,n_steps,n_paths,r,theta,b0,0,0,0,lamb))
# phi2 = (-1/1.41)*np.log(-1*compute_expected_utility(S0,mu,K1,K2,T,n_steps,n_paths,r,theta,b0,1,0,0,lamb))
# phi3 = (-1/1.41)*np.log(-1*compute_expected_utility(S0,mu,K1,K2,T,n_steps,n_paths,r,theta,b0,0,1,0,lamb))
# phi4 = (-1/1.41)*np.log(-1*compute_expected_utility(S0,mu,K1,K2,T,n_steps,n_paths,r,theta,b0,0,0,1,lamb))
# print(phi1,phi2,phi3,phi4)

#test optimization
res = scipy.optimize.minimize(fun=objective,x0=[0,0,0],constraints=[constraintDic],method='SLSQP')
print(res)

