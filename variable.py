from math import sqrt, log2
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0

H = 10
V = 20
T = 20
S = 40
P_WPT = math.pow(10, 7) 


delta_t = 0.5
d = V*delta_t
X = 20
Y = 10
w_s = [5, 0]
w_d = [15, 0]
q_I = [0, 10]
q_F = [20, 10]
N = int(T/delta_t)
N1 = N + 1

delta = 0.012
s = 0.05
A = 0.8
omega = 100
R = 0.08
I = 0.1
rho = 1.225
W = 0.5
d_0 = 0.0151/s/A
e = math.e
E = 0.5772156649
alpha = 2.2
a2 = alpha/2
B = 20
omega_0 = 10**(-3)

sigma = 0.7
n_u = 0.5
P_u = 5  # 7.5 mW
P_b = 10**(-3)
P_s = 10**(1.6)  # 16dBm
sigma_exp_2 = 10**(-6)
eta = 0.5

miu = 0.84

v_0 = sqrt(W / (2 * rho * A))

P_0 = delta / 8 * rho * s * A * omega**3 * R ** 3
k_1 = 3 / omega**2 / R**2
P_1 = (1 + I) * W**1.5 / sqrt(2*rho*A)
k_2 = 1 / 2 / v_0**2
k_3 = 0.5 * d_0 * rho * s * A
P_u_bar  = P_u*(1 + math.ceil(sigma))


#New model
fc = 2.4*(10**9) #2.4 GHz
c = 3*(10**8)    # van toc anh sang - 3.10^8 m/s
Tb = 0.1*(10**-3) # Time sampling
fd = V*fc/c
pi = math.pi

Z_sqrt = j0([2*pi*fd*Tb])
Z = Z_sqrt**2
Z = Z.tolist()[0]
Z2 = Z**2


#Z = 1 case
#Z = 1
#Z2 = 1
#print(Z2)
# denote that sigma u and sigma g = sigma d
sigma_u2 = 10**(-6)
sigma_g2 = 10**(-6)
theta = e**(-E) * omega_0 / (sigma_exp_2 + (1-Z2)**2*sigma_g2**2 + (1-Z2)*sigma_g2) # bo sung Z vao theta, thay doi cong thuc rate