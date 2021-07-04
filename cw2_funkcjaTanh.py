# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:03:34 2020

@author: Franka
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Podział danych ===============================================
data = np.loadtxt('Dane\dane5.txt')
X =data[:,0]
Y = data[:,1]
# X = np.linspace(-2, 2, np.linalg.norm(-2 - 2)/0.1+1)
# Y = X**2 + 1*(np.random.rand(len(X))-0.5)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=4)

P=np.array(X_train)
T=np.array(y_train)

# Uczenie sieci ================================================= 
S1 = 20 # ILOSC NEURONOW W WARSTWIE UKRYTEJ (siec 1 warstwowa)
W1 = (np.random.rand(S1, 1)- 0.5) # inicjalizacja wag
B1 = (np.random.rand(S1, 1)- 0.5)
W2 = (np.random.rand(1, S1)- 0.5)
B2 = np.random.rand(1,1) -0.5
lr = 0.001

for epoka in range(2000):
    # odpowiedz sieci (feedforward)
    Q = np.matmul(W1,np.array(P, ndmin=2)) + np.matmul(B1, np.array(np.ones(P.shape), ndmin=2))
    A1 = np.tanh(Q)
    A2 = np.matmul(W2,A1) + B2  
    
    # propagacja wsteczna (backpropagation)
    E2 = T - A2;
    E1 = np.matmul(W2.T,E2)
    
    # reguła delta - obliczanie (error * pochodna)
    dW2 = lr* np.matmul(E2, A1.T)   
    dB2 = lr * np.matmul(E2, np.ones(E2.shape).T)
    dW1 = lr * np.matmul(np.multiply((1 - np.multiply(A1,A1)), E1), P.T)
    dB1 = lr * np.matmul(np.multiply((1 - np.multiply(A1,A1)), E1), np.ones(len(P)).T)
    
    # update wag
    W2 = W2 + dW2 
    B2 = B2 + dB2
    W1 = W1+np.array(dW1,ndmin=2).T 
    B1 = B1 + np.array(dB1,ndmin=2).T
    
    # wykres 
    if np.mod(epoka, 10)==0 : 
        # P.sort()
        plt.plot(P, T, 'r*')
        plt.plot(P,A2.T,'b_')
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        
plt.show()
 
# ocena błedu aproksymacji dla danych treningowych
print("Error train: ",np.sum(np.fabs(E2))/len(T))


# OCENA ZGODNOSCI ==================================================
P=X_test
T=y_test
# feedforward
A1 = np.tanh(np.matmul(W1,np.array(P, ndmin=2)) + np.matmul(B1, np.array(np.ones(P.shape), ndmin=2)))
A2 = np.matmul(W2,A1) + B2
# propagacja wsteczna - liczenie błędu
E2 = T - A2;
E1 = np.matmul(W2.T,E2)

print("error test: ",np.sum(np.fabs(E2))/len(T))


plt.plot(P, T, 'r*')
plt.plot(P,A2.T, 'b_')
plt.show()
    

