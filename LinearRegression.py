#%% klasa

class LinearRegression:
    
    #w = []
    #X_mean = 0
    #X_std = 1
    
    def fit(self, X, y, iteration: int=100, alpha: float=0.0005,
            min_grad: float=0.001, l: float=1000):
        self.X_mean = X.mean()
        self.X_std = X.std()
        X = (X - self.X_mean) / self.X_std
        X['X0'] = 1
        X = X.fillna(0)
        m,n = X.shape
        X = X.to_numpy()
        y = y.to_numpy()
        
        self.w = np.random.random((1,n))
        
        for it in range(iteration):
            pred = X.dot(self.w.T)
            err = pred - y
            new_w = copy.copy(self.w)
            new_w[0][-1] = 0
            grad = err.T.dot(X) / m + 2*l*new_w #Ridge
            self.w = self.w - alpha*grad
            MSE = err.T.dot(err) / m
            grad_norm = abs(grad).sum()    
            print(it, grad_norm, MSE)
            if grad_norm < min_grad: return
    
    def fit_online(self, X, y, iteration: int=1000, alpha: float=0.1,
            min_grad: float=0.001, l: float=0.0005):
        
        self.X_mean = X.mean()
        self.X_std = X.std()
        X = (X - self.X_mean) / self.X_std
        X['X0'] = 1
        X = X.fillna(0)
        m,n = X.shape
        X = X.to_numpy()
        y = y.to_numpy()
        
        self.w = np.random.random((1,n))
        
        for it in range(iteration):
            for j in random.sample(range(m), m):
                # moglo bi da se stavi umesto m, neki mnaji broj
                # for j in random.sample(range(m), 20):
                pred = X[j].dot(self.w.T)
                err = pred - y[j]
                grad=(err*X[j] + np.concatenate((2*l*self.w[0][:-1],[0])))/m
                self.w = self.w - alpha*grad
            
            pred_ukupno=X.dot(self.w.T)
            err_ukupno=pred_ukupno-y
            grad_ukupno=err_ukupno.T.dot(X)/m + np.concatenate(
                (2*l*self.w[0][:-1],[0]))
            MSE_ukupno=err_ukupno.T.dot(err_ukupno)/m
            grad_norm_ukupno=abs(grad_ukupno).sum()
            print(it, grad_norm_ukupno, MSE_ukupno)
            if grad_norm_ukupno < min_grad: return
            
    def fit_with_exponentially_decay(self, X, y, iteration: int = 100, 
                                     alpha_o: float=0.30, 
                                     min_grad: float=0.001, l: float=0.1):
        self.X_mean = X.mean()
        self.X_std = X.std()
        X = (X - self.X_mean) / self.X_std
        X['X0'] = 1
        X = X.fillna(0)
        m,n = X.shape
        X = X.to_numpy()
        y = y.to_numpy()
        
        self.w = np.random.random((1,n))
        
        for it in range(iteration):
            pred = X.dot(self.w.T)
            err = pred - y
            new_w = copy.copy(self.w)
            new_w[0][-1] = 0
            grad = err.T.dot(X) / m + 2*l*new_w
            alpha = math.pow(0.95, it)*alpha_o # exponentinally
            self.w = self.w - alpha*grad
            MSE = err.T.dot(err) / m
            grad_norm = abs(grad).sum()    
            print(it, grad_norm, MSE, alpha)
            if grad_norm < min_grad: return
    
    def fit_with_decay_rate(self, X, y, iteration: int=100, 
                            alpha_o: float=0.30, decay_alpha: float=1, 
                            min_grad: float=0.001, l: float=0.1):
        self.X_mean = X.mean()
        self.X_std = X.std()
        X = (X - self.X_mean) / self.X_std
        X['X0'] = 1
        X = X.fillna(0)
        m,n = X.shape
        X = X.to_numpy()
        y = y.to_numpy()
        
        self.w = np.random.random((1,n))
        
        for it in range(iteration):
            pred = X.dot(self.w.T)
            err = pred - y
            new_w = copy.copy(self.w)
            new_w[0][-1] = 0
            grad = err.T.dot(X) / m + 2*l*new_w
            alpha = (alpha_o) / (1 + decay_alpha*it) # it je epoha
            self.w = self.w - alpha*grad
            MSE = err.T.dot(err) / m
            grad_norm = abs(grad).sum()    
            print(it, grad_norm, MSE, alpha)
            if grad_norm < min_grad: return
    
    def fit_with_beta(self, X, y, iteration: int=10000, alpha: float=0.6,
                      min_grad: float=0.001, l: float=0.1,
                      beta: float=0.9, alpha_o: float=0.1):
        # pokusaj sa polu-momentumom i alpha decay
        self.X_mean = X.mean()
        self.X_std = X.std()
        X = (X - self.X_mean) / self.X_std
        X['X0'] = 1
        X = X.fillna(0)
        m,n = X.shape
        X = X.to_numpy()
        y = y.to_numpy()
        
        self.w = np.random.random((1,n))
        V_teta = np.zeros((1,n))
        
        for it in range(iteration):
            pred = X.dot(self.w.T)
            err = pred - y
            new_w = copy.copy(self.w)
            new_w[0][-1] = 0
            grad = err.T.dot(X) / m + 2*l*new_w
            V_teta = beta*V_teta + (1-beta)*grad 
            #V_teta = beta*V_teta + grad # radi malo bolje
            if(it%500 == 0):
                alpha = math.pow(0.95, it)*alpha_o
            self.w = self.w - alpha*V_teta
            MSE = err.T.dot(err) / m
            grad_norm = abs(grad).sum()    
            print(it, grad_norm, MSE, alpha)
            if grad_norm < min_grad: return
    
    def predict(self, X):
        X = (X - self.X_mean) / self.X_std
        X['X0'] = 1
        X = X.fillna(0)
        X = X.to_numpy()
        return X.dot(self.w.T)
    
    
        

#%% PROBA

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import copy
import random as random

lr = LinearRegression()

data = pd.read_csv('C:/Users/Zbook G3/Desktop/boston.csv')

train, test = train_test_split(data, test_size=0.2)

X_train = train.iloc[:, 0:-1]
y_train = train.iloc[:, [-1]]

X_test = test.iloc[:, 0:-1]
y_test = test.iloc[:, [-1]]


#lr.fit(X_train, y_train)
lr.fit_online(X_train, y_train)
prediction = lr.predict(X_test)
lr.w
#prediction = pd.DataFrame(prediction)
#y_test = y_test.reset_index()
#new_df = pd.concat([y_test, prediction], axis = 1)


#data = pd.read_csv('C:/Users/Zbook G3/Desktop/house.csv')

#X = data.iloc[:, 0:-1]
#y = data.iloc[:, [-1]]
#lr.fit(X, y)
#data_new = pd.read_csv('C:/Users/Zbook G3/Desktop/house_new.csv')
#prediction = lr.predict(data_new)
#lr.w

#print()

#lr = LinearRegression()
#lr.fit(X, y)
#prediction = lr.predict(data_new)

