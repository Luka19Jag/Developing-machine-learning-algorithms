class AdaBoost:
    
    def fit(self, X, y, alg = None, ensemble_size=10, learning_rate=1):
    
        self.ensemble_size = ensemble_size
        self.learning_rate = learning_rate
        self.n, self.m = X.shape
        alfas = pd.Series(np.array([1/self.n]*self.n), index=X.index)
        ensemble = []
        weights = np.zeros(ensemble_size)
        self.probs = pd.DataFrame()
        
        for t in range(self.ensemble_size):
            
            if(str(alg) == 'DecisionTreeClassifier(max_depth=2)'):
                alg_new = DecisionTreeClassifier(max_depth=2)
            if(str(alg) == 'GaussianNB()'):
                alg_new = GaussianNB()
            if(str(alg) == 'LogisticRegression()'):
                alg_new = LogisticRegression()
            if(alg == None):
                broj = random.randint(0,2)
                if(broj == 0):
                    alg_new = DecisionTreeClassifier(max_depth=2)
                if(broj == 1):
                    alg_new = GaussianNB()
                if(broj == 2):
                    alg_new = LogisticRegression()
            # moze i preko deepcopy
                    
                    
            model = alg_new.fit(X=X, y=y, sample_weight=alfas)
            predictions = model.predict(X)
            self.probs[t] = np.array(pd.DataFrame(model.predict_proba(X)).max(axis=1))
            error = (predictions != y).astype(int)
            weighted_error = (error*alfas).sum()
            w = self.learning_rate * 1/2 * math.log(
                (1-weighted_error)/weighted_error) # 2. tacka
            weights[t] = w 
            ensemble.append(alg_new)
            
            factor = np.exp(-w*predictions*y)
            alfas = alfas * factor
            alfas = alfas / alfas.sum()
            
        return ensemble, weights

    def predict(self, X, y, ensemble, weights):
        
        predictions = pd.DataFrame([
            model.predict(X) for model in ensemble]).T
        predictions['ensemble'] = np.sign(predictions.dot(weights))
        
        predikcija = predictions.add(y, axis=0).abs()/2
        accuracy = predikcija.mean()
        print('Tacnost ensembla je:', accuracy['ensemble'])
        print('Tacnost svakog od slabih klasifikatora je:')
        print(accuracy[0:-1])
        #print('1) Povernje celog ansambla je:')
        #tezine = weights[weights>0] ili tezine = abs(weights) -> ovo je bolje
        #print((max(tezine) + min(tezine)) / 2 * accuracy['ensemble'] * self.ensemble_size)
        # / self.learning_rate
        #print('Povernje celog ansambla je:')
        #print((accuracy[0:-1].mean() + accuracy['ensemble'])/2)
        pom = predikcija.drop('ensemble', axis=1)
        sum_weights = sum(weights)
        ver_tacnih = pom*self.probs
        novo = ver_tacnih.dot(weights)/sum_weights
        #print('Poverenje celog ansambla je:', novo.mean())
        print(novo)
        
        return predikcija['ensemble']




#%%
import numpy as np
import pandas as pd
import math as math
import random as random
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('C:/Users/Zbook G3/Desktop/drugY.csv')
#data.drop(['Sex', 'BP','Cholesterol'], inplace=True, axis=1)
X = data.drop('Drug', axis=1)
X = pd.get_dummies(X)
y = data['Drug']*2-1

adaBoost = AdaBoost()
model, tezine = adaBoost.fit(X=X,y=y, alg=DecisionTreeClassifier(max_depth=2))
predikcija = adaBoost.predict(X, y, model, tezine)


