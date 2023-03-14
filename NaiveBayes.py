# Naive-Bayes

import pandas as pd
import numpy as np
import math as math
from scipy.stats import norm
from sklearn.model_selection import train_test_split

#%% BIBLIOTEKA


class NaiveBayes:

    #numDF =  pd.DataFrame()
    #nonnumDF = pd.DataFrame()
    
    def fit(self, X, y, alfa: int=1000):
        
        model = {}
        
        apriori = (X[y].value_counts() + alfa)/(
            X[y].value_counts() + alfa).sum() #Smoothing
        model['_apriori'] = apriori
        
        self.numDF = X.select_dtypes(include='number')
        self.nonnumDF = X.select_dtypes(exclude='number')
        
        for attribute in self.nonnumDF.drop(y, axis = 1).columns:
            mat_cont = pd.crosstab(X[attribute], X[y])
            mat_cont += alfa #Smoothing
            mat_cont = mat_cont / mat_cont.sum(axis = 0)
            model[attribute] = mat_cont
            
            
        for attribute in self.numDF.columns:
            param = {}
            for class_value in model['_apriori'].index:
                param[class_value] = {
                    'mean' : (X.loc[X[y] == class_value])[attribute].mean(),
                    'std' : (X.loc[X[y] == class_value])[attribute].std()}
                model[attribute] = param
        
        return model
    
    
    def fill(self, model, new_instance):
        
        class_probabilities_Log = {}
        
        for class_value in model['_apriori'].index:
            probabilityLog = 0
            numeric = 0 # ako je verovatnoca jednaka 0 za log
            # moze i umesto toga da se dodaje najmanja float vrednost-(epsilon)
            for attribute in model:
                
                if(attribute in self.numDF):
                    mean = model[attribute][class_value]['mean']
                    std = model[attribute][class_value]['std']
                    prob = norm.pdf(new_instance[attribute], mean, std)
                    
                    if(prob != 0):
                        probabilityLog += math.log(prob)
                    else:
                        numeric = 1
                    
                else:
                    
                    if(attribute == '_apriori'):
                        prob = model['_apriori'][class_value]
                        if(prob != 0):
                            probabilityLog += math.log(prob)
                        else:
                            numeric = 1
                    else:
                        prob = model[
                            attribute][class_value][new_instance[attribute]]
                        
                        if(prob != 0):
                            probabilityLog += math.log(prob)
                        else:
                            numeric = 1
                        
            if(numeric == 1):
                class_probabilities_Log[class_value] = 0
                numeric = 0
            else:
                class_probabilities_Log[class_value] = math.exp(probabilityLog)                
                
        prediction_Log = max(
            class_probabilities_Log, key = class_probabilities_Log.get)
        # moze bolje da se uzima max vrednost pre exp!
        max_value = class_probabilities_Log[prediction_Log]
        probabiliyMax = max_value / (sum(class_probabilities_Log.values()))
        return prediction_Log, probabiliyMax
    
    def predict(self, model, X):
        
        df = pd.DataFrame()
        i = 0
        for i in range(len(X)):
            prediction_Log, confidence_Log = self.fill(model, X.iloc[i])
            df.loc[i,'Prediction'] = prediction_Log
            df.loc[i,'Confidence'] = confidence_Log
        
        X = X.reset_index()
        new_df = pd.concat([X, df], axis = 1)
        
        return new_df
        
        

#%% PROBA

#data = pd.read_csv('C:/Users/Zbook G3/Desktop/prehlada.csv')
#data_new = pd.read_csv('C:/Users/Zbook G3/Desktop/prehlada_novi.csv')

#nb = NaiveBayes()

#model = nb.fit(data, 'Prehlada')
#rez = nb.predict(model, data_new)

data = pd.read_csv('C:/Users/Zbook G3/Desktop/drug.csv')
train, test = train_test_split(data, test_size=0.2)
test = test.drop('Drug', axis = 1)

nb = NaiveBayes()

model = nb.fit(train, 'Drug')
rez = nb.predict(model, test)

