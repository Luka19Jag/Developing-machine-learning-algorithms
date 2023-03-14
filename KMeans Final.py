#%% KLasa

class KMeans():
    
    # k-broj klastera
    # N-broj iteracija za ponavljanje kmeans-a
    # N_iter-broj pokusaja da se ne ponovi isti kvalitet
    # Var_granica-uslov za 5.tacku
    # Br_el_granica-uslov za 5. tacku
    # p-za minkowski na koji stepen ide
    def fit(self, data, list_of_weights=[], k=3, N=1, N_iter=50,
            distance_metric='euclidean',
            Var_granica = 1000, Br_el_granica = 150, p=3):
        
        list_of_distances = ['euclidean', 'city_block', 'minkowski',
                             'chebyshev']
        if (distance_metric in list_of_distances) == False:
            print('Nije odabrana odgovarajuca metrika merenja udaljenosti')
            return
        
        self.n, self.m = data.shape
        
        self.data_mean = data.mean()
        self.data_std = data.std()
        data = (data - self.data_mean) / self.data_std
        
        if(len(list_of_weights) == 0):
            list_of_weights = [1 for i in range(self.m)]
        
        dictionary_centroids = {}
        dictionary_subset = {}
        dictionary_ocena_kvaliteta = {}
        for number in range(N):
            dictionary_subset[number] = {}
            dictionary_ocena_kvaliteta[number] = {}
        for number1 in range(N):    
            for number2 in range(k):
                dictionary_ocena_kvaliteta[number1][number2] = {'Var': 0,
                                                                'Broj el.': 0}
        
        for number in range(N):

            print('ITERATION:', number)    
            
            centroids = data.sample(k).reset_index(drop = True)
            assign = np.zeros((self.n,1))
            old_quality = float('inf')    

            for iteration in range(N_iter):
                quality = np.zeros(k)
                
                # 1. dodela tacaka klasterima
                for i in range(self.n):
                    
                    slucaj = data.iloc[i]
                    
                    if distance_metric == 'euclidean':
                        df = (slucaj-centroids)*list_of_weights
                        dist = ((df)**2).sum(axis=1)
                        # mislim da treba dodati sqrt
                        # dist = math.sqrt(((df)**2).sum(axis=1))
                    
                    if distance_metric == 'city_block':
                        df = abs(slucaj-centroids)*list_of_weights
                        dist = df.sum(axis=1)
                    
                    if distance_metric == 'minkowski':
                        df = abs(slucaj-centroids)*list_of_weights
                        dist = (((df)**p).sum(axis=1))**(1/p)
                        # za p=1 -> doboja se city-block
                        # za p=2 -> doboja se euclidien
                    
                    if distance_metric == 'chebyshev':
                        df = abs(slucaj-centroids)*list_of_weights
                        dist = df.max(axis=1)
                        
                    assign[i] = np.argmin(dist)
                    
                # 2. preracunavanje centroida
                for c in range(k):
                    subset = data[assign == c]
                    dictionary_subset[number][c] = subset
                    centroids.loc[c] = subset.mean()
                    # 1. tacka - tezinski koeficjenti
                    quality[c] = (subset).var().sum() * len(subset)
                    # subset * list_of_weights
                    dictionary_ocena_kvaliteta[number][c][
                        'Var'] = quality[c]
                    dictionary_ocena_kvaliteta[number][c][
                        'Broj el.'] = len(subset)
                
                total_quality = quality.sum()
                print(iteration, total_quality)
                if old_quality == total_quality: break
                old_quality = total_quality 
            
            
            print()
            centroids = centroids * self.data_std + self.data_mean
            print(centroids)
            print()
            
            dictionary_centroids[number] = {'centroids':centroids,
                                            'total_quality':old_quality}
            
            # 5.tacka pocetak
            for c_el in range(k):
                if dictionary_ocena_kvaliteta[number][c_el][
                    'Broj el.'] < Br_el_granica and dictionary_ocena_kvaliteta[
                        number][c_el]['Var'] > Var_granica:
                            print('Centroid ne odredjuje najbolje klaster: ',
                                  c_el)
                            print()
            
           
            for cent in range(k):
                stari_df = centroids.iloc[cent]*0.1
                novi_df = centroids.iloc[cent] - centroids.iloc[
                    centroids.index != cent, :]
                novi_df = abs(novi_df)
                razlika = (stari_df > novi_df).sum(axis = 1)
                razlika = pd.DataFrame(razlika == 7) # broj kolona po kome 
                #treba da budu slicni za 10 posto
                for cent2 in range(k-1):
                    if razlika.iloc[cent2][0] == True:
                        print('Slicni su klasteri', cent, 'i', list(
                            razlika.index)[cent2])
                        print()
                
            # 5.tacka kraj
            
        model = 0
        for number in range(N):
            if dictionary_centroids[
                    number]['total_quality'] < dictionary_centroids[
                    model]['total_quality']:
                model = number
        
        print('Najbolji model:')
        print(dictionary_centroids[model]['centroids'])
        print()
        print('Najbolji kvalitet:')
        print(dictionary_centroids[model]['total_quality'])
        print()
        
        for number in range(k):
            dictionary_subset[model][number] = dictionary_subset[
                model][number] * self.data_std + self.data_mean
        
        return dictionary_centroids, dictionary_centroids[
            model], dictionary_subset[
            model], dictionary_ocena_kvaliteta

            

    def fit_kmeans_proisern(self, data, k=3, N=5, N_iter=50):
        #4. tacka
        self.n, self.m = data.shape
        
        self.data_mean = data.mean()
        self.data_std = data.std()
        data = (data - self.data_mean) / self.data_std
        
        dictionary_centroids = {}
        dic_veliki_veliki = {}
        for number in range(N):
            dic_veliki_veliki[number] = {}
        
        for number in range(N):
            
            print('ITERATION:', number)    
            
            dic_veliki = {}
            dataFrameCentroids = pd.DataFrame()
            centroid = data.sample(1).reset_index(drop = True)
            # mozda bi bolji nacin bio da se izracuna mean od svih
            # opservacija, da se izracuna udaljenost svake opservacije od
            # mean-a i da se uzme kao pocetna tacka, ona koja je najdalja
            # od tog mean-a
            dataFrameCentroids = dataFrameCentroids.append(centroid)

            for iteration in range(k-1):#treba k-1

                dic = {}
                najdalji = pd.DataFrame(dataFrameCentroids.mean()).T
                najdalji_vr = float(((najdalji - pd.DataFrame(
                    dataFrameCentroids.mean()).T)**2).sum(axis = 1))
                for proba in range(self.n): #treba n, odnosno za svaku opservaciju
                    pokusaj = float(((data.iloc[proba] - pd.DataFrame(
                        dataFrameCentroids.mean()).T)**2).sum(axis = 1))
                    dic[proba] = {'cvor':pd.DataFrame(data.iloc[proba]).T,
                                  'vrednost':pokusaj}
                    if (pokusaj > najdalji_vr):
                        potencijalni = pd.DataFrame(
                            data.iloc[proba]).T.reset_index(drop = True)
                        cent_n, cent_m = dataFrameCentroids.shape
                        vr = False
                        for cent in range(cent_n):
                            if((potencijalni == dataFrameCentroids.iloc[
                                    cent]).T.all()[0] == True):
                                vr = True
                                break
                                #da ne budu isti centroidi
                        if(vr == False):
                            najdalji = pd.DataFrame(data.iloc[
                                proba]).T.reset_index(drop = True)
                            najdalji_vr = float(copy.copy(pokusaj))
                #dic = {}
                #najdalji = 1
                #najdalji_vr = float('-inf')
                #for proba in range(5): #treba n
                 #   pokusaj = min(((data.iloc[
                 #proba] - dataFrameCentroids)**2).sum(axis = 1))
                  #  dic[proba] = {'cvor':pd.DataFrame(data.iloc[
                  #proba]).T, 'vrednost':pokusaj}
                   # if (pokusaj > najdalji_vr):
                    #    najdalji = pd.DataFrame(data.iloc[
                    #proba]).T.reset_index(drop = True)
                     #   najdalji_vr = float(copy.copy(pokusaj))
                
                dic_veliki[iteration] = dic
                dataFrameCentroids = dataFrameCentroids.append(najdalji)
                
            dic_veliki_veliki[number] = dic_veliki
                
            dataFrameCentroids = dataFrameCentroids.reset_index()
            print('provera')
            print(dataFrameCentroids)
            print('provera')
            
            dataFrameCentroids.drop("index", axis=1, inplace=True)
                
            assign = np.zeros((self.n,1))
            old_quality = float('inf')    

            for iteration in range(N_iter):
                quality = np.zeros(k)
                
                # 1. dodela tacaka klasterima
                for i in range(self.n):
                    slucaj = data.iloc[i]
                    df = slucaj-dataFrameCentroids
                    dist = ((df)**2).sum(axis=1)
                    assign[i] = np.argmin(dist)
                
                # 2. preracunavanje centroida
                for c in range(k):
                    subset = data[assign == c]
                    dataFrameCentroids.loc[c] = subset.mean()
                    quality[c] = subset.var().sum() * len(subset)
                
                total_quality = quality.sum()
                print(iteration, total_quality)
                if old_quality == total_quality: break
                old_quality = total_quality
            
            
            
            print()
            dataFrameCentroids = dataFrameCentroids * self.data_std + self.data_mean
            print(dataFrameCentroids)
            print()
            print('quality', total_quality)
            print()
            
            dictionary_centroids[number] = {'centroids':dataFrameCentroids, 
                                  'total_quality':total_quality}
        
        model = 0
        for number in range(N):
            if dictionary_centroids[
                    number]['total_quality'] < dictionary_centroids[
                    model]['total_quality']:
                model = number
        
        print('Najbolji model:')
        print(dictionary_centroids[model]['centroids'])
        print()
        print('Najbolji kvalitet:')
        print(dictionary_centroids[model]['total_quality'])
        print()
        
        return dictionary_centroids[
            model], dictionary_centroids,dic_veliki_veliki
    
    
    def fit_silhouette(self, data):
        
        self.n, self.m = data.shape
        
        self.data_mean = data.mean()
        self.data_std = data.std()
        data = (data - self.data_mean) / self.data_std
        
        dictionary_za_siluet = {}
        dictionary_pomocni = {}
        broj_klastera = 2
        max_broj_klastera = 3 #treba da bude n, koliko ima i opservacija
        
        silhouette_ukupno = 0
        
        dic_siluet_za_klastere = {}
        for nu in range(broj_klastera, max_broj_klastera+1):
            dic_siluet_za_klastere[nu] = {}

        while(True):
            
            if(broj_klastera > max_broj_klastera):
                break
            
            for nu in range(broj_klastera):
                dic_siluet_za_klastere[
                    broj_klastera][nu] = {'Ukupno':0, 'Broj':0, 'Siluet':0}

            for number in range(5): 
            # uradi 5 iteracija da vidis koji 
            # je najbolji model za taj borj klastera
            
                print('ITERATION:', number)    
            
                centroids = data.sample(broj_klastera).reset_index(drop = True)
                assign = np.zeros((self.n,1))
                old_quality = float('inf')    
            
                for iteration in range(50):
                    quality = np.zeros(broj_klastera)
                    
                    # 1. dodela tacaka klasterima
                    for i in range(self.n):
                        slucaj = data.iloc[i]
                        df = slucaj-centroids
                        dist = ((df)**2).sum(axis=1)
                        assign[i] = np.argmin(dist)
                        
                    # 2. preracunavanje centroida
                    for c in range(broj_klastera):
                        subset = data[assign == c]
                        centroids.loc[c] = subset.mean()
                        quality[c] = subset.var().sum() * len(subset)
                    
                    total_quality = quality.sum()
                    print(iteration, total_quality)
                    if old_quality == total_quality: break
                    old_quality = total_quality 
                
                dictionary_pomocni[number] = { 'broj klastera': broj_klastera,
                    'centroids':centroids, 'total_quality':total_quality}
            
            model = 0
            for nu in range(5):
                if dictionary_pomocni[nu]['total_quality'] < dictionary_pomocni[
                        model]['total_quality']:
                    model = nu
                    
            nas_model = dictionary_pomocni[model]
            
            for i in range(self.n):
                
                slucaj = data.iloc[i]
                a_index = int(assign[i][0])
                dictionary_za_instancu = {}
                
                for prom in range(broj_klastera):            
                    dictionary_za_instancu[prom] = {'Ukupno':0, 'Broj':0}
                
                for prom in range(self.n):
                    if(prom != i): # nemoj za istu instancu nista da racunas
                    # ali svakako, za tu instancu bi vrendost bila 0 za Ukupno,
                    # ali bi moralo da se Broj smanji za jedan
                        index = assign[prom][0]
                        vrednost = ((slucaj - data.iloc[prom])**2).sum(
                            axis = 0)
                        dictionary_za_instancu[index][
                            'Ukupno'] = dictionary_za_instancu[
                                index]['Ukupno'] + vrednost
                        dictionary_za_instancu[index][
                            'Broj'] = dictionary_za_instancu[
                                index]['Broj'] + 1
                
                if(dictionary_za_instancu[a_index]['Broj'] == 0):
                    a_vrednost = 0
                    # nemoguce je da se ovo desi
                else:
                    a_vrednost = dictionary_za_instancu[
                        a_index]['Ukupno']/dictionary_za_instancu[
                            a_index]['Broj']
                
                nova_lista_avg_vrednosti = []
                for prom in range(broj_klastera):
                    if(prom != a_index):
                        if(dictionary_za_instancu[prom]['Broj'] == 0):
                            nova_lista_avg_vrednosti.append(float('inf'))
                        else:
                            nova_lista_avg_vrednosti.append(
                                dictionary_za_instancu[
                                    prom]['Ukupno']/dictionary_za_instancu[
                                        prom]['Broj'])
                
                b_vrednost = min(nova_lista_avg_vrednosti)
                
                silhouette_za_instancu = (b_vrednost-a_vrednost)/(
                    max(b_vrednost,a_vrednost))
                silhouette_ukupno += silhouette_za_instancu
                
                dic_siluet_za_klastere[
                    broj_klastera][a_index]['Ukupno'] = dic_siluet_za_klastere[
                        broj_klastera][
                            a_index]['Ukupno'] + silhouette_za_instancu
                    
                dic_siluet_za_klastere[
                    broj_klastera][a_index]['Broj'] =  dic_siluet_za_klastere[
                        broj_klastera][
                            a_index]['Broj'] + 1
            
            silhouette_ukupno /= self.n
            
            
                
            print()
            print('BROJ KLASTERA: ', broj_klastera)
            print('SILHOUETTE: ', silhouette_ukupno)
            print()
            centroids = centroids * self.data_std + self.data_mean
            print(centroids)
            print()
            
            print()
            print('Silhouette za svaki klaster')
            for nu in range(broj_klastera):
                if(dic_siluet_za_klastere[broj_klastera][nu]['Broj'] == 0):
                    dic_siluet_za_klastere[broj_klastera][nu]['Siluet'] = 0
                else:
                    dic_siluet_za_klastere[broj_klastera][
                        nu]['Siluet'] = dic_siluet_za_klastere[broj_klastera][
                            nu]['Ukupno'] / dic_siluet_za_klastere[
                                broj_klastera][nu]['Broj']
                                
                vr = dic_siluet_za_klastere[broj_klastera][nu]['Siluet']
                if(vr > 0.6):
                    print('Klaster', nu, ': je odlicno klasterovan')
                if(vr > 0.4 and vr <= 0.6):
                    print('Klaster', nu, ': je vrlo dobro klasterovan')
                if(vr > 0.1 and vr <= 0.4):
                    print('Klaster', nu, ': nije dobro klasterovan, treba razmotriti da li moze biti koristan')
                if(vr <= 0.1):
                    print('Klaster', nu, ': je jako lose klasterovan')
                    
                    
            print('Silhouette za svaki klaster')
            print()
            dictionary_za_siluet[broj_klastera] = {
                'SILHOUETTE' : silhouette_ukupno}
            broj_klastera += 1

        return dictionary_za_siluet, dic_siluet_za_klastere
    
    def transform(self, centroids, slucaj):
        df = slucaj-centroids
        dist = ((df)**2).sum(axis=1)
        klaster = np.argmin(dist)
        print('Slucaj pripada klasteru', klaster)
        
        
        
        
        
        

#%% Main

import numpy as np
import pandas as pd
import math as math
import copy as copy
import random

data = pd.read_csv('C:/Users/Zbook G3/Desktop/boston.csv')#.set_index('country')

niz = np.ones(5)
#niz[0] = 2
#niz[1] = 3

kmeans = KMeans()
#dictionary, model, subset, ocena_kvaliteta = kmeans.fit(data=data, k=3, list_of_weights=niz)
#model, dic, dic_veliki, ocena_kvaliteta = kmeans.fit(data=data)
dictionary_centroids, model, dic_veliki_veliki = kmeans.fit_kmeans_proisern(data)
#dictionary_za_siluet, dic_siluet_za_klastere = kmeans.fit_silhouette(data=data)

#slucaj = data.iloc[random.randint(0,37)]
#kmeans.transform(model['centroids'], slucaj)

    
