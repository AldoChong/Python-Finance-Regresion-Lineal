#finanzas.py
"""Clase que tiene varias herramientas para hacer valuacion del precio de acciones bursatiles"""

#Importamos las librerias 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import seaborn as sns
import sqldf 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math


class Beta:

    #Los atributos inicializados deben de ser privados. Recibe 2 strings con el nombre de las bases de datos.
    def __init__(self, accion, ipc):
        """Inicializa el objeto con nombre del archivo de precios e indice."""
        acciondf = pd.read_csv(accion)
        ipcdf = pd.read_csv(ipc)
        acciondf2 = acciondf[['Date', 'Close']]
        ipcdf2 = ipcdf[['Date', 'Close']]
        self._accion = acciondf2
        self._ipc = ipcdf2
        self._nombreAccion = accion
        self._nombreIPC = ipc

    #Funcion que calcula el vector de rendimientos de un serie
    def rend(self, s):
        t = []
        for i in range(0, len(s)-1):
            a = (s[i+1]-s[i])/s[i]
            t.append(a)
        return t
    
    #Funcion que regresa el valor de la beta. 
    def summary(self):
        """Metodo que regresa el valor de la beta y dice si es estadisticamente significativa."""
        slope, intercept, r, p, se = st.linregress(self.rend(self._ipc['Close']), self.rend(self._accion['Close']))
        print('Empresa: ' + self._nombreAccion)
        if p <= 0.05:
            print('La beta no es estadisticamete significativa')
        else:
            print('La beta es estadisticamente significativa')

        print('El valor de la beta es: '+ str(slope))

class Razones:

    
    #Metodo que regresa el resuman con las razones financieras de la empresa. 
    #Para poder usar este metodo es necesario correr manualmente el comando razones = pd.read_csv('razones.csv')
    def summary(self, id_empresa):
        df =sqldf.run('select * from razones where id =="' + id_empresa + '"')
        RC = df['AC'][0]/df['PC'][0]
        CTN = df['AC'][0] - df['PC'][0]
        RAT = df['VT'][0] / df['AT'][0]
        MUN = df['UN'][0] / df['VT'][0]
        ROA = df['UN'][0] / df['AT'][0]
        ROE = df['UN'][0] / df['CT'][0]
        MC = df['AT'][0] / df['CT'][0]
        UPA = df['UN'][0] /df['NA'][0]
        DA = df['PD'][0] / df['NA'][0]
        PU = df['PA'][0] / UPA
        RP = DA / UPA

        print('Razon corriente:', RC)
        print('Capital de trabajo neto: ', CTN)
        print('Rotacion de activos totales: ', RAT)
        print('Margen de utilidad neta: ', MUN)
        print('Rendimiento sobre activos (ROA): ', ROA)
        print('Rendimiento sobre capital (ROE): ', ROE)
        print('Multiplicador de capital: ', MC)
        print('Utilidad por accion: ', UPA)
        print('Precio-utilidades: ', PU)
        print('Dividendos por accion: ', DA)
        print('Razon de pago: ', RP)

class lstm:

    def __init__(self, prices):

        data = pd.read_csv(prices)
        self._precios = data

    #Definimos la funcion que regresa la grafica de las predicciones, el objeto ajustado de la red y las predicciones. Por defecto solo se predice un dia. 
    #Recibo un arreglo de precios, se supone que debe recibir 
    def predict(self):
        precios = self._precios['Close']
        #Hacemos reeslacalmiento de los datos 

        scaler = MinMaxScaler(feature_range = (0,1))
        precios = np.array(precios)
        precios = np.reshape(precios, (-1,1))
        precios = scaler.fit_transform(precios)

        #Hacemos los conjuntos de entrenamiento y prueba. 
        #Definimos el numero de elementos que sirven como etiquetas
        r = 60 
        training_data_len = math.ceil(len(precios)*0.8)
        train_data = precios[:training_data_len]

        X_train = []
        y_train = []

        for i in range(r, len(train_data)):
            X_train.append(train_data[i-r:i])
            y_train.append(train_data[i])

        #Convertimos los conjuntos de entrenamiento en nparrays

        X_train, y_train = np.array(X_train), np.array(y_train)

        #Hacemos reshape para poder usar los metodos de TensorFlow

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        #Se pasa a construir la red neuronal. 
        rnn = Sequential()
        rnn.add(LSTM(50, return_sequences = True, input_shape = (X_train.shape[1],1)))
        rnn.add(LSTM(50, return_sequences = False))
        rnn.add(Dense(10))
        rnn.add(Dense(1))

        rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

        #Hacemos el entrenamiento de la red
        rnn.fit(X_train, y_train, batch_size = 1, epochs = 5)
    
        #Creamos los datos de prueba para la prediccion de un dia 
        test_data = precios[training_data_len-r:]
        X_test_pred1 = []

        for i in range(r, len(test_data)+1):
            X_test_pred1.append(test_data[i-r:i])

        #los hacemos arreglos de numpy
        X_test_pred1 = np.array(X_test_pred1)

        #Hacemos reshape de los datos de prueba
        X_test_pred1 = np.reshape(X_test_pred1, (X_test_pred1.shape[0], X_test_pred1.shape[1], 1))

        #predicciones que incluyen un dia en el futuro. 
        predictions_1 = rnn.predict(X_test_pred1)

        predictions_1 = scaler.inverse_transform(predictions_1)

        #Hacemos la grafica 
        datos = self._precios.filter(['Close'])
        entren_graph_data = datos[:training_data_len]
        valid_graph_data = datos[training_data_len:]
        valid_graph_data['Predictions'] = predictions_1[0:len(predictions_1) -1]

        plt.plot(entren_graph_data)
        plt.plot(valid_graph_data[['Close', 'Predictions']])
        plt.show()

        return rnn, predictions_1[-1][0]

    #Funciona igual que predict pero recibe manualmente el conjunto muestral, se debe meter como self._precios['Close'] y no hace la grafica
    def predict_alter(self, precios):
        #Hacemos reeslacalmiento de los datos 
        scaler = MinMaxScaler(feature_range = (0,1))
        precios = np.array(precios)
        precios = np.reshape(precios, (-1,1))
        precios = scaler.fit_transform(precios)

        #Hacemos los conjuntos de entrenamiento y prueba. 
        #Definimos el numero de elementos que sirven como etiquetas
        r = 60 
        training_data_len = math.ceil(len(precios)*0.8)
        train_data = precios[:training_data_len]

        X_train = []
        y_train = []

        for i in range(r, len(train_data)):
            X_train.append(train_data[i-r:i])
            y_train.append(train_data[i])

        #Convertimos los conjuntos de entrenamiento en nparrays

        X_train, y_train = np.array(X_train), np.array(y_train)

        #Hacemos reshape para poder usar los metodos de TensorFlow

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        #Se pasa a construir la red neuronal. 
        rnn = Sequential()
        rnn.add(LSTM(50, return_sequences = True, input_shape = (X_train.shape[1],1)))
        rnn.add(LSTM(50, return_sequences = False))
        rnn.add(Dense(10))
        rnn.add(Dense(1))

        rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

        #Hacemos el entrenamiento de la red
        rnn.fit(X_train, y_train, batch_size = 1, epochs = 5)
    
        #Creamos los datos de prueba para la prediccion de un dia 
        test_data = precios[training_data_len-r:]
        X_test_pred1 = []

        for i in range(r, len(test_data)+1):
            X_test_pred1.append(test_data[i-r:i])

        #los hacemos arreglos de numpy
        X_test_pred1 = np.array(X_test_pred1)

        #Hacemos reshape de los datos de prueba
        X_test_pred1 = np.reshape(X_test_pred1, (X_test_pred1.shape[0], X_test_pred1.shape[1], 1))

        #predicciones que incluyen un dia en el futuro. 
        predictions_1 = rnn.predict(X_test_pred1)

        predictions_1 = scaler.inverse_transform(predictions_1)

        return predictions_1[-1][0]

    #Hacemos la funcion para predecir varios periodos, recibe un arreglo del mismo estilo que predict_alter. Esta funcion funciona bien para pocas observaciones pero 
    #es demasiado lenta para muchas
    def super_predict(self, precios, n):
        base = list(self._precios['Close'])

        for i in range(0, n):
            base.append(self.predict_alter(base))
            print('Iteracion: ', i)

        return base

    


