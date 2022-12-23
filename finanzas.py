#finanzas.py
"""Clase que tiene varias herramientas para hacer valuacion del precio de acciones bursatiles"""
#Autor: Aldo Enrique Chong Valentin.

#Importamos las librerias 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import seaborn as sns
import sqldf 
from sklearn.preprocessing import MinMaxScaler
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

