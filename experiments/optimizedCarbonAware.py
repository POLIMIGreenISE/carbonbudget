#!/usr/bin/env python

from statistics import mean 
import pandas as pd

import sys
import csv

import sys 
import os
path = os.path.dirname(os.getcwd())
sys.path.append(path)
from lib import *

path = os.path.dirname(os.getcwd())



data, vars_ms, userMax, energyDemand, q, QoE, rev = getConstantsFromBPMN('flightBooking.json')

df = pd.read_csv(r'data/DE_2021.csv')
ci_data_2021 = df['carbon_intensity_avg']

for CB_version in range(5):
    carbonBudget = retrieveCarbonBudget(CB_version)

    if CB_version == 0:
        x = "MAX_CB_"
    elif CB_version == 1:
        x = "HIGH_CB_"
    elif CB_version == 2:
        x = "AVG_CB_"
    elif CB_version == 3:
        x = "LOW_CB_"
    elif CB_version == 4:
        x = "MIN_CB_"


    indices = []
    for ms in data['components']:
        indices.append([0] * len(data['components'][ms]))

    df = pd.read_csv(r'data/projectcount_wikiDE_2015.csv') # Leggo click orari anno 2015
    clickData_hourly = df["De"].tolist()

    row = ["q","user-throughput", "QoE", "total rev", "rev in percentage", "ed","eb","U","ce", "Objective"] # Preparo la prima riga del file csv
    f = open('results/optimizedCarbonAware/' + x + 'optimization_result.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(row)
    f.close()

    QoE_hourly = 0
    rev_hourly = 0
    alpha = 0.5
    beta = 0.5

    for t in range(len(ci_data_2021)): # per ogni ora dell'anno
        eb = carbonBudget/ci_data_2021[t] # Calcolo l'energy budget come il CB diviso la carbon intensity CI in quell'ora 
    
        qValue, userThrougput, ed, QoE_hourly, rev_hourly, rev_percentage = optimizationHourly(eb, clickData_hourly[t], indices, userMax, 
                                                                            energyDemand, q, QoE, rev, alpha, beta)
        if(qValue == -1): # Se non ho modo di rientrare nel CB nemmeno con la configurazione meno performante, metto comunque questa
                # in schedule
                
            ed = math.ceil(clickData_hourly[t] / userMax[0][0]) * energyDemand[0][0] # Domanda energetica definita come il numero di
                # click per quell'ora, diviso il numero massimo di utenti del primo ms (quindi il num. di istanze del primo ms)
                # il tutto moltiplicato per la richiesta energetica di una singola istanza del ms
            user_temp = clickData_hourly[t] * q[0][0] # Il numero di utenti in uscita è dato da il num di click orari per q del primo ms
   
            for i in range(1,len(userMax)): # Itero tra i vari ms
                if (userMax[i][0] == 10000000000): # se vedo che la versione del ms è "off")
                    ed += 0 # La domanda energetica sarà 0
                else:
                    ed += math.ceil(user_temp / userMax[i][0]) * energyDemand[i][0] # Altrimenti la domanda energetica è data
                        # da il numero di utenti in entrata al ms diviso il numero massimo del ms (quindi il num. di istanze del ms)
                        # il tutto moltiplicato per la richiesta energetica di una singola istanza del ms
                user_temp *= q[i][0] # Il numero di utenti in uscita è dato dal num di utenti in entrata al ms per q del ms

            ce = ed * ci_data_2021[t]
            qValue = user_temp
            userThrougput = qValue * clickData_hourly[t]
            QoE_hourly = sum(QoE[ms][0] for ms in range(len(QoE))) / len(QoE) 
            rev_hourly = 0
            rev_percentage = 0
            objective = -1
        else:
            ce = ed * ci_data_2021[t] # Calcolo Carbon Emissions dell'ora come la domanda energetica ED per la CI 
                                    # in quell'ora
            objective = alpha * QoE_hourly + beta * rev_percentage
        row = [qValue, userThrougput, QoE_hourly, rev_hourly, rev_percentage, ed, eb, clickData_hourly[t], ce, objective]
        f = open('results/optimizedCarbonAware/' + x + 'optimization_result.csv', 'a')
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()