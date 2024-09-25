#!/usr/bin/env python
from datetime import *
import csv
import pandas as pd

import math
from statistics import mean 
import sys 
import os

path = os.path.dirname(os.getcwd())
sys.path.append(os.path.abspath(path))
from lib import *


df = pd.read_csv(r'data/projectcount_wikiDE_2015.csv') # Leggo dai click del 2015
clickData_hourly = df["De"].tolist()

df = pd.read_csv(r'data/DE_2021.csv') # Leggo dati CI 2021
ci_data_2021 = df['carbon_intensity_avg']


for inputVersion in range(4):

    if inputVersion == 0:
        y = "HH"
    elif inputVersion == 1:
        y = "HL"
    elif inputVersion == 2:
        y = "LH"
    elif inputVersion == 3:
        y = "LL"

    data, vars_ms, userMax, energyDemand, q, QoE, rev = getConstantsFromBPMN("flightBooking_" + y +".json")

 
    # Defining 3 execution formats

    lowPower = []
    normalPower = []
    highPower = []
    for i in range(len(q)): # Definisco valori di q, ed, userMax, QoE e rev per le varie configurazioni
        lowPower.append([q[i][0], energyDemand[i][0], userMax[i][0], QoE[i][0], rev[i][0]])
        highPower.append([q[i][-1], energyDemand[i][-1], userMax[i][-1], QoE[i][-1], rev[i][-1]])

        for j in range(len(vars_ms[i])):
            if(vars_ms[i][j][0] == "Normal"):
                normalPower.append([vars_ms[i][j][1]['q'], vars_ms[i][j][1]['energy-demand'], vars_ms[i][j][1]['user-scaling'],
                                vars_ms[i][j][1]['QoE'], vars_ms[i][j][1]['rev']])
        
        if(len(normalPower) == i): # Se non era definita una versione "Normal" del microservizio
            normalPower.append([q[i][-1], energyDemand[i][-1], userMax[i][-1], QoE[i][-1], rev[i][-1]]) # Uso quella a prestazioni più alte


    for CB_version in range(8):
        carbonBudget = retrieveCarbonBudget(CB_version)

        if CB_version == 0:
            x = "CONSTANT_MAX_CB_"
        elif CB_version == 1:
            x = "CONSTANT_HIGH_CB_"
        elif CB_version == 2:
            x = "CONSTANT_AVG_CB_"
        elif CB_version == 3:
            x = "CONSTANT_LOW_CB_"
        elif CB_version == 4:
            x = "CONSTANT_MIN_CB_"
        elif CB_version == 5:
            x = "ADAPTIVE_MAX_CB_"
        elif CB_version == 6:
            x = "ADAPTIVE_AVG_CB_"
        elif CB_version == 7:
            x = "ADAPTIVE_MIN_CB_" 

        n_HP = 0
        n_NP = 0
        n_LP = 0

        maxRev = sum(highPower[i][-1] for i in range(len(vars_ms))) # Definisco la revenue massima ottenibile come somma delle rev
        # in HP dei vari microservizi

        row = ["q","user-throughput","ed","QoE","Revenue per user", "Revenue", "Revenue in percentage","eb","U","ce"] # Preparo il file csv
        f = open('results/' + y + '/simpleCarbonAware/'+ x +'simpleCarbonAwareBaseline_' + y + '.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()
                
        for t in range(8760): # Itero per tutte le ore dell'anno
            q_hourly = 1
            ed = 0
            QoE_hourly = 0
            revenue_perUser = 0
            rev_hourly = 0
            rev_percentage = 0
            s = clickData_hourly[t]
            ed_hp = calculateEnergyDemand(highPower, clickData_hourly[t]) # Calcolo ed orario in HP
            ed_np = calculateEnergyDemand(normalPower, clickData_hourly[t]) # Calcolo ed orario in NP

            if CB_version <= 4: # Se sto operando con CB costanti
                eb = carbonBudget/ci_data_2021[t] # Calcolo budget energetico orario come CB diviso la CI per quell'ora
            else: # Se sto operando con CB adattivi
                eb = carbonBudget[t]/ci_data_2021[t] # Calcolo l'energy budget come il CB in quell'ora diviso la carbon intensity CI 
                                                    # in quell'ora 
            
            if ed_hp <= eb: # Se l'energia richiesta in HP è minore del budget
                ed = ed_hp # Scelgo la modalità HP e quindi ed è uguale a ed in HP
                QoE_hourly = 1 # In HP ho il massimo della QoE
                rev_percentage = 1 # In HP ho il massimo della revenue

                for j in range(len(vars_ms)):
                    q_hourly *= highPower[j][0] # q è dato dal prodotto totale dei q dei vari ms in HP
                    revenue_perUser += highPower[j][-1]
                    rev_hourly += s * highPower[j][-1]
                    s *= highPower[j][0]   
                n_HP += 1
                        
            elif ed_np <= eb: # Se l'energia richiesta in NP è minore del budget, ma quella in HP è maggiore
                ed = ed_np # Scelgo la modalità NP e quindi ed è uguale a ed in NP

                for j in range(len(vars_ms)):
                    q_hourly *= normalPower[j][0] # q è dato dal prodotto totale dei q dei vari ms in NP
                    QoE_hourly += normalPower[j][3] / sum(highPower[i][3] for i in range(len(vars_ms)))
                    # QoE orario è dato dalla somma dei QoE della config scelta, diviso la QoE massima (che è quella della HP) per 
                    # riportare il valore in percentuale
                    revenue_perUser += normalPower[j][-1]
                    rev_hourly += s * normalPower[j][-1]
                    if(maxRev != 0):
                        rev_percentage += normalPower[j][-1] / maxRev
                    else:
                        rev_percentage = 1 # Se non ho definito alcuna revenue per alcun ms dell'app, allora sto sempre estraendo il
                            # massimo della revenue (zero)
                    s *= normalPower[j][0]
                n_NP += 1
            
            else: # Se l'energia richiesta sia in HP che in NP è maggiore del budget
                ed = calculateEnergyDemand(lowPower, clickData_hourly[t]) # Scelgo la modalità LP e quindi ed è uguale a ed in LP
                
                for j in range(len(vars_ms)):
                    q_hourly *= lowPower[j][0] # q è dato dal prodotto totale dei q dei vari ms in NP
                    QoE_hourly += lowPower[j][3] / sum(highPower[i][3] for i in range(len(vars_ms)))
                    # QoE orario è dato dalla somma dei QoE della config scelta, diviso la QoE massima (che è quella della HP) per 
                    # riportare il valore in percentuale
                    revenue_perUser += lowPower[j][-1]
                    rev_hourly += s * lowPower[j][-1]
                    if(maxRev != 0):
                        rev_percentage = lowPower[j][-1] / maxRev
                    else:
                        rev_percentage = 1 # Se non ho definito alcuna revenue per alcun ms dell'app, allora sto sempre estraendo il
                            # massimo della revenue (zero)
                    s *= lowPower[j][0]
                n_LP += 1
            
            row = [q_hourly, q_hourly*clickData_hourly[t], ed, QoE_hourly, revenue_perUser, rev_hourly, rev_percentage] 
                # Salvo nella riga da stampare q, il throughput  (q * i click orari), l'ed, la QoE e la revenue

            # Writing results to CSV
            f = open('results/' + y + '/simpleCarbonAware/'+ x +'simpleCarbonAwareBaseline_' + y + '.csv', 'a')
            row.append(eb) # Aggiungo alla row in budget energetico orario
            row.append(clickData_hourly[t]) # Aggiungo alla row il numero di click per quell'ora (U)
            row.append(ed * ci_data_2021[t]) # Aggiungo alla row le emissioni di anidride carbonica
            writer = csv.writer(f)
            writer.writerow(row) # Scrivo la row sul file csv
            f.close()

        print(n_HP, "; ", n_NP, "; ", n_LP, "; ")

