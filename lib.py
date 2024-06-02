import json
from datetime import *
import gurobipy as gb
from gurobipy import GRB
import math
from statistics import mean 

import subprocess
import requests
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd 


import os
from lib import *
path = os.getcwd()

# Selective Optimisation Algorithm 
def optimizationHourly(energyBudget, user_demand, indices, userMax, energyDemand, q, QoE, rev, alpha, beta):
    """
    Find optimal configuration for a given energy budget and user demand

    :param p1: energy budget
    :param p2: user demand
    :param p3: indices of configuration options (for the iterations)
    :param p4: dataset of maximum users per configuration option
    :param p5: dataset of energy demand per configuration option
    :param p6: dataset of throughput per configuration option
    :param p7: quality of experience per configuration option
    :param p8: revenue per configuration option
    
    :return: energy demand for optimal configuration
    """ 
    # Create a list of tuples representing all possible combinations of 
    # configuration options across microservices
    shape = [(ms,x) for ms in range(len(indices)) for x in range(len(indices[ms]))]

    # Create optimization model
    m = gb.Model('ca_microservice_global')

    # Variables
    # Define binary variables (b) to represent if a configuration option is chosen (1) or not (0)
    b = m.addVars(shape, vtype=GRB.BINARY, name="b")
    
    # Define continuous variables (u) to represent the scaling factor applied to each microservice stage
    u = m.addVars(len(indices)+1, lb=0.0, vtype=GRB.CONTINUOUS, name="u")

    # Scaling Factor. Indicates how many instances are needed to serve the user demand. 
    # For each microservice
    sf = m.addVars(shape, lb=0, vtype=GRB.INTEGER, name="sf")

    # CHECK: Definisco la variabile qoe_final (valore compreso tra 0 e 1), che vogliamo massimizzare insieme a rev_final 
    # nell'objective function. Rappresenta la QoE complessiva, in percentuale, per la configurazione scelta
    qoe_final = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="qoe_final")

    # CHECK: Definisco la variabile total_rev, che rappresenterà TODO commenta
    total_rev = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="total_rev")

    # CHECK: Definisco la variabile rev_final (valore compreso tra 0 e 1), che vogliamo massimizzare insieme a qoe_final 
    # nell'objective function.  Rappresenta la revenue complessiva, in percentuale, per la configurazione scelta
    rev_final = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="rev_final")

    # Update the model to reflect changes
    m.update()

# Constraints
        
    # Scaling Factor constraint
    for ms in range(len(indices)):
        # Ensure only one configuration option is chosen for each microservice stage
        m.addConstr(sum(b[ms,x] for x in range(len(indices[ms]))) == 1, name="c-ms"+str(ms))


        for x in range(len(indices[ms])):
            # If user capacity for a configuration option is unlimited, set scaling factor to 0
            if userMax[ms][x] == 10000000000:
                m.addConstr(sf[ms,x] == 0, name="c-sf"+str(ms)+str(x))
            else:
                # Ensure user demand is met by the chosen configuration option considering the scaling factor
                m.addConstr(userMax[ms][x]*sf[ms,x] >= u[ms]*user_demand, name="c-sf"+str(ms)+str(x))


    # Constraint for the energy budget: total energy consumption must be less than or equal to the budget
    m.addConstr(sum(b[ms,x]*energyDemand[ms][x]*sf[ms,x] for ms in range(len(indices)) for x in range(len(indices[ms])) ) 
                <= energyBudget, name="c-eb")
                

    # User-Throughput Constraint
    # Set the initial scaling factor to 1
    m.addConstr(u[0] == 1, name="c-q-initial")

    # For each microservice stage, user throughput after the stage is the product of the previous stage's 
    # throughput and the weighted throughput of the chosen configuration option in the current stage, 
    # summed across all options
    for ms in range(0,len(indices)):
        m.addConstr(u[ms] * sum(b[ms,x] * q[ms][x] for x in range(len(indices[ms]))) == u[ms+1], name="c-q"+str(ms))

    # CHECK: Normalized quality of experience of the configuration determined as the sum of all the QoE of the microservices divided
    # by the number of microservices
    totalQoe = 0
    for ms in range(len(indices)): # Itero tra i microservizi della configurazione in esame
        for x in range(len(indices[ms])): # Itero tra le versioni del microservizio in esame
            totalQoe += QoE[ms][x] * b[ms, x] # Sommo i valori di QoE delle versioni dei microservizi scelti nella configurazione
                # (b è la variabile binaria che indica se una versione del ms è stata scelta o meno per la configurazione)
    m.addConstr(qoe_final == totalQoe / len(indices), name="qoe_final_constraint") # qoe_final (cioè il valore normalizzato di QoE)
        # definito come la QoE totale della configurazione, diviso il numero di microservizi (che equivale a dividere per il valore
        # massimo di QoE, essendo questo al massimo 1 per ciascuna versione di ms e potendo noi scegliere solo una di queste per ms)

    # CHECK: total_rev è definita come il totale della revenue, per utente, della configurazione (rev[ms][x]) moltiplicata per
    # il numero attuale di utenti che fanno uso del ms (u[ms] * user_demand). b[ms, x] serve a far sì che solo le versioni
    # del ms scelte nella configurazione attuale siano usate nel calcolo.
    m.addConstr(total_rev == sum(rev[ms][x] * b[ms, x] * u[ms] * user_demand for ms in range(len(indices)) for x in range(len(indices[ms]))),
            name="total_rev_constraint")

    # CHECK: Normalized revenue of the configuration determined as the sum of all the rev of the microservices divided
    # by the maximum revenue
    total_rev_perUser = 0
    maxRev = 0
    for ms in range(len(indices)): # Per ogni ms
        maxRev += rev[ms][-1] # N.B. la revenue, se definita per il ms, è definita nel json in input come ultima versione del
            # ms, in quanto andiamo convenzionalmente a definire per ultima la versione più performante
        for x in range(len(indices[ms])): # Per ogni versione del ms in esame
            total_rev_perUser += rev[ms][x] * b[ms, x]
    m.addConstr(rev_final == total_rev_perUser / maxRev, name="rev_final_constraint")

    # CHECK: Ha senso mettere questo constraint? alpha e beta sono parametri in input all'intera funzione (che qui abbiamo
    # impostato a 0.5, per dare ugual peso a revenue e QoE nell'objective function) e vogliamo che la loro somma sia 1, così
    # da ottenere un risultato dell'objective in percentuale. Ma, appunto, i parametri sono settati ad un valore fisso al momento
    # della chiamata della funzione, quindi non credo serva definire il constraint qua, ma solo imporlo all'utente per una
    # corretta esecuzuine del codice. Add constraint: alpha + beta = 1
    m.addConstr(alpha + beta == 1.0, name="sum_constraint")


    m.update() # Update the model
    
    # CHECK: The objective function is set to maximize QoE and revenue and express the result in percentage
    m.setObjective(alpha * qoe_final + beta * rev_final, GRB.MAXIMIZE)

    # Compute optimal solution
    m.params.NonConvex = 2
    m.optimize()

    if m.status == GRB.INFEASIBLE: # Se non posso rientrare nel CB, ritorno tutto a -1 per segnalare di eseguire l'applicazione in LP
        print("Model is infeasible!")
        return(-1, -1, -1, -1, -1, -1)
    else:
        # Extract Output

        # The achieved user throughput with the optimal configuration is extracted from the variables named 
        # u and indexed by 5.
        qValue = [var.X for var in m.getVars() if "u" in var.VarName][5] 
    
        # The user throughput after considering the scaling factor is calculated by multiplying the achieved 
        # throughput (qValue) by the user demand.
        userThrougput = [var.X for var in m.getVars() if "u" in var.VarName][5] * user_demand
        ed = 0

        # The chosen configuration options are extracted from the variables named b.
        execFormats = [var.X for var in m.getVars() if "b" in var.VarName]

        # The scaling factors for the chosen configuration options are extracted from the variables named sf.
        scalingFactor = [var.X for var in m.getVars() if "sf" in var.VarName]

        # The QoE for the chosen configuration is extracted from the variable named qoe_final.
        qualityOfExperience = m.getVarByName("qoe_final").x

        # The revenue for the chosen configuration is extracted from the variable named total_rev.
        revenue = m.getVarByName("total_rev").x

        # The revenue for the chosen configuration in percentage is extracted from the variable named rev_final.
        revenuePercentage = m.getVarByName("rev_final").x

        # A flattened list of energy demand values is created from the nested energyDemand list.
        energyDemand_flat = [item for sublist in energyDemand for item in sublist]

        #The total energy demand (ed) is calculated by summing the products of the chosen configuration 
        # options (execFormats), their scaling factors
        for i in range(len(execFormats)):
            ed += execFormats[i]*scalingFactor[i]*energyDemand_flat[i]
    return(qValue, userThrougput, ed, qualityOfExperience, revenue, revenuePercentage) # Ritorno il throughput in percentuale, quello effettivo in base alla domanda dell'user 
        # e la domanda energetica


# Helper Functions

# Data Import functions
## 1.  Get BPMN data constants
def getConstantsFromBPMN(bpmnFile):
    """
    Extracts the constants from a given JSON file

    :param p1: bpmnFile [in JSON]
    return: data, vars_ms, userMax, energyDemand, q, QoE and rev

    This function extracts constants from a JSON file in BPMN format.

    Args:
      bpmnFile (str): The path to the JSON file containing the BPMN data.

    Returns:
        tuple: A tuple containing five elements:
            - data (dict): The complete data loaded from the JSON file.
            - vars_ms (list): A list of lists containing variable names for each microservice.
            - userMax (list): A list of lists containing the maximum user capacity for each resource in each microservice.
            - energyDemand (list): A list of lists containing the energy demand for each resource in each microservice.
            - q (list): A list of lists containing the scaling factor (q) for each resource in each microservice.
            - QoE (list): quality of experience for each version of the microservices (max: 1, when in maximum performance version; min: 0 for skipped optional ms)
            - rev (list): revenue for each version of the microservices (usually 0, non equal to 0 only for specific optional ms, when executed)
    """

    with open(bpmnFile) as data_file:
        data = json.load(data_file) # Intero file json
        vars_ms = [[]] * len(data['components'])
        userMax = []
        energyDemand = []
        q = []
        QoE = []
        rev = []
        i = 0 
        for ms in data['components']:  # For each microservice in the data (flight search, weather information, etc.)
            ca = data['components'][ms] # Varie configurazioni (LP, N, HP e eventualmente Off) e rispettivi parametri, per ogni ms
            vars_ms[i] = [ms]
    

             # Initialize empty lists for userMax, energyDemand, q, QoE and rev for the current microservice
            userMax.append([])
            energyDemand.append([])
            q.append([])
            QoE.append([])
            rev.append([])
            for x in ca.items(): # Loop through each key-value pair in ca (i parametri di ciascuna config)

                    # TODO credo che vars_ms sia inutilmente una list
                    # di list, in quanto sembra contenere la stessa list in ogni elemento

                    vars_ms[i].append(x)
                    if x[1]["user-scaling"] is None:
                        userMax[i].append(10000000000) # Se non c'è user scaling (versione off del ms) metto 10000000000
                    else:
                        userMax[i].append(x[1]["user-scaling"]) # Altrimenti estrapolo il valore corretto e lo aggiungo alla list userMAx
                    energyDemand[i].append(x[1]["energy-demand"]) # Aggiungo alla list energyDeman il valore dell'ED della versione corrente
                    q[i].append(x[1]["q"]) # Aggiungo alla list q il valore dell'user throughput
                    QoE[i].append(x[1]["QoE"])
                    rev[i].append(x[1]["rev"])
                   # j += 1
            i += 1

    return(data, vars_ms, userMax, energyDemand, q, QoE, rev)


# 2. Functions to generate the carbon budgets in input for the experiments
def calcEnergyDemandMax(maxClickData_hourly): 
    """ 
    Calculates an Energy demand similar to that chosen to run the entire app in HP

    :param p1: user request Data
    return: Energy Demand per Hour
    """
    data, vars_ms, userMax, energyDemand, q, QoE, rev = getConstantsFromBPMN('flightBooking.json')

    ## Max q per ms in Architecture (quindi q massimo per le varie versioni del ms)
    maxQ = [max(q[i]) for i in range(len(q))]

    ## Max users per Microservice
    click_max = [] # list che conterrà per ogni ora il numero di click 
                                         # e il numero di click moltiplicato per q massimo di ogni ms

    click_max.append(maxClickData_hourly)
    click_max.append(maxClickData_hourly * maxQ[0])
    for i in range(1,len(maxQ)):
        click_max.append(click_max[i] * maxQ[i])

    ## and number of machines needed per microservice (Scaling Factor)

    max_numInstances = []

    for i in range(len(userMax)):  # Gestisco il caso di userMax = 0
        temp = click_max[i] / userMax[i][-1] # temp sarà uguale al numero massimo previsto di utenti eseguendo una versione HP 
                                             # dell'app diviso il numero massimo di utenti gestibile dal ms
        if math.isnan(temp): 
            max_numInstances.append(0) 
            print(temp,i)
        else:
            max_numInstances.append(math.ceil(click_max[i] / userMax[i][-1]))

    ## Max ED per ms in Architecture (domanda energetica massima, tra le varie versioni, per ms)
    maxEnergyDemand = [max(energyDemand[i]) for i in range(len(energyDemand))]

    ## SF of AVG deployment
    ed = 0 # in kwH per microservice per hour

    # L'energia richiesta ad ogni ora (costante) da ciascun ms sarà uguale alla richiesta massima per quell'ora
    # di quel ms per il numero di istanze richieste per il ms in quell'ora
    ed = [maxEnergyDemand[i] * max_numInstances[i] for i in range(len(maxEnergyDemand))]

    return(ed) 


def calcEnergyDemandMin(maxClickData_hourly): 
    """ 
    Calculates an Energy demand similar to that chosen to run the entire app in LP

    :param p1: user request Data
    return: Energy Demand per Hour
    """
    data, vars_ms, userMax, energyDemand, q, QoE, rev = getConstantsFromBPMN('flightBooking.json')

    ## Min q per ms in Architecture (quindi q massimo per le varie versioni del ms)
    minQ = [min(q[i]) for i in range(len(q))]

    ## Max users per Microservice
    click_min = [] # list che conterrà per ogni ora il numero di click 
                   # e il numero di click moltiplicato per q minimo di ogni ms

    click_min.append(maxClickData_hourly)
    click_min.append(maxClickData_hourly * minQ[0])
    for i in range(1,len(minQ)):
        click_min.append(click_min[i] * minQ[i])

    ## and number of machines needed per microservice (Scaling Factor)

    min_numInstances = []

    for i in range(len(userMax)):  # Gestisco il caso di userMax = 0
        temp = click_min[i] / userMax[i][0] # temp sarà uguale al numero massimo previsto di utenti eseguendo una versione HP 
                                             # dell'app diviso il numero massimo di utenti gestibile dal ms
        if math.isnan(temp): 
            min_numInstances.append(0) 
            print(temp,i)
        else:
            min_numInstances.append(math.ceil(click_min[i] / userMax[i][0]))

    ## Min ED per ms in Architecture (domanda energetica minima, tra le varie versioni, per ms)
    minEnergyDemand = [min(energyDemand[i]) for i in range(len(energyDemand))]

    ## SF of AVG deployment
    ed = 0 # in kwH per microservice per hour

    # L'energia richiesta ad ogni ora (costante) da ciascun ms sarà uguale alla richiesta minima per quell'ora
    # di quel ms per il numero di istanze richieste per il ms in quell'ora
    ed = [minEnergyDemand[i] * min_numInstances[i] for i in range(len(minEnergyDemand))]

    return(ed) 


def calcCarbonEmissionFromEnergyDemand(ed, ci_data):
    """
    Calculates the carbon emissions for a given energy demand and carbon intensity data

    :param p1: energy demand per hour per ms [kWh]
    :param p2: carbon intensity data per hour [gCO2eq per kWh]

    return: carbon emissions per hour
    """
    data, vars_ms, userMax, energyDemand, q, QoE, rev = getConstantsFromBPMN('flightBooking.json')

    if len(ci_data) > 8760:
        ci_list = ci_data.tolist()
        ci_data = ci_list[0:1416] + ci_list[1416+24:8784] # Se l'anno è bisestile, salto il 29 febbraio

    avgCI = mean(ci_data)

    emissionsPerHour = sum(ed[i] * avgCI for i in range(len(energyDemand))) # Itero per tutti i ms e calcolo le emissioni
                    # orarie associate come la somma di ciascuna domanda energetica dei ms moltiplicata per la carbon intensity 
                    # media per l'anno
    
    return(emissionsPerHour)


def generateConstantCarbonBudgets(year, clickData_hourly, ci_data):
    '''
    Generate a csv file containing 5 constant carbon budgets, that will be taken as input fir the experiments.
    These CB are calculated as follows:
    - for the energy demand of the application in high performance mode, considering the maximum number of hourly clicks
      in the year we're taking into account
    - for the energy demand of the application in low performance mode, considering the maximum number of hourly clicks
      in the year we're taking into account
    - the average between the first two CB (the one for ed in HP and the one for ed in LP)
    - the average between the first (the one considering ed in HP) and the third (the average one)
    - the average between the second (the one considering ed in LP) and the third (the average one)

    '''
    # Calcolo il numero massimo di click orario per l'anno in esame 
    maxClickData_hourly = 0
    maxClickData_hourly = max(clickData_hourly[i] for i in range(len(clickData_hourly)))

    # Calcolo Carbon Emission per una versione HP dell'applicazione con un numero massimo di utenti e lo userò come uno dei CB
    # costanti in input per gli esperimenti
    maxCB = calcCarbonEmissionFromEnergyDemand(calcEnergyDemandMax(maxClickData_hourly), ci_data) 
   
    # Calcolo Carbon Emission per una versione LP dell'applicazione con un numero massimo di utenti e lo userò come uno dei CB
    # costanti in input per gli esperimenti
    minCB = calcCarbonEmissionFromEnergyDemand(calcEnergyDemandMin(maxClickData_hourly), ci_data)

    # Calcolo tre valori, che utilizzerò come CB in input degli esperimenti, compresi tra maxCB e minCB
    avgCB = (maxCB + minCB) / 2 # CB medio
    highCB = (avgCB + maxCB) / 2 # Media tra CB tra massimo e medio
    lowCB = (avgCB + minCB) / 2 # Media tra CB minimo e medio

    # Scrivo su un file csv i CB che utilizzerò come input
    row = ["MaxCarbonBudget", "MinCarbonBudget", "AVGCarbonBudget", "HighCarbonBudget", "LowCarbonBudget"] # Preparo il file csv
    f = open('data/CB_constant.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(row)

    row=[maxCB, minCB, avgCB, highCB, lowCB]
    writer.writerow(row)

    f.close()

    return


# 3. Function to retrieve carbon busget from the csv file in input for the experiments
def retrieveCarbonBudget(select):
    '''
    Retrieves the carbon budget from a csv file

    :param p1: select (which can be "0", "1", "2", "3", or "4", respectively for the "max", high", "avg", "low" or "min" version of 
        the csv file we want to get our data from)

    return: carbonBudget
    '''
    df = pd.read_csv(r'data/CB_constant.csv')

    if (select == 0):
        carbonBudget = df["MaxCarbonBudget"].values[0] # Leggo dati carbon budget più elevato
    elif (select == 1):
        carbonBudget = df["HighCarbonBudget"].values[0] # Leggo dati del secondo carbon budget più elevato
    elif (select == 2):
        carbonBudget = df["AVGCarbonBudget"].values[0] # Leggo dati del carbon budget medio
    elif (select == 3):
        carbonBudget = df["LowCarbonBudget"].values[0] # Leggo dati del secondo carbon budget meno elevato
    elif (select == 4):
        carbonBudget = df["MinCarbonBudget"].values[0] # Leggo dati del secondo carbon budget meno elevato
    else:
        print("invalid parameter for function retrieveCarbonBudget(select)")
        return

    return carbonBudget
