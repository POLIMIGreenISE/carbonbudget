#!/usr/bin/env python
from datetime import *
from statistics import mean 
from itertools import product
import sys 
import os
path = os.path.dirname(os.getcwd())
sys.path.append(path)
from lib import *

data, vars_ms, userMax, energyDemand, q, QoE, rev = getConstantsFromBPMN('flightBooking.json')
# User Data
df = pd.read_csv(r'data/projectcount_wikiDE_2015.csv') # Leggo dati click orari 2015
clickData_hourly = df["De"].tolist()
#CHECK: Non credo che la prossima linea di codice serva più: in pratica, andava a spostare il primo giorno dell'anno del file
# dei click orari a fine anno (dato che il 2015, anno dei dati sui click orari, iniziava di giovedì, mentre il 2021, anno
# dei dati sulla Carbon Intensity di venerdì), dato che nei vecchi esperimenti era centrale il concetto di ora della settimana.
# Nel nostro caso penso abbia forse più senso far corrispondere giorno dell'anno con giorno (1 gennaio con 1 gennaio ecc.),
# indipendentemente da che giorno della settimana fossero.
clickData_hourly = clickData_hourly[24:] + clickData_hourly[0:24]

# Given that this baseline is carbon UNaware, the results coming from here will be valid for all the CB in input

# Write to CSV la domanda di energia oraria, QoE (che in questo caso sarà sempre a 1, visto che usiamo sempre la versione con performance

f = open('results/highPerformanceBaseline.csv', 'w')
writer = csv.writer(f)
row = ["Endergy demand", "QoE", "Revenue per user", "Revenue", "Revenue in percentage"]
writer.writerow(row)
f.close()
f = open('results/highPerformanceBaseline.csv', 'a')
writer = csv.writer(f)

# High Performance

for i in range(8760):  # Per ogni ora dell'anno
    ed = 0
    revenue_perUser = 0
    rev_hourly = 0
    rev_percentage = 1 # In HP avrò sicuramente il 100% della revenue
    s = clickData_hourly[i] # Inizializzo s (numero utenti nel ms in quell'ora) con il numero di click 
                    # in quell'ora

    for j in range(len(q)): # Itero tra i microservizi
        if userMax[j][-1] > 0: # Check if there is a maximum number of users for this resource in HP mode
            ed += energyDemand[j][-1] * math.ceil(s / userMax[j][-1]) # La domanda energetica viene incrementata
                            # del prodotto tra la domanda energetica in HP del microservizio in esame e 
                            # il numero di utenti in quel ms in  quell'ora fratto il numero max di utenti 
                            # in HP (cioè il numero di istanze richiesto), arrotondato per eccesso

        revenue_perUser += rev[j][-1] # Aggiorno la revenue per utente
        rev_hourly += rev[j][-1] * s # Aggiorno la revenue assoluta oraria
        s *= q[j][-1] # Aggiorno il valore di s con il prodotto tra quello in entrata al ms e 
                     # il valore q del ms

    row = []
    row.append(ed)
    row.append(1)
    row.append(revenue_perUser)
    row.append(rev_hourly)
    row.append(rev_percentage)
    writer.writerow(row)

f.close()
