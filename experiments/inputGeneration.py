#!/usr/bin/env python

from statistics import mean 
import pandas as pd

import sys
import csv
import os
path = os.path.dirname(os.getcwd())
sys.path.append(path)
from lib import *

path = os.path.dirname(os.getcwd())



data, vars_ms, userMax, energyDemand, q, QoE, rev = getConstantsFromBPMN('flightBooking.json')


df = pd.read_csv(r'data/DE_2021.csv')
ci_data_2021 = df['carbon_intensity_avg']

df = pd.read_csv(r'data/projectcount_wikiDE_2015.csv') # Leggo click orari anno 2015
clickData_hourly = df["De"]

generateConstantCarbonBudgets(2021, clickData_hourly, ci_data_2021)

#calcCarbonBudgetHourInWeekAVG(2021, clickData_hourly, ci_data_2021) # Generate 5 csv files for CB in input
