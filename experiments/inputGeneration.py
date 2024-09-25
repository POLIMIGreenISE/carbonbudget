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

df = pd.read_csv(r'data/DE_2021.csv')
ci_data_2021 = df['carbon_intensity_avg']

df = pd.read_csv(r'data/projectcount_wikiDE_2015.csv') # Leggo click orari anno 2015
clickData_hourly = df["De"]

generateConstantCarbonBudgets(clickData_hourly, ci_data_2021)

df = pd.read_csv(r'data/DE_2020.csv')
ci_data_2020 = df['carbon_intensity_avg']

df = pd.read_csv(r'data/projectcount_wikiDE_2014.csv') # Leggo click orari anno 2014
clickData_hourly_2014 = df["de"]

generateAdaptiveCarbonBudgets(clickData_hourly_2014, ci_data_2020) # Generate 3 csv files for CB in input
