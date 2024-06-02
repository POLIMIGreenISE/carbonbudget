# Reducing the Carbon Footprint of Microservice-Based Cloud Applications by Periodically Scaling their Energy Demand

![Overview of the Approach](approach-overview.jpg)

# Experiments
There are in total 8 different experiments that were conducted:
1. Selective Optimization
2. Carbon-Unaware (static) Baselines
3. Simple carbon-aware approach
4. Randomized Energy Budget with up to 40% exceeding budget
5. Uniform Recycle Budget
6. Hourly Recycle Budget 
7. Weighted Recycle Budget
8. Mathematical Optimum Solution (Note this Approach is not part of the Thesis)
9. Randomized Energy Budget with random budget between 80% and 120% (Validation Approach)


The results of the experiments can be found in the form of CSV files in the [results folder](results).


# Data visualisation 
The [dataViz folder](dataViz) contains the Jupyter Notebook used to create the visualisations of the results.  
The images that were created with the notebook can be found in the [img folder](img).
0. General Data Visualisation
  - Carbon Intensity Heatmap
  - Energy Budget Heatmap
  - Mean Hour in Week Carbon Intensity
  - Mean Hour in Week Carbon Budget
1. Carbon Budget Utilization
2. Revenue
3. Revenue to energy demand ratio
4. Carbon Emissions
99. Statistics Calculations
  - Statistical Results that are used in "Interpretation of the Results" Section of the Thesis


# Data 

## Data Sources
The data used for the experiments can be found in the [data folder](data).  
It consists of the following data sets:
- Carbon Intensity Data from [Electricity Maps](https://www.electricitymaps.com/data-portal) for Germany
  - [2020](/data/DE_2020.csv)
  - [2021](/data/DE_2021.csv)
- User-Request Data from [Wikimedia ](https://dumps.wikimedia.org/other/pagecounts-raw/)
  - [2014](data/projectcount_wikiDE_2014.csv)
  - [2015](data/projectcount_wikiDE_2015.csv)

The carbon intensity data is licensed under [ODbl](https://opendatacommons.org/licenses/odbl/).  
The wikimedia pageview dataset is licensed under [Creative Commons Zero (CC0) public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/) as it is part of the [Wikimedia: Analytics Datasets](https://dumps.wikimedia.org/other/analytics/), whose license is stated in [https://dumps.wikimedia.org/legal.html](https://dumps.wikimedia.org/legal.html).

## Application Data
The data for the application architecture of the flight booking use case is stored in a [JSON file](flightBooking.json).  
Additionally there is an [XML-Parser](xml_parser/bpmnToJSON.py) that can parse BPMN-models in the form of XML files to JSON files.  
The BPMN XML file could for example have been derived from a BMPN modeller, such as [bpmn.io](https://bpmn.io/).  
The derived JSON file can then be enriched with the needed parameters for selectibe optimisation algorithm.



#  How to run the experiments

## Requirements
In order to run the experiments, the following requirements need to be met:  
- Python 3.9.12
  - [Requirements](requirements.txt)
- [Gurobi Optimizer](https://www.gurobi.com/) (for the mathematical optimum solution): a license is needed
  - An academic license can be obtained [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Installation of the python environment
We advise to use a virtual environment, such as [Anaconda](https://www.anaconda.com/).  
If you do not have Anaconda installed, you can download it [here](https://www.anaconda.com/products/individual).
With Anaconda and the following commands, it can be ensured that the same environment is used for the experiments:
```bash
conda create --name <env-name> python=3.9.12
conda activate <env-name>
conda install gurobi
pip install -r requirements.txt
```

To leave your conda envrionment again you can use the following command:
```bash
conda deactivate
```


## Acitvating the Gurobi License
The Gurobi license has to be activated before the experiments can be run.
```bash
grbgetkey <license-key>
```

# Running the experiments
The experiments can be run by executing the following command:
```python
python3 <x-experiment.py>
```
DISCLAIMER: The experiments can take a long time to run, depending on the parameters that are used.
or by executing the shell script [main.sh](results/main.sh) in the results folder:
```bash
cd experiments
chmod +x main.sh
./main.sh
```


# Library of functions
The [lib.py](lib.py) file contains all the functions used in multiple experiments.
Some of the experiment files itself have single functions that are only used in that specific experiment  

In [lib.py](lib.py) the following functions can be found:
- The Selectibe Optimization Algorithm 
- num_days_between
- weekdayfrequency(year)
- getConstantsFromBPMN: Extracts the parameters from the JSON file that is used to describe the application architecture
- calcEnergyDemandFromAVG
- calcQFromAVG
- calcCarbonEmissionFromEnergyDemand
- calcCarbonBudgetFrom_AVG_CE
- getS_hourlyAVG
- calcEnergyBudgetHourInWeekAVG
- calcCarbonBudgetHourInWeekAVG
- calcED_LP  

Each of these functions and its input parameters and output are explained as a comment at the start of their definition in [lib.py](lib.py).