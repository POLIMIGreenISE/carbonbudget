import argparse
import csv
import json
import math
import os
from datetime import datetime
from statistics import mean
from typing import List, Tuple

import pandas as pd
import gurobipy as gb
from gurobipy import GRB


# =============================================================================
# CORE OPTIMIZATION FUNCTIONS
# =============================================================================

def optimizationHourly(energyBudget, user_demand, indices, userMax, energyDemand, q, QoE, rev, alpha, beta):
    """
    Find optimal microservice configuration for given energy budget and user demand.

    Returns: (qValue, userThroughput, energyDemand, QoE, revenue, revenuePercentage)
    """
    shape = [(ms, x) for ms in range(len(indices)) for x in range(len(indices[ms]))]

    m = gb.Model('ca_microservice_global')
    m.setParam('OutputFlag', 0)  # Silent mode

    # Variables
    b = m.addVars(shape, vtype=GRB.BINARY, name="b")  # Configuration selection
    u = m.addVars(len(indices) + 1, lb=0.0, vtype=GRB.CONTINUOUS, name="u")  # User flow
    sf = m.addVars(shape, lb=0, vtype=GRB.INTEGER, name="sf")  # Scaling factor
    qoe_final = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="qoe_final")
    total_rev = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="total_rev")
    rev_final = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="rev_final")

    m.update()

    # Constraints
    for ms in range(len(indices)):
        # One configuration per microservice
        m.addConstr(sum(b[ms, x] for x in range(len(indices[ms]))) == 1)

        for x in range(len(indices[ms])):
            if userMax[ms][x] == 10000000000:  # Off configuration
                m.addConstr(sf[ms, x] == 0)
            else:
                m.addConstr(userMax[ms][x] * sf[ms, x] * b[ms, x] >= u[ms] * user_demand * b[ms, x])

    # Energy budget constraint
    m.addConstr(sum(b[ms, x] * energyDemand[ms][x] * sf[ms, x]
                   for ms in range(len(indices)) for x in range(len(indices[ms]))) <= energyBudget)

    # User throughput constraints
    m.addConstr(u[0] == 1)
    for ms in range(len(indices)):
        m.addConstr(u[ms] * sum(b[ms, x] * q[ms][x] for x in range(len(indices[ms]))) == u[ms + 1])

    # QoE and Revenue constraints
    m.addConstr(qoe_final == sum(b[ms, x] * QoE[ms][x] for ms in range(len(indices)) for x in range(len(indices[ms]))) / len(indices))
    m.addConstr(total_rev == sum(b[ms, x] * rev[ms][x] for ms in range(len(indices)) for x in range(len(indices[ms]))))

    # Calculate max revenue
    maxRev = sum(max(rev[ms]) for ms in range(len(indices)))
    m.addConstr(rev_final == total_rev / maxRev)

    # Objective: Multi-objective optimization
    m.setObjective(alpha * qoe_final + beta * rev_final, GRB.MAXIMIZE)

    # Solve
    m.optimize()

    if m.status == GRB.Status.OPTIMAL:
        # Extract solution
        qValue = u[len(indices)].X
        userThroughput = qValue * user_demand
        totalEnergyDemand = sum(b[ms, x].X * energyDemand[ms][x] * sf[ms, x].X
                               for ms in range(len(indices)) for x in range(len(indices[ms])))

        return qValue, userThroughput, totalEnergyDemand, qoe_final.X, total_rev.X, rev_final.X
    else:
        print(f"Optimization failed with status: {m.status}")
        return 0, 0, 0, 0, 0, 0


def getConstantsFromBPMN(filename="data/flightBooking_HH.json"):
    """Extract constants from BPMN JSON configuration file."""
    try:
        with open(filename) as f:
            data = json.load(f)
        
        # Extract microservice data
        vars_ms = []
        userMax = []
        energyDemand = []
        q = []
        QoE = []
        rev = []
        
        for ms_name, config in data['components'].items():
            vars_ms.append([ms_name] + list(config.items()))
            
            ms_userMax = []
            ms_energyDemand = []
            ms_q = []
            ms_QoE = []
            ms_rev = []
            
            for config_name, params in config.items():
                ms_userMax.append(10000000000 if params["user-scaling"] is None else params["user-scaling"])
                ms_energyDemand.append(params["energy-demand"])
                ms_q.append(params["q"])
                ms_QoE.append(params["QoE"])
                ms_rev.append(params["rev"])
            
            userMax.append(ms_userMax)
            energyDemand.append(ms_energyDemand)
            q.append(ms_q)
            QoE.append(ms_QoE)
            rev.append(ms_rev)
        
        # Create indices for optimization
        indices = [list(range(len(userMax[i]))) for i in range(len(userMax))]
        
        return data, vars_ms, userMax, energyDemand, q, QoE, rev, indices
        
    except FileNotFoundError:
        print(f"Configuration file {filename} not found")
        return None, None, None, None, None, None, None, None


def calcEnergyDemandMax(click_data):
    """Calculate energy demand for high performance mode - matches High Performance baseline."""
    _, _, userMax, energyDemand, q, _, _, _ = getConstantsFromBPMN()
    
    # Calculate user flow through microservices using HP mode Q values
    s = click_data
    ed_per_microservice = []
    
    # Iterate through microservices like in High Performance baseline
    for j in range(len(q)):
        if userMax[j][-1] > 0:  # Check if there is a maximum number of users for this resource in HP mode
            # Calculate energy demand for this microservice
            ed_ms = energyDemand[j][-1] * math.ceil(s / userMax[j][-1])
            ed_per_microservice.append(ed_ms)
        else:
            ed_per_microservice.append(0)
        
        # Update s for next microservice
        s *= q[j][-1]
    
    return ed_per_microservice


def calcEnergyDemandMin(click_data):
    """Calculate energy demand for low performance mode."""
    _, _, userMax, energyDemand, q, _, _, _ = getConstantsFromBPMN()
    
    minQ = [min(q[i]) for i in range(len(q))]
    
    # Calculate user flow through microservices
    click_min = [click_data]
    for i in range(len(minQ)):
        click_min.append(click_min[i] * minQ[i])
    
    # Calculate required instances
    min_numInstances = []
    for i in range(len(userMax)):
        temp = click_min[i] / userMax[i][0]
        min_numInstances.append(0 if math.isnan(temp) else math.ceil(temp))
    
    # Calculate energy demand
    minEnergyDemand = [min(energyDemand[i]) for i in range(len(energyDemand))]
    return [minEnergyDemand[i] * min_numInstances[i] for i in range(len(minEnergyDemand))]


def calcCarbonEmissionFromEnergyDemand(ed, ci_data, index):
    """Calculate carbon emissions from energy demand and carbon intensity."""
    _, _, _, energyDemand, _, _, _, _ = getConstantsFromBPMN()
    
    ci_data_list = ci_data.tolist() if hasattr(ci_data, 'tolist') else ci_data
    
    if index == -1:  # Constant carbon budget
        avgCI = mean(ci_data_list)
        return sum(ed[i] * avgCI for i in range(len(ed)))
    else:  # Adaptive carbon budget
        return sum(ed[i] * ci_data_list[index] for i in range(len(ed)))


def calcHighPerformanceEnergyDemand(click_data, config_file=None):
    """Calculate energy demand using actual High Performance baseline logic."""
    if config_file:
        _, _, userMax, energyDemand, q, _, _, _ = getConstantsFromBPMN(config_file)
    else:
        _, _, userMax, energyDemand, q, _, _, _ = getConstantsFromBPMN()
    
    ed = 0
    s = click_data
    
    # Use the same logic as High Performance baseline
    for j in range(len(q)):
        if userMax[j][-1] > 0:
            ed += energyDemand[j][-1] * math.ceil(s / userMax[j][-1])
        else:
            ed += energyDemand[j][-1]
        
        s *= q[j][-1]
    
    return ed


def calcMinimalEnergyDemand(click_data, config_file=None):
    """Calculate minimal energy demand using lowest power configurations."""
    if config_file:
        _, _, userMax, energyDemand, q, _, _, _ = getConstantsFromBPMN(config_file)
    else:
        _, _, userMax, energyDemand, q, _, _, _ = getConstantsFromBPMN()
    
    ed = 0
    s = click_data
    
    # Use minimal energy configurations (first option for each microservice)
    for j in range(len(q)):
        if userMax[j][0] > 0:
            ed += energyDemand[j][0] * math.ceil(s / userMax[j][0])
        else:
            ed += energyDemand[j][0]
        
        s *= q[j][0]
    
    return ed


def extractRequestTraceByYear(workload_file, year):
    """Extract request trace data for a specific year from workload file."""
    df = pd.read_csv(workload_file)
    df['ds'] = pd.to_datetime(df['ds'])
    df['year'] = df['ds'].dt.year
    
    year_data = df[df['year'] == year]
    if len(year_data) == 0:
        raise ValueError(f"No data found for year {year} in {workload_file}")
    
    return year_data['y'].tolist()


def generateHistoricalCarbonBudgets(region, workload, config_file=None, cache_dir=None):
    """Generate carbon budgets based on 2022 data for use in 2023 experiments."""
    import os
    import pickle
    
    # Find the project root directory (contains run_experiments.py)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(current_file)
    
    # Set default cache directory relative to project root
    if cache_dir is None:
        cache_dir = os.path.join(project_root, "data", "cache")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename
    workload_name = workload.replace('.csv', '')
    if config_file:
        config_name = os.path.basename(config_file).replace('.json', '')
    else:
        config_name = 'default'
    cache_file = os.path.join(cache_dir, f"historical_budgets_{region}_{workload_name}_{config_name}.pkl")
    
    # Check if cached results exist
    if os.path.exists(cache_file):
        print(f"Loading cached historical budgets from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Generating historical budgets for {region}/{workload}/{config_name}...")
    
    # Load 2022 carbon intensity data (absolute path)
    ci_2022_file = os.path.join(project_root, "data", f"{region}_2022_hourly.csv")
    if not os.path.exists(ci_2022_file):
        raise FileNotFoundError(f"2022 carbon intensity data not found: {ci_2022_file}")
    
    df_carbon_2022 = pd.read_csv(ci_2022_file)
    ci_data_2022 = df_carbon_2022['Carbon Intensity gCO₂eq/kWh (direct)']
    ci_data_2022 = ci_data_2022.fillna(ci_data_2022.mean())
    
    # Load 2022 request traces (absolute path)
    workload_file = os.path.join(project_root, "data", workload)
    if not os.path.exists(workload_file):
        raise FileNotFoundError(f"Workload data not found: {workload_file}")
    
    clickData_2022 = extractRequestTraceByYear(workload_file, 2022)
    
    # Normalize request trace to match carbon intensity data length
    if len(clickData_2022) > len(ci_data_2022):
        clickData_2022 = clickData_2022[:len(ci_data_2022)]
    elif len(clickData_2022) < len(ci_data_2022):
        multiplier = len(ci_data_2022) // len(clickData_2022) + 1
        clickData_2022 = (clickData_2022 * multiplier)[:len(ci_data_2022)]
    
    # Calculate historical budgets using 2022 data
    highCB_list = []     # High Performance energy demand (100%)
    averageCB_list = []  # Average between high and low
    lowCB_list = []      # Minimal energy demand
    
    click_data_copy = clickData_2022.copy()
    
    for i in range(len(click_data_copy)):
        if click_data_copy[i] == 0:
            click_data_copy[i] = mean(click_data_copy)
        
        # Calculate High Performance energy demand with specific config
        hp_energy_demand = calcHighPerformanceEnergyDemand(click_data_copy[i], config_file)
        
        # Calculate Minimal energy demand with specific config
        minimal_energy_demand = calcMinimalEnergyDemand(click_data_copy[i], config_file)
        
        # Calculate carbon emissions for different budget levels using 2022 carbon intensity
        high_carbon_emission = hp_energy_demand * ci_data_2022.iloc[i]
        low_carbon_emission = minimal_energy_demand * ci_data_2022.iloc[i]
        
        # Define budgets
        highCB = high_carbon_emission  # High Performance deployment
        lowCB = low_carbon_emission    # Minimal deployment
        averageCB = (highCB + lowCB) / 2  # Average between high and low
        
        highCB_list.append(highCB)
        averageCB_list.append(averageCB)
        lowCB_list.append(lowCB)
    
    # Create results
    result = (pd.Series(highCB_list, name="HighCarbonBudget"),
              pd.Series(averageCB_list, name="AverageCarbonBudget"),
              pd.Series(lowCB_list, name="LowCarbonBudget"))
    
    # Cache results
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Historical budgets cached to {cache_file}")
    return result


def generateAdaptiveCarbonBudgets(click_data, ci_data, config_file=None, use_historical=True, region=None, workload=None):
    """Generate adaptive carbon budgets using actual High Performance energy demand at 100%, 80%, and 50%.
    
    Args:
        click_data: Request trace data
        ci_data: Carbon intensity data
        config_file: Configuration file name
        use_historical: If True, use historical 2022-based budgets
        region: Region code (required if use_historical=True)
        workload: Workload filename (required if use_historical=True)
    """
    if use_historical:
        if region is None or workload is None:
            raise ValueError("region and workload must be specified when use_historical=True")
        return generateHistoricalCarbonBudgets(region, workload, config_file)
    
    # Original dynamic budget calculation
    highCB_list = []     # High Performance energy demand (100%)
    averageCB_list = []  # Average between high and low
    lowCB_list = []      # Minimal energy demand
    
    click_data_copy = click_data.copy()
    
    for i in range(len(click_data_copy)):
        if click_data_copy[i] == 0:
            click_data_copy[i] = mean(click_data_copy)
        
        # Calculate High Performance energy demand with specific config
        hp_energy_demand = calcHighPerformanceEnergyDemand(click_data_copy[i], config_file)
        
        # Calculate Minimal energy demand with specific config
        minimal_energy_demand = calcMinimalEnergyDemand(click_data_copy[i], config_file)
        
        # Calculate carbon emissions for different budget levels
        high_carbon_emission = hp_energy_demand * ci_data.iloc[i]
        low_carbon_emission = minimal_energy_demand * ci_data.iloc[i]
        
        # Define budgets
        highCB = high_carbon_emission  # High Performance deployment
        lowCB = low_carbon_emission    # Minimal deployment
        averageCB = (highCB + lowCB) / 2  # Average between high and low
        
        highCB_list.append(highCB)
        averageCB_list.append(averageCB)
        lowCB_list.append(lowCB)
    
    return (pd.Series(highCB_list, name="HighCarbonBudget"),
            pd.Series(averageCB_list, name="AverageCarbonBudget"),
            pd.Series(lowCB_list, name="LowCarbonBudget"))


def selectCarbonBudget(highCB, averageCB, lowCB, select):
    """Select appropriate carbon budget based on index."""
    budgets = [highCB, averageCB, lowCB]
    if 0 <= select < len(budgets):
        return budgets[select]
    print(f"Invalid select parameter: {select}. Valid values: 0 (HIGH), 1 (AVERAGE), 2 (LOW)")
    return None


def calculateEnergyDemand(configuration, user_demand):
    """Calculate hourly energy demand for given configuration and user demand."""
    ed = math.ceil(user_demand / configuration[0][2]) * configuration[0][1]
    user_temp = user_demand * configuration[0][0]
    
    for i in range(1, len(configuration)):
        if configuration[i][2] == 10000000000:  # Off configuration
            ed += 0
        else:
            ed += math.ceil(user_temp / configuration[i][2]) * configuration[i][1]
            user_temp *= configuration[i][0]
    
    return ed


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

# Configuration
CARBON_REGIONS = {
    'DE': {'file': 'DE_2023_hourly.csv', 'name': 'Germany'},
    'CISO': {'file': 'CISO_2023_hourly.csv', 'name': 'California'},
    'ERCOT': {'file': 'ERCOT_2023_hourly.csv', 'name': 'Texas'},
}

WORKLOADS = {
    'wiki_en': {'file': 'wiki_en.csv', 'name': 'Wikipedia English'},
}

APP_CONFIGS = ['HH', 'HL', 'LH', 'LL']

CARBON_BUDGET_TYPES = [
    'ADAPTIVE_HIGH_CB_',
    'ADAPTIVE_AVERAGE_CB_',
    'ADAPTIVE_LOW_CB_'
]

# Module-level cache for carbon budgets
_carbon_budgets_cache = {}


def load_data(region: str, workload: str) -> Tuple[pd.Series, List[int]]:
    """Load carbon intensity and workload data."""
    # Load 2023 carbon intensity data
    carbon_file = f'data/{CARBON_REGIONS[region]["file"]}'
    df_carbon = pd.read_csv(carbon_file)
    ci_data = df_carbon['Carbon Intensity gCO₂eq/kWh (direct)']
    ci_data = ci_data.fillna(ci_data.mean())

    # Load 2023 workload data
    workload_file = f'data/{WORKLOADS[workload]["file"]}'
    click_data = extractRequestTraceByYear(workload_file, 2023)

    # Normalize workload to match carbon intensity data length
    if len(click_data) > len(ci_data):
        click_data = click_data[:len(ci_data)]
    elif len(click_data) < len(ci_data):
        multiplier = len(ci_data) // len(click_data) + 1
        click_data = (click_data * multiplier)[:len(ci_data)]

    return ci_data, click_data


def result_exists(result_file: str, expected_hours: int) -> bool:
    """Check if result file exists and has expected number of rows."""
    try:
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            return len(df) == expected_hours
        return False
    except:
        return False


def calc_low_power_energy_demand(click_data, userMax, energyDemand, q):
    """Calculate energy demand for low power mode."""
    return calcEnergyDemandMin(click_data)


def calc_low_power_throughput(click_data, q):
    """Calculate throughput for low power mode."""
    q_value = 1.0
    for ms_q in q:
        q_value *= min(ms_q)
    return q_value


def get_microservice_configs(userMax, energyDemand, q, QoE, rev):
    """Get microservice configurations for different power modes."""
    configs = {
        'lowPower': [],
        'normalPower': [],
        'highPower': []
    }

    for ms in range(len(userMax)):
        # Low power: first configuration
        configs['lowPower'].append((q[ms][0], energyDemand[ms][0], userMax[ms][0], QoE[ms][0], rev[ms][0]))

        # High power: last configuration
        last_idx = len(q[ms]) - 1
        configs['highPower'].append((q[ms][last_idx], energyDemand[ms][last_idx], userMax[ms][last_idx], QoE[ms][last_idx], rev[ms][last_idx]))

        # Normal power: middle if available, else high power
        if len(q[ms]) >= 3:
            configs['normalPower'].append((q[ms][1], energyDemand[ms][1], userMax[ms][1], QoE[ms][1], rev[ms][1]))
        else:
            configs['normalPower'].append((q[ms][last_idx], energyDemand[ms][last_idx], userMax[ms][last_idx], QoE[ms][last_idx], rev[ms][last_idx]))

    return configs


def run_selective_optimization(config: str, ci_data: pd.Series,
                               click_data: List[int], region: str,
                               workload: str, output_dir: str = 'results') -> None:
    """Run the selective optimization algorithm. Requires valid Gurobi license."""
    data, vars_ms, userMax, energyDemand, q, QoE, rev, indices = getConstantsFromBPMN(f"data/flightBooking_{config}.json")

    # Generate historical carbon budgets (cached)
    cache_key = f"{region}_{workload}_{config}"
    if cache_key not in _carbon_budgets_cache:
        highCB, averageCB, lowCB = generateAdaptiveCarbonBudgets(
            click_data, ci_data, f"data/flightBooking_{config}.json",
            use_historical=True, region=region, workload=f"{workload}.csv"
        )
        _carbon_budgets_cache[cache_key] = (highCB, averageCB, lowCB)
    else:
        highCB, averageCB, lowCB = _carbon_budgets_cache[cache_key]

    for cb_idx, cb_name in enumerate(CARBON_BUDGET_TYPES):
        # Create output directory and file
        result_dir = f'{output_dir}/{region}/{workload}/{config}/optimized'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/{cb_name}result.csv'

        # Skip if result already exists
        if result_exists(result_file, len(click_data)):
            print(f"Skipping optimized: {region}/{workload}/{config}/{cb_name} (already exists)")
            continue

        # Get carbon budget
        carbonBudget = selectCarbonBudget(highCB, averageCB, lowCB, cb_idx)

        # Write header
        header = ["hour", "energy_demand", "carbon_budget", "user_requests",
                 "carbon_emissions", "objective", "QoE", "rev_per_user",
                 "total_revenue", "rev_percentage", "region", "workload",
                 "config", "algorithm", "budget_type"]

        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        print(f"Running optimized algorithm: {region}/{workload}/{config}/{cb_name}")

        # Run optimization for each hour
        for t in range(len(click_data)):
            # Convert carbon budget to energy budget by dividing by carbon intensity
            eb = carbonBudget.iloc[t] / ci_data.iloc[t]

            # Run optimization
            result = optimizationHourly(eb, click_data[t], indices, userMax,
                                      energyDemand, q, QoE, rev, 0.5, 0.5)

            qValue, userThroughput, ed, qoe, revenue, objective = result

            if click_data[t] == 0:
                ed = calc_low_power_energy_demand(click_data[t], userMax, energyDemand, q)
                qValue = calc_low_power_throughput(click_data[t], q)
                userThroughput = qValue * click_data[t]

            # Calculate carbon emissions
            ce = ed * ci_data.iloc[t]

            # Calculate revenue per user and percentage
            rev_per_user = revenue / click_data[t] if click_data[t] > 0 else 0
            maxRev = sum(max(rev[ms]) for ms in range(len(rev)))
            rev_percentage = revenue / maxRev if maxRev > 0 else 0

            # Save result
            with open(result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([t, ed, carbonBudget.iloc[t], click_data[t], ce, objective,
                               qoe, rev_per_user, revenue, rev_percentage, region, workload,
                               config, "optimized", cb_name.rstrip('_')])

def run_high_performance_baseline(config: str, click_data: List[int],
                                region: str, workload: str, output_dir: str = 'results') -> None:
    """Run the high performance baseline algorithm."""
    data, vars_ms, userMax, energyDemand, q, QoE, rev, indices = getConstantsFromBPMN(f"data/flightBooking_{config}.json")

    # Create output directory and file
    result_dir = f'{output_dir}/{region}/{workload}/{config}/high_performance'
    os.makedirs(result_dir, exist_ok=True)
    result_file = f'{result_dir}/baseline_result.csv'
    
    # Skip if result already exists
    if result_exists(result_file, len(click_data)):
        print(f"Skipping high performance baseline: {region}/{workload}/{config} (already exists)")
        return
    
    # Write header
    header = ["hour", "energy_demand", "QoE", "rev_per_user", "total_revenue", 
             "rev_percentage", "user_requests", "region", "workload", "config", "algorithm"]
    
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    print(f"Running high performance baseline: {region}/{workload}/{config}")
    
    # Calculate maximum revenue
    maxRev = sum(rev[ms][-1] for ms in range(len(rev)))
    
    # Run for each hour
    for i in range(len(click_data)):
        ed = 0
        revenue_perUser = 0
        s = click_data[i]
        
        # Calculate energy demand and revenue for high performance mode
        for j in range(len(q)):
            if userMax[j][-1] > 0:
                ed += energyDemand[j][-1] * math.ceil(s / userMax[j][-1])
            else:
                ed += energyDemand[j][-1]
            
            revenue_perUser += rev[j][-1]
            s *= q[j][-1]
        
        rev_hourly = revenue_perUser * click_data[i]
        rev_percentage = revenue_perUser / maxRev if maxRev > 0 else 0
        
        # Write result
        row = [i, ed, 1.0, revenue_perUser, rev_hourly, rev_percentage, 
               click_data[i], region, workload, config, "high_performance"]
        
        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def run_simple_carbon_aware_baseline(config: str, ci_data: pd.Series,
                                  click_data: List[int], region: str,
                                  workload: str, output_dir: str = 'results') -> None:
    """Run the simple carbon aware baseline algorithm."""
    data, vars_ms, userMax, energyDemand, q, QoE, rev, indices = getConstantsFromBPMN(f"data/flightBooking_{config}.json")
    
    # Generate historical carbon budgets (cached)
    cache_key = f"{region}_{workload}_{config}"
    if cache_key not in _carbon_budgets_cache:
        highCB, averageCB, lowCB = generateAdaptiveCarbonBudgets(
            click_data, ci_data, f"data/flightBooking_{config}.json",
            use_historical=True, region=region, workload=f"{workload}.csv"
        )
        _carbon_budgets_cache[cache_key] = (highCB, averageCB, lowCB)
    else:
        highCB, averageCB, lowCB = _carbon_budgets_cache[cache_key]
    
    # Get microservice configurations
    ms_configs = get_microservice_configs(userMax, energyDemand, q, QoE, rev)
    
    for cb_idx, cb_name in enumerate(CARBON_BUDGET_TYPES):
        # Create output directory and file
        result_dir = f'{output_dir}/{region}/{workload}/{config}/simple_carbon_aware'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/{cb_name}result.csv'
        
        # Skip if result already exists
        if result_exists(result_file, len(click_data)):
            print(f"Skipping simple carbon aware: {region}/{workload}/{config}/{cb_name} (already exists)")
            continue
        
        # Get carbon budget
        carbonBudget = selectCarbonBudget(highCB, averageCB, lowCB, cb_idx)
        
        # Write header
        header = ["hour", "energy_demand", "QoE", "rev_per_user", "total_revenue", 
                 "rev_percentage", "carbon_budget", "user_requests", "carbon_emissions", 
                 "region", "workload", "config", "algorithm", "budget_type"]
        
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
        print(f"Running simple carbon aware: {region}/{workload}/{config}/{cb_name}")
        
        # Calculate maximum revenue
        maxRev = sum(max(rev[ms]) for ms in range(len(rev)))
        
        # Run for each hour
        for t in range(len(click_data)):
            eb = carbonBudget.iloc[t]
            
            # Test configurations in order: LP, NP, HP
            ed_hp = calculateEnergyDemand(ms_configs['highPower'], click_data[t])
            ed_np = calculateEnergyDemand(ms_configs['normalPower'], click_data[t])
            ed_lp = calculateEnergyDemand(ms_configs['lowPower'], click_data[t])
            
            # Select configuration based on carbon budget
            if ed_lp * ci_data.iloc[t] <= eb:
                if ed_np * ci_data.iloc[t] <= eb:
                    if ed_hp * ci_data.iloc[t] <= eb:
                        chosen_config = ms_configs['highPower']
                    else:
                        chosen_config = ms_configs['normalPower']
                else:
                    chosen_config = ms_configs['lowPower']
            else:
                chosen_config = ms_configs['lowPower']  # Fallback
            
            # Calculate metrics
            ed = calculateEnergyDemand(chosen_config, click_data[t])
            ce = ed * ci_data.iloc[t]
            
            # Calculate revenue and QoE
            revenue_perUser = sum(ms_config[4] for ms_config in chosen_config)
            rev_hourly = revenue_perUser * click_data[t]
            rev_percentage = revenue_perUser / maxRev if maxRev > 0 else 0
            
            q_hourly = 1.0
            for ms_config in chosen_config:
                q_hourly *= ms_config[0]
            userThroughput = q_hourly * click_data[t]
            
            qoe = sum(ms_config[3] for ms_config in chosen_config) / len(chosen_config)
            
            # Write result
            row = [t, ed, qoe, revenue_perUser, rev_hourly, rev_percentage, eb, click_data[t], ce, 
                   region, workload, config, "simple_carbon_aware", cb_name.rstrip('_')]
            
            with open(result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

def run_sequential_carbon_aware_baseline(config: str, ci_data: pd.Series,
                                      click_data: List[int], region: str,
                                      workload: str, output_dir: str = 'results') -> None:
    """Run the sequential carbon aware baseline algorithm."""
    data, vars_ms, userMax, energyDemand, q, QoE, rev, indices = getConstantsFromBPMN(f"data/flightBooking_{config}.json")
    
    # Generate historical carbon budgets (cached)
    cache_key = f"{region}_{workload}_{config}"
    if cache_key not in _carbon_budgets_cache:
        highCB, averageCB, lowCB = generateAdaptiveCarbonBudgets(
            click_data, ci_data, f"data/flightBooking_{config}.json",
            use_historical=True, region=region, workload=f"{workload}.csv"
        )
        _carbon_budgets_cache[cache_key] = (highCB, averageCB, lowCB)
    else:
        highCB, averageCB, lowCB = _carbon_budgets_cache[cache_key]
    
    # Get microservice configurations
    ms_configs = get_microservice_configs(userMax, energyDemand, q, QoE, rev)
    
    for cb_idx, cb_name in enumerate(CARBON_BUDGET_TYPES):
        # Create output directory and file
        result_dir = f'{output_dir}/{region}/{workload}/{config}/sequential_carbon_aware'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/{cb_name}result.csv'
        
        # Skip if result already exists
        if result_exists(result_file, len(click_data)):
            print(f"Skipping sequential carbon aware: {region}/{workload}/{config}/{cb_name} (already exists)")
            continue
        
        # Get carbon budget
        carbonBudget = selectCarbonBudget(highCB, averageCB, lowCB, cb_idx)
        
        # Write header
        header = ["hour", "energy_demand", "QoE", "rev_per_user", "total_revenue", 
                 "rev_percentage", "carbon_budget", "user_requests", "carbon_emissions", 
                 "region", "workload", "config", "algorithm", "budget_type"]
        
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
        print(f"Running sequential carbon aware: {region}/{workload}/{config}/{cb_name}")
        
        # Calculate maximum revenue
        maxRev = sum(max(rev[ms]) for ms in range(len(rev)))
        
        # Run for each hour
        for t in range(len(click_data)):
            eb = carbonBudget.iloc[t]
            
            # Sequential selection: start with HP, downgrade as needed
            selected_config = []
            remaining_budget = eb
            user_flow = click_data[t]
            
            for ms_idx in range(len(userMax)):
                best_config = None
                best_ed = float('inf')
                
                # Try configurations in order: HP, NP, LP
                for config_idx in [2, 1, 0]:  # HP, NP, LP
                    if config_idx < len(userMax[ms_idx]):
                        # Calculate energy demand for this microservice
                        instances = math.ceil(user_flow / userMax[ms_idx][config_idx]) if userMax[ms_idx][config_idx] > 0 else 1
                        ed = instances * energyDemand[ms_idx][config_idx]
                        ce = ed * ci_data.iloc[t]
                        
                        if ce <= remaining_budget:
                            best_config = config_idx
                            best_ed = ed
                            break
                
                if best_config is not None:
                    selected_config.append(best_config)
                    remaining_budget -= best_ed * ci_data.iloc[t]
                    user_flow *= q[ms_idx][best_config]
                else:
                    # Fallback to lowest power
                    selected_config.append(0)
                    user_flow *= q[ms_idx][0]
            
            # Calculate final metrics
            ed = 0
            revenue_perUser = 0
            qoe = 0
            q_hourly = 1.0
            user_flow = click_data[t]
            
            for ms_idx, config_idx in enumerate(selected_config):
                instances = math.ceil(user_flow / userMax[ms_idx][config_idx]) if userMax[ms_idx][config_idx] > 0 else 1
                ed += instances * energyDemand[ms_idx][config_idx]
                revenue_perUser += rev[ms_idx][config_idx]
                qoe += QoE[ms_idx][config_idx]
                q_hourly *= q[ms_idx][config_idx]
                user_flow *= q[ms_idx][config_idx]
            
            ce = ed * ci_data.iloc[t]
            rev_hourly = revenue_perUser * click_data[t]
            rev_percentage = revenue_perUser / maxRev if maxRev > 0 else 0
            qoe = qoe / len(selected_config)
            userThroughput = q_hourly * click_data[t]
            
            # Write result
            row = [t, ed, qoe, revenue_perUser, rev_hourly, rev_percentage, eb, click_data[t], ce, 
                   region, workload, config, "sequential_carbon_aware", cb_name.rstrip('_')]
            
            with open(result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

def run_experiments(regions: List[str], workloads: List[str],
                   configs: List[str], algorithms: List[str],
                   output_dir: str = 'results') -> None:
    """Run experiments for specified parameters."""
    total_experiments = len(regions) * len(workloads) * len(configs) * len(algorithms)
    experiment_count = 0

    print(f"Starting experiments at {datetime.now()}")
    print(f"Algorithms: {algorithms}")
    print(f"Regions: {regions}")
    print(f"Workloads: {workloads}")
    print(f"Configurations: {configs}")
    print("-" * 60)

    for region in regions:
        for workload in workloads:
            # Load data once per region/workload combination
            ci_data, click_data = load_data(region, workload)

            for config in configs:
                for algorithm in algorithms:
                    experiment_count += 1
                    print(f"[{experiment_count}/{total_experiments}] Running {algorithm} - {region}/{workload}/{config}")

                    if algorithm == 'optimized':
                        run_selective_optimization(config, ci_data, click_data, region, workload, output_dir)
                    elif algorithm == 'high_performance':
                        run_high_performance_baseline(config, click_data, region, workload, output_dir)
                    elif algorithm == 'simple_carbon_aware':
                        run_simple_carbon_aware_baseline(config, ci_data, click_data, region, workload, output_dir)
                    elif algorithm == 'sequential_carbon_aware':
                        run_sequential_carbon_aware_baseline(config, ci_data, click_data, region, workload, output_dir)

    print(f"\nExperiments completed at {datetime.now()}")
    print(f"Results saved to: {output_dir}")


def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description='Run carbon-aware optimization experiments')
    
    # Algorithm selection
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['optimized', 'high_performance', 'simple_carbon_aware', 'sequential_carbon_aware'],
                       default=['optimized', 'high_performance', 'simple_carbon_aware', 'sequential_carbon_aware'],
                       help='Algorithms to run')
    
    # Region selection
    parser.add_argument('--regions', nargs='+',
                       choices=list(CARBON_REGIONS.keys()),
                       default=list(CARBON_REGIONS.keys()),
                       help='Regions to evaluate')
    
    # Workload selection
    parser.add_argument('--workloads', nargs='+',
                       choices=list(WORKLOADS.keys()),
                       default=list(WORKLOADS.keys()),
                       help='Workloads to evaluate')
    
    # Configuration selection
    parser.add_argument('--configs', nargs='+',
                       choices=APP_CONFIGS,
                       default=APP_CONFIGS,
                       help='Application configurations to test')
    
    # Output directory
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()

    # Run experiments
    run_experiments(args.regions, args.workloads, args.configs, args.algorithms, args.output_dir)

    print("\n✅ All experiments completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()