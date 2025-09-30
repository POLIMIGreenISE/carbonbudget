"""
Pareto Front Experiment: Generate results with 10 different budget levels
for Germany + wiki_en to enable proper Pareto front analysis.

Budget levels: 100%, 90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10% of HIGH scenario
"""
import math
import os
import pandas as pd
from run_experiments import ExperimentRunner, generateHistoricalCarbonBudgets, extractRequestTraceByYear

def generate_pareto_budgets():
    """Generate 10 different budget levels for Germany + wiki_en."""
    
    print("Generating Pareto budget levels...")
    
    # Load base data
    region = 'DE'
    workload = 'wiki_en'
    
    # Generate HIGH budget for all configs
    configs = ['HH', 'HL', 'LH', 'LL']
    budget_levels = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    
    # Load carbon intensity and workload data
    ci_file = f'data/{region}_2023_hourly.csv'
    df_carbon = pd.read_csv(ci_file)
    ci_data = df_carbon['Carbon Intensity gCO₂eq/kWh (direct)']
    ci_data = ci_data.fillna(ci_data.mean())
    
    workload_file = f'data/{workload}.csv'
    clickData_hourly = extractRequestTraceByYear(workload_file, 2023)
    
    # Normalize workload
    if len(clickData_hourly) > len(ci_data):
        clickData_hourly = clickData_hourly[:len(ci_data)]
    elif len(clickData_hourly) < len(ci_data):
        multiplier = len(ci_data) // len(clickData_hourly) + 1
        clickData_hourly = (clickData_hourly * multiplier)[:len(ci_data)]
    
    # Generate budgets for each config
    pareto_budgets = {}
    for config in configs:
        print(f"  Generating budgets for {config}...")
        
        # Get HIGH budget (100% level)
        high_budget, _, _ = generateHistoricalCarbonBudgets(
            region, f'{workload}.csv', f'data/flightBooking_{config}.json'
        )
        
        # Create budgets for different percentage levels
        config_budgets = {}
        for level in budget_levels:
            config_budgets[level] = high_budget * (level / 100.0)
        
        pareto_budgets[config] = config_budgets
    
    return pareto_budgets, ci_data, clickData_hourly

def run_pareto_experiments():
    """Run experiments with 10 budget levels."""
    
    print("PARETO FRONT EXPERIMENTS")
    print("=" * 50)
    print("Region: DE (Germany)")
    print("Workload: wiki_en (English Wikipedia)")
    print("Budget levels: 100%, 90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%")
    print("Algorithms: optimized, high_performance, simple_carbon_aware, sequential_carbon_aware")
    print("Configurations: HH, HL, LH, LL")
    print("-" * 50)
    
    # Generate budgets
    pareto_budgets, ci_data, clickData_hourly = generate_pareto_budgets()
    
    # Create custom experiment runner
    runner = ExperimentRunner('pareto_results')
    
    # Run experiments for each budget level
    algorithms = ['optimized', 'high_performance', 'simple_carbon_aware', 'sequential_carbon_aware']
    configs = ['HH', 'HL', 'LH', 'LL']
    budget_levels = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    
    total_experiments = len(algorithms) * len(configs) * len(budget_levels)
    experiment_count = 0
    
    for config in configs:
        for budget_level in budget_levels:
            for algorithm in algorithms:
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}] Running {algorithm} - {config} - {budget_level}%")
                
                # Create output directory
                result_dir = f'pareto_results/DE/wiki_en/{config}/{algorithm}'
                os.makedirs(result_dir, exist_ok=True)
                result_file = f'{result_dir}/BUDGET_{budget_level}PCT_result.csv'
                
                # Skip if result already exists
                if os.path.exists(result_file):
                    try:
                        existing_df = pd.read_csv(result_file)
                        if len(existing_df) == len(clickData_hourly):
                            print(f"  ✓ Skipping (already exists)")
                            continue
                    except:
                        pass
                
                # Get carbon budget for this level
                carbon_budget = pareto_budgets[config][budget_level]
                
                # Run algorithm with custom budget
                try:
                    if algorithm == 'optimized':
                        runner.run_selective_optimization_custom(
                            config, ci_data, clickData_hourly, 'DE', 'wiki_en', 
                            carbon_budget, result_file, budget_level
                        )
                    elif algorithm == 'high_performance':
                        runner.run_high_performance_baseline_custom(
                            config, clickData_hourly, 'DE', 'wiki_en', result_file, budget_level
                        )
                    elif algorithm == 'simple_carbon_aware':
                        runner.run_simple_carbon_aware_baseline_custom(
                            config, ci_data, clickData_hourly, 'DE', 'wiki_en', 
                            carbon_budget, result_file, budget_level
                        )
                    elif algorithm == 'sequential_carbon_aware':
                        runner.run_sequential_carbon_aware_baseline_custom(
                            config, ci_data, clickData_hourly, 'DE', 'wiki_en', 
                            carbon_budget, result_file, budget_level
                        )
                    
                    print(f"  ✓ Completed")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
    
    print(f"\n✅ Pareto experiments completed!")
    print(f"Results saved to: pareto_results/")

# Add custom methods to ExperimentRunner
def run_selective_optimization_custom(self, config, ci_data, clickData_hourly, region, workload, 
                                    carbon_budget, result_file, budget_level):
    """Run selective optimization with custom budget."""
    from run_experiments import getConstantsFromBPMN, optimizationHourly
    import csv
    
    data, vars_ms, userMax, energyDemand, q, QoE, rev, indices = getConstantsFromBPMN(f"data/flightBooking_{config}.json")
    
    # Write header
    header = ["hour", "energy_demand", "carbon_budget", "user_requests", 
             "carbon_emissions", "objective", "QoE", "rev_per_user", 
             "total_revenue", "rev_percentage", "budget_level", "region", 
             "workload", "config", "algorithm"]
    
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    # Run optimization for each hour
    for t in range(len(clickData_hourly)):
        # Convert carbon budget to energy budget
        eb = carbon_budget.iloc[t] / ci_data.iloc[t]

        # Run optimization
        result = optimizationHourly(eb, clickData_hourly[t], indices, userMax,
                                  energyDemand, q, QoE, rev, 0.5, 0.5)

        qValue, userThroughput, ed, qoe, revenue, objective = result
        
        # Calculate metrics
        ce = ed * ci_data.iloc[t]
        rev_per_user = revenue / clickData_hourly[t] if clickData_hourly[t] > 0 else 0
        maxRev = sum(max(rev[ms]) for ms in range(len(rev)))
        rev_percentage = revenue / maxRev if maxRev > 0 else 0
        
        # Save result
        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([t, ed, carbon_budget.iloc[t], clickData_hourly[t], ce, objective, 
                           qoe, rev_per_user, revenue, rev_percentage, budget_level, 
                           region, workload, config, "optimized"])

def run_high_performance_baseline_custom(self, config, clickData_hourly, region, workload, 
                                       result_file, budget_level):
    """Run high performance baseline with budget level tracking."""
    from run_experiments import getConstantsFromBPMN
    import csv
    import math
    
    data, vars_ms, userMax, energyDemand, q, QoE, rev, indices = getConstantsFromBPMN(f"data/flightBooking_{config}.json")
    
    # Write header
    header = ["hour", "energy_demand", "QoE", "rev_per_user", "total_revenue", 
             "rev_percentage", "user_requests", "budget_level", "region", 
             "workload", "config", "algorithm"]
    
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    # Calculate maximum revenue
    maxRev = sum(rev[ms][-1] for ms in range(len(rev)))
    
    # Run for each hour
    for i in range(len(clickData_hourly)):
        ed = 0
        revenue_perUser = 0
        s = clickData_hourly[i]
        
        # Calculate energy demand and revenue for high performance mode
        for j in range(len(q)):
            if userMax[j][-1] > 0:
                ed += energyDemand[j][-1] * math.ceil(s / userMax[j][-1])
            else:
                ed += energyDemand[j][-1]
            
            revenue_perUser += rev[j][-1]
            s *= q[j][-1]
        
        rev_hourly = revenue_perUser * clickData_hourly[i]
        rev_percentage = revenue_perUser / maxRev if maxRev > 0 else 0
        
        # Write result
        row = [i, ed, 1.0, revenue_perUser, rev_hourly, rev_percentage, 
               clickData_hourly[i], budget_level, region, workload, config, "high_performance"]
        
        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def run_simple_carbon_aware_baseline_custom(self, config, ci_data, clickData_hourly, region, workload, 
                                          carbon_budget, result_file, budget_level):
    """Run simple carbon aware baseline with custom budget."""
    from run_experiments import getConstantsFromBPMN
    import csv
    import math
    
    data, vars_ms, userMax, energyDemand, q, QoE, rev, indices = getConstantsFromBPMN(f"data/flightBooking_{config}.json")
    
    # Get microservice configurations
    ms_configs = self._get_microservice_configs(userMax, energyDemand, q, QoE, rev)
    
    # Write header
    header = ["hour", "energy_demand", "QoE", "rev_per_user", "total_revenue", 
             "rev_percentage", "carbon_budget", "user_requests", "carbon_emissions", 
             "budget_level", "region", "workload", "config", "algorithm"]
    
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    # Calculate maximum revenue
    maxRev = sum(max(rev[ms]) for ms in range(len(rev)))
    
    # Run for each hour
    for t in range(len(clickData_hourly)):
        eb = carbon_budget.iloc[t]
        
        # Test configurations in order: LP, NP, HP
        ed_hp = self._calculate_energy_demand_simple(ms_configs['highPower'], clickData_hourly[t])
        ed_np = self._calculate_energy_demand_simple(ms_configs['normalPower'], clickData_hourly[t])
        ed_lp = self._calculate_energy_demand_simple(ms_configs['lowPower'], clickData_hourly[t])
        
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
        ed = self._calculate_energy_demand_simple(chosen_config, clickData_hourly[t])
        ce = ed * ci_data.iloc[t]
        
        # Calculate revenue and QoE
        revenue_perUser = sum(ms_config[4] for ms_config in chosen_config)
        rev_hourly = revenue_perUser * clickData_hourly[t]
        rev_percentage = revenue_perUser / maxRev if maxRev > 0 else 0
        
        qoe = sum(ms_config[3] for ms_config in chosen_config) / len(chosen_config)
        
        # Write result
        row = [t, ed, qoe, revenue_perUser, rev_hourly, rev_percentage, eb, 
               clickData_hourly[t], ce, budget_level, region, workload, config, "simple_carbon_aware"]
        
        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def run_sequential_carbon_aware_baseline_custom(self, config, ci_data, clickData_hourly, region, workload, 
                                              carbon_budget, result_file, budget_level):
    """Run sequential carbon aware baseline with custom budget."""
    from run_experiments import getConstantsFromBPMN
    import csv
    import math
    
    data, vars_ms, userMax, energyDemand, q, QoE, rev, indices = getConstantsFromBPMN(f"data/flightBooking_{config}.json")
    
    # Write header
    header = ["hour", "energy_demand", "QoE", "rev_per_user", "total_revenue", 
             "rev_percentage", "carbon_budget", "user_requests", "carbon_emissions", 
             "budget_level", "region", "workload", "config", "algorithm"]
    
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    # Calculate maximum revenue
    maxRev = sum(max(rev[ms]) for ms in range(len(rev)))
    
    # Run for each hour
    for t in range(len(clickData_hourly)):
        eb = carbon_budget.iloc[t]
        
        # Sequential selection: start with HP, downgrade as needed
        selected_config = []
        remaining_budget = eb
        user_flow = clickData_hourly[t]
        
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
        user_flow = clickData_hourly[t]
        
        for ms_idx, config_idx in enumerate(selected_config):
            instances = math.ceil(user_flow / userMax[ms_idx][config_idx]) if userMax[ms_idx][config_idx] > 0 else 1
            ed += instances * energyDemand[ms_idx][config_idx]
            revenue_perUser += rev[ms_idx][config_idx]
            qoe += QoE[ms_idx][config_idx]
            user_flow *= q[ms_idx][config_idx]
        
        ce = ed * ci_data.iloc[t]
        rev_hourly = revenue_perUser * clickData_hourly[t]
        rev_percentage = revenue_perUser / maxRev if maxRev > 0 else 0
        qoe = qoe / len(selected_config)
        
        # Write result
        row = [t, ed, qoe, revenue_perUser, rev_hourly, rev_percentage, eb, 
               clickData_hourly[t], ce, budget_level, region, workload, config, "sequential_carbon_aware"]
        
        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def _calculate_energy_demand_simple(configuration, user_demand):
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

# Add methods to ExperimentRunner
ExperimentRunner.run_selective_optimization_custom = run_selective_optimization_custom
ExperimentRunner.run_high_performance_baseline_custom = run_high_performance_baseline_custom
ExperimentRunner.run_simple_carbon_aware_baseline_custom = run_simple_carbon_aware_baseline_custom
ExperimentRunner.run_sequential_carbon_aware_baseline_custom = run_sequential_carbon_aware_baseline_custom
ExperimentRunner._calculate_energy_demand_simple = _calculate_energy_demand_simple

if __name__ == "__main__":
    run_pareto_experiments()