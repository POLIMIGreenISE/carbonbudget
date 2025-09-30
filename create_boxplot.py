import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from statistics import mean

from run_experiments import (
    generateAdaptiveCarbonBudgets,
    extractRequestTraceByYear,
    CARBON_REGIONS,
    WORKLOADS,
    APP_CONFIGS,
    CARBON_BUDGET_TYPES
)

BUDGET_NAMES = ['HIGH', 'AVERAGE', 'LOW']
COLORS = {
    'optimized': '#2CC820',
    'simple_carbon_aware': '#22CAD8',
    'sequential_carbon_aware': 'orange',
    'high_performance': '#FB5534',
}
LABELS = {
    'optimized': 'OSCA',
    'simple_carbon_aware': 'SCA',
    'sequential_carbon_aware': 'SeqCA',
    'high_performance': 'HPE',
}
ALGORITHMS = list(LABELS.keys())

def load_experiment_data(region='DE', workload='wiki_en', year=2023):
    """Load carbon intensity and workload data."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load carbon intensity data
    ci_file = os.path.join(project_root, "data", f"{region}_{year}_hourly.csv")
    df_carbon = pd.read_csv(ci_file)
    ci_data = df_carbon['Carbon Intensity gCO₂eq/kWh (direct)']
    ci_data = ci_data.fillna(ci_data.mean())

    # Load workload data
    workload_file = os.path.join(project_root, "data", f"{workload}.csv")
    clickData_hourly = extractRequestTraceByYear(workload_file, year)

    # Normalize workload to match carbon intensity data length
    if len(clickData_hourly) > len(ci_data):
        clickData_hourly = clickData_hourly[:len(ci_data)]
    elif len(clickData_hourly) < len(ci_data):
        multiplier = len(ci_data) // len(clickData_hourly) + 1
        clickData_hourly = (clickData_hourly * multiplier)[:len(ci_data)]

    return ci_data, clickData_hourly

def load_experiment_results(region='DE', workload='wiki_en', results_dir=None):
    """Load experiment results for all algorithms and configurations."""
    if results_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, 'results')

    results = {}

    for budget_idx, budget_type in enumerate(CARBON_BUDGET_TYPES):
        budget_name = BUDGET_NAMES[budget_idx]
        results[budget_name] = {}

        for config in APP_CONFIGS:
            results[budget_name][config] = {}

            for algo, subdir in [('optimized', 'optimized'), ('high_performance', 'high_performance'),
                               ('simple_carbon_aware', 'simple_carbon_aware'), ('sequential_carbon_aware', 'sequential_carbon_aware')]:
                try:
                    if algo == 'high_performance':
                        file_path = f"{results_dir}/{region}/{workload}/{config}/{subdir}/baseline_result.csv"
                    else:
                        file_path = f"{results_dir}/{region}/{workload}/{config}/{subdir}/{budget_type}result.csv"
                    results[budget_name][config][algo] = pd.read_csv(file_path)
                except FileNotFoundError:
                    results[budget_name][config][algo] = None

    return results

def calculate_utilization(budget_name, results, ci_data, clickData_hourly, region='DE', workload='wiki_en'):
    """Calculate carbon budget utilization for all algorithms and configs."""
    # Pre-calculate historical carbon budgets for all configs
    config_budgets = {}
    for config in APP_CONFIGS:
        high, average, low = generateAdaptiveCarbonBudgets(
            clickData_hourly, ci_data, f'data/flightBooking_{config}.json',
            use_historical=True, region=region, workload=f'{workload}.csv'
        )
        config_budgets[config] = {'HIGH': high, 'AVERAGE': average, 'LOW': low}

    # Initialize storage
    utilization = {algo: {config: [] for config in APP_CONFIGS + ['avg']} for algo in ALGORITHMS}
    overshoots = {algo: {config: 0 for config in APP_CONFIGS + ['avg']} for algo in ALGORITHMS}

    # Calculate for each hour
    max_hours = min(8760, len(ci_data))

    for hour in range(max_hours):
        hour_utilization = {algo: [] for algo in ALGORITHMS}

        for algo in ALGORITHMS:
            for config in APP_CONFIGS:
                if results[budget_name][config][algo] is not None:
                    if hour < len(results[budget_name][config][algo]):
                        # Calculate carbon emissions
                        if algo == 'optimized':
                            ce = results[budget_name][config][algo]['carbon_emissions'].iloc[hour]
                        elif algo == 'high_performance':
                            ed = results[budget_name][config][algo]['energy_demand'].iloc[hour]
                            ce = ed * ci_data.iloc[hour]
                        else:
                            ce = results[budget_name][config][algo]['carbon_emissions'].iloc[hour]

                        # Get budget value
                        budget_value = config_budgets[config][budget_name].iloc[hour]

                        # Calculate utilization ratio
                        util_ratio = ce / budget_value
                        utilization[algo][config].append(util_ratio)
                        hour_utilization[algo].append(util_ratio)

                        # Count overshoots
                        if util_ratio > 1.0:
                            overshoots[algo][config] += 1

        # Calculate average across configurations
        for algo in ALGORITHMS:
            if hour_utilization[algo]:
                avg_util = mean(hour_utilization[algo])
                utilization[algo]['avg'].append(avg_util)

                if avg_util > 1.0:
                    overshoots[algo]['avg'] += 1

    return utilization, overshoots

def analyze_scenario(region='DE', workload='wiki_en', experiment_year=2023, results_dir=None, budget_names=None):
    """Complete analysis pipeline for a scenario."""
    if budget_names is None:
        budget_names = BUDGET_NAMES

    print(f"Analyzing scenario: {region}/{workload}/{experiment_year}")

    # Load data
    ci_data, clickData_hourly = load_experiment_data(region, workload, experiment_year)
    results = load_experiment_results(region, workload, results_dir)

    # Calculate utilization for each budget type
    utilization_data = {}
    overshoot_data = {}

    for budget_name in budget_names:
        print(f"Calculating utilization for {budget_name} budget...")
        utilization_data[budget_name], overshoot_data[budget_name] = calculate_utilization(
            budget_name, results, ci_data, clickData_hourly, region, workload
        )

    print("Analysis completed!")

    return {
        'utilization_data': utilization_data,
        'overshoot_data': overshoot_data,
    }

def create_boxplot(utilization_data, budget_names=None, title_suffix="", save_path=None, y_limit=2.0):
    """Create side-by-side box plots for carbon budget utilization."""
    sns.set_style("darkgrid")

    if budget_names is None:
        budget_names = BUDGET_NAMES

    fig, axes = plt.subplots(1, len(budget_names), figsize=(12, 2.5))
    if len(budget_names) == 1:
        axes = [axes]

    configs = ['HH', 'HL', 'LH', 'LL']

    for idx, budget_name in enumerate(budget_names):
        ax = axes[idx]
        box_data, box_labels, box_colors = [], [], []

        for algo in ALGORITHMS:
            combined_data = []
            for config in configs:
                if utilization_data[budget_name][algo][config]:
                    combined_data.extend(utilization_data[budget_name][algo][config])

            if combined_data:
                box_data.append(combined_data)
                box_labels.append(LABELS[algo])
                box_colors.append(COLORS[algo])

        # Create boxplot
        box_plot = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                             flierprops=dict(marker='o', markerfacecolor='black',
                                           markersize=1, markeredgecolor='none'))

        # Style boxes
        for patch, color in zip(box_plot['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Configure plot
        ax.set_ylim(0, y_limit)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1)

        # Set labels
        y_labels = [f'{int(x * 50)}%' for x in range(0, int(y_limit * 2) + 1)]
        y_ticks = np.linspace(0, y_limit, len(y_labels))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        if idx == 0:
            ax.set_ylabel(title_suffix.split("/")[0] + '\n\nCarbon Budget Utilization')
        ax.set_title(f'{budget_name} Budget', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig

def main():
    """Generate boxplots for all region/workload combinations."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    results = {}

    # Analyze all scenarios
    for region in CARBON_REGIONS.keys():
        results[region] = {}
        for workload in WORKLOADS.keys():
            print(f"Analyzing {region}/{workload}")
            analysis = analyze_scenario(region=region, workload=workload, experiment_year=2023)
            results[region][workload] = analysis['utilization_data']

    # Generate boxplots
    for region in CARBON_REGIONS.keys():
        for workload in WORKLOADS.keys():
            save_path = f'plots/boxplot_{region}_{workload}.pdf'
            fig = create_boxplot(results[region][workload], title_suffix=f"{region}/{workload}",
                               save_path=save_path)
            plt.show()

    print("\n✅ All boxplots generated in plots/ directory!")

if __name__ == "__main__":
    main()