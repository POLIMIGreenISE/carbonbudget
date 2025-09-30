import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from run_experiments import APP_CONFIGS

# Import shared constants from create_boxplot
from create_boxplot import ALGORITHMS, LABELS, COLORS

# Configuration
BUDGET_LEVELS = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

sns.set_palette("husl")
sns.set_style("ticks")


def load_pareto_data(results_dir='pareto_results', region='DE', workload='wiki_en'):
    """Load all Pareto experiment results."""
    data_frames = []

    for config in APP_CONFIGS:
        for algorithm in ALGORITHMS:
            for budget_level in BUDGET_LEVELS:
                file_path = Path(results_dir) / region / workload / config / algorithm / f'BUDGET_{budget_level}PCT_result.csv'

                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        df['config'] = config
                        df['algorithm'] = algorithm
                        df['budget_level'] = budget_level
                        data_frames.append(df)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        print(f"✓ Loaded {len(combined_df):,} data points")
        return combined_df
    else:
        print("❌ No data found! Run run_pareto_experiments.py first.")
        return None


def calculate_summary_statistics(df):
    """Calculate summary statistics for Pareto analysis."""
    if df is None:
        return None

    summary_data = []

    for config in APP_CONFIGS:
        for algorithm in ALGORITHMS:
            for budget_level in BUDGET_LEVELS:
                subset = df[
                    (df['config'] == config) &
                    (df['algorithm'] == algorithm) &
                    (df['budget_level'] == budget_level)
                ]

                if len(subset) > 0:
                    # Calculate utilization
                    if 'carbon_emissions' in subset.columns and 'carbon_budget' in subset.columns:
                        utilization = subset['carbon_emissions'] / subset['carbon_budget']
                    else:
                        utilization = subset['energy_demand'] / subset['energy_demand'].max()

                    qoe = subset.get('QoE', pd.Series([1.0] * len(subset)))
                    rev_pct = subset.get('rev_percentage', pd.Series([1.0] * len(subset)))

                    summary_data.append({
                        'config': config,
                        'algorithm': algorithm,
                        'budget_level': budget_level,
                        'mean_utilization': utilization.mean(),
                        'mean_qoe': qoe.mean(),
                        'mean_revenue_pct': rev_pct.mean(),
                    })

    return pd.DataFrame(summary_data)


def plot_qoe_vs_budget(summary_df, config='LL', save_path=None):
    """Create QoE vs Budget Utilization plot."""
    config_data = summary_df[summary_df['config'] == config].copy()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    for algorithm in ALGORITHMS:
        algo_data = config_data[config_data['algorithm'] == algorithm].sort_values('budget_level', ascending=False)

        if len(algo_data) > 0:
            ax.scatter(algo_data['mean_utilization'], algo_data['mean_qoe'],
                      color=COLORS[algorithm], s=30, alpha=1, label=LABELS[algorithm])
            ax.plot(algo_data['mean_utilization'], algo_data['mean_qoe'],
                   color=COLORS[algorithm], alpha=0.5, linewidth=1, linestyle='--')

            # Annotate key budget levels
            for _, row in algo_data.iterrows():
                if row['budget_level'] in [10, 30, 100]:
                    if row['budget_level'] == 100 and algorithm == "optimized":
                        continue
                    ax.annotate(f"{row['budget_level']}%",
                               (row['mean_utilization'], row['mean_qoe']),
                               xytext=(0, 4), textcoords='offset points', alpha=0.8)

    ax.set_xlabel('Carbon Budget Utilization')
    ax.set_ylabel('Quality of Experience (QoE)')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.legend(["OSCA", "SCA", "SeqCA", "HPE (not showing because utilization >>1)"],
              loc='lower left', frameon=False, handletextpad=0.1)
    ax.grid(True, alpha=0.5)
    sns.despine(fig)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_qoe_vs_revenue(summary_df, config='LL', save_path=None):
    """Create QoE vs Revenue plot."""
    config_data = summary_df[summary_df['config'] == config].copy()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    for algorithm in ALGORITHMS:
        algo_data = config_data[config_data['algorithm'] == algorithm].sort_values('budget_level', ascending=False)

        if len(algo_data) > 0:
            ax.scatter(algo_data['mean_revenue_pct'], algo_data['mean_qoe'],
                      s=30, color=COLORS[algorithm], alpha=1, label=LABELS[algorithm])
            ax.plot(algo_data['mean_revenue_pct'], algo_data['mean_qoe'],
                   color=COLORS[algorithm], alpha=0.5, linewidth=1, linestyle='--')

            # Annotate key budget levels
            for _, row in algo_data.iterrows():
                if row['budget_level'] in [10, 30, 100]:
                    if row['budget_level'] == 100 and algorithm == "simple_carbon_aware":
                        continue
                    if algorithm == "high_performance":
                        continue
                    ax.annotate(f"{row['budget_level']}%",
                               (row['mean_revenue_pct'], row['mean_qoe']),
                               xytext=(4, -3), textcoords='offset points', alpha=0.8)

    ax.set_xlabel('Revenue (% of Maximum)')
    ax.set_ylabel('Quality of Experience (QoE)')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', frameon=False, handletextpad=0.1)
    ax.grid(True, alpha=0.5)
    sns.despine(fig)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def main():
    """Generate Pareto front plots."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    # Load data
    print("Loading Pareto data...")
    pareto_data = load_pareto_data()
    if pareto_data is None:
        return

    # Calculate summary statistics
    print("Calculating summary statistics...")
    summary = calculate_summary_statistics(pareto_data)

    # Generate plots for each configuration
    for config in APP_CONFIGS:
        print(f"\nGenerating plots for {config}...")
        plot_qoe_vs_budget(summary, config=config, save_path=f'plots/pareto_qoe_vs_budget_{config}.pdf')
        plot_qoe_vs_revenue(summary, config=config, save_path=f'plots/pareto_qoe_vs_revenue_{config}.pdf')

    print("\n✅ All plots generated in plots/ directory!")


if __name__ == "__main__":
    main()