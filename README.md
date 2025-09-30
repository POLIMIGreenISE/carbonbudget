# Adaptive Green Cloud Applications

Code and data for the paper:

- Monica Vitali, Philipp Wiesner, Kevin Kreutz, Roberto Gandola. "[Adaptive green cloud applications: Balancing emissions, revenue, and user experience through approximate computing](https://doi.org/10.1016/j.future.2025.108143)". *Future Generation Computer Systems*, Volume 176, 2026.

## Setup

Requires [Gurobi Optimizer](https://www.gurobi.com/) (here you can get an [academic license](https://www.gurobi.com/academia)).

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
grbgetkey <your-license-key>
```

## Run and Analyze Experiments

```bash
# Run experiments
python run_experiments.py
python run_pareto_experiments.py

# Generate plots (saved to plots/)
python create_boxplot.py
python create_pareto_plots.py
```

## Bibtex

If you use this code for your research, please cite:

```bibtex
@article{vitali2026adaptive,
  author = {Monica Vitali and Philipp Wiesner and Kevin Kreutz and Roberto Gandola},
  title = {Adaptive green cloud applications: Balancing emissions, revenue, and user experience through approximate computing},
  journal = {Future Generation Computer Systems},
  volume = {176},
  pages = {108143},
  year = {2026},
  doi = {https://doi.org/10.1016/j.future.2025.108143}
}
```
