# Project Context: Time Series Anomaly Detection Foundations

This is a foundations-first learning project focused on building domain-agnostic intuition for time series anomaly detection. It progresses from simple fixed thresholds to deep learning methods.

## Directory Structure & Navigation
- **`docs/plan/`**: Contains the project overview and detailed instructions for each module. 
  - **Start here:** `docs/plan/00-project-overview.md` is the source of truth for the project's current status and roadmap.
- **`notebooks/`**: Contains the actual implementation. Notebooks should be numbered sequentially to match the modules (e.g., `01-eda-fixed-thresholds.ipynb`, `02-statistical-baselines.ipynb`).
- **`data/`**: Contains the datasets used across the modules (NAB, SKAB, SWaT). **Do not modify raw data files.**
- **`venv/`**: Python virtual environment.

## Current Workflow
1. Check `docs/plan/00-project-overview.md` to see which module is currently active or up next.
2. Read the specific module's markdown file in `docs/plan/` (e.g., `02-module-2-statistical-baselines.md`) to understand the requirements, algorithms, and expected deliverables.
3. Implement the solution in the corresponding Jupyter Notebook in the `notebooks/` directory.
4. Update the "Status" table in `docs/plan/00-project-overview.md` when a module is completed.

## Coding Conventions & Technical Stack
- **Language:** Python
- **Key Libraries:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`, `pytorch`.
- **Implementation Style:** Code should be written in Jupyter Notebooks. Prioritize clear markdown explanations, data visualizations (especially plotting anomaly scores against ground truth labels), and empirical comparisons between algorithms.
- **Reproducibility:** Always set random seeds (e.g., `random_state=42`, `torch.manual_seed(42)`) when using stochastic algorithms like Isolation Forest or training neural networks.
