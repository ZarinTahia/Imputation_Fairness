## OT Imputation Fairness
A research project exploring missing data imputation using Optimal Transport and its impact on fairness in machine learning models.
This repository investigates how imputing missing values with Optimal Transport affects biases in datasets and evaluates whether the imputed data improves fairness in downstream models. We use Conditional Mutual Information (CMI) to evaluate biases in the datasets. CMI measures the dependency between features and sensitive attributes, helping us identify potential sources of unfairness before and after imputation. The project is ongoing.

## Features
- Implements Optimal Transport (OT) imputation for tabular datasets.
- Injects missing values into datasets to simulate real-world scenarios.
- Evaluates bias and fairness metrics after imputation.
- Supports analysis on popular fairness benchmark datasets like Adult, COMPAS, German Credit, etc.

## Project Structure
- data/ – Original and processed datasets.
- experiments/ – Scripts for running OT imputation and fairness analysis.
- imputation/ – Optimal Transport imputer implementation.
- metrics/ – Functions for evaluating fairness and bias.
- code/notebooks/ – Dataset-specific analysis and visualizations.
- Output/ – Generated figures, tables, and logs.
- utils/ – Data loading and preprocessing utilities.
- requirements.txt – Required Python packages.
- README.md – Project overview and instructions.

## Contributing
The project is ongoing. Contributions are welcome:
- Add new datasets
- Extend fairness metrics
- Improve visualizations
- Please submit issues or pull requests.
