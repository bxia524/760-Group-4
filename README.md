# Group 4 Final Code Package

## Project
Explainable and Fair Machine Learning for Community-Based Diabetes Risk Prediction

## Authoritative files
- `Group4_Final_Diabetes_Project.ipynb`: main reproducible notebook used for the final results.
- `run_final_experiments.py`: script version of the same final pipeline. Running this script regenerates the CSV result files and PNG figures.
- `diabetes_012_health_indicators_BRFSS2015.csv`: CDC BRFSS 2015 diabetes health indicators dataset.
- `final_outputs/`: generated result tables, including model metrics, fairness gaps, confusion matrices, feature importance, SHAP outputs, and run summary.
- `final_charts/`: presentation figures generated from the result files in `final_outputs/`.

## How to run
Option 1: Notebook
1. Put all files in the same folder.
2. Open `Group4_Final_Diabetes_Project.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells from top to bottom.
4. The notebook writes result tables to `final_outputs/` and presentation figures to `final_charts/`.

Option 2: Script
```bash
python run_final_experiments.py
```
The script regenerates the same types of outputs in `final_outputs/` and `final_charts/`.

## Required packages
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib pillow
```
Notes:
- `imbalanced-learn` is not required because the final pipeline does not use SMOTE.
- `xgboost` is optional in the code. If XGBoost is unavailable, the code falls back to Random Forest for tree-based importance and SHAP explanation.
- `shap` is required for the notebook SHAP section. If SHAP is not installed, install it before running the explainability cells.

## Final method summary
The final pipeline does not rely on SMOTE. Instead, it uses:
- stratified train/validation/test splitting to preserve the original class imbalance;
- class-weighted Logistic Regression;
- balanced-subsample Random Forest;
- sample-weighted XGBoost when XGBoost is available;
- Neural Network / MLP as an additional baseline with early stopping;
- validation-based decision tuning to avoid extreme over-prediction of minority classes;
- imbalance-aware evaluation metrics, including balanced accuracy, macro-F1, class-specific recall, and predicted class distribution.

Validation and test sets are kept in the original imbalanced distribution. Weighting is applied only during model training or decision tuning, not to the held-out evaluation data.

## Output files
### `final_outputs/`
- `model_metrics.csv`: final test-set Accuracy, Balanced Accuracy, Macro-F1, class-specific recall, and predicted class rates.
- `selected_model_settings.csv`: selected hyperparameters and validation decision weights.
- `fairness_gaps.csv`: subgroup gaps across Sex, Age, and Income.
- `confusion_*.csv`: confusion matrices for each model.
- `classification_report_*.csv`: per-class precision, recall, and F1-score for each model.
- `feature_importance.csv`: built-in tree feature importance from XGBoost if available, otherwise Random Forest. This is not the same as SHAP.
- `shap_feature_importance.csv`: global mean absolute SHAP importance.
- `shap_global_bar.png`: global SHAP bar plot.
- `shap_summary_diabetes_class.png`: SHAP summary plot for the diabetes class.
- `shap_summary.json`: metadata for the SHAP explanation.
- `run_summary.json`: overall runtime, data split sizes, class weights, trained models, selected model, and output summary.

### `final_charts/`
The figures in `final_charts/` are generated from the current run outputs, not manually drawn.

| Figure | Generated from | Purpose |
|---|---|---|
| `class_distribution.png` | class proportions in the original, train, validation, and test sets | Shows that stratified splitting preserves the original class imbalance. |
| `model_performance.png` | `model_metrics.csv` | Compares final models using Accuracy, Balanced Accuracy, and Macro-F1 on the untouched test set. |
| `fairness_gaps.png` | `fairness_gaps.csv` | Shows the max-minus-min Macro-F1 subgroup gap for Sex, Age, and Income. |
| `feature_importance.png` | `feature_importance.csv` | Shows the top predictive features from the available tree model. This is built-in feature importance, not SHAP. |
| `rf_confusion_matrix.png` | Random Forest predictions on the test set | Shows detailed class-level errors for the selected balanced model. |
| `rf_pred_distribution.png` | Random Forest predicted labels on the test set | Checks that predictions are not collapsed into the majority class. |
| `rf_recall.png` | Random Forest class-specific recall on the test set | Shows that diabetes is detected better than prediabetes, while prediabetes remains the hardest class. |
| `shap_global_bar.png` | `shap_feature_importance.csv` | Shows the most influential features according to mean absolute SHAP values. |
| `shap_summary_diabetes_class.png` | SHAP values on a sampled test set | Shows how feature values affect the diabetes-class prediction. |

## Interpretation of the main charts
1. The class distribution chart confirms that the train, validation, and test sets keep almost the same class proportions as the original BRFSS dataset. This is important because the final test evaluation reflects the real imbalanced setting.
2. The model performance chart shows why accuracy alone is not enough. A model can achieve higher accuracy by focusing on the majority healthy class, while balanced accuracy and macro-F1 provide a fairer view of performance across all three classes.
3. The Random Forest prediction distribution chart shows that the selected balanced model does not simply predict every record as healthy. It still predicts prediabetes and diabetes cases, which directly responds to the feedback that imbalance handling should not produce extreme classification behaviour.
4. The Random Forest recall chart shows that diabetes is detected much better than prediabetes. Prediabetes remains the hardest class because it is the smallest class in the data and is clinically close to both healthy and diabetes groups.
5. The fairness gap chart shows that Age has the largest subgroup performance gap. Sex has the smallest gap, while Income has a moderate gap. This suggests that age-related performance differences should be interpreted carefully in public-health screening.
6. The feature-importance chart identifies the strongest predictive variables used by the tree model, such as HighBP, GenHlth, HighChol, Age, BMI, and Income. This chart is useful for model interpretation but should not be described as SHAP.
7. The SHAP figures provide the final explainability output. The SHAP bar plot ranks features by mean absolute contribution, while the SHAP summary plot explains how those features push predictions toward or away from the diabetes class.

## AI use acknowledgement
GenAI was used to help organise the final presentation structure, improve wording, and format reproducible code. The experimental design, dataset, model outputs, interpretation, and final claims were checked and finalised by the group.
