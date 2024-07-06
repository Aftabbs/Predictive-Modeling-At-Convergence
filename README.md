# Predictive Modeling Project: Predicting Fund Launching Managers

## Overview
This project aims to predict managers likely to launch new funds based on historical data. The dataset spans from 2016 to 2023, and predictions are made for managers expected to launch funds in 2024. The dataset is prepared by domain experts and SMEs of our organization, containing 110 features.

## Objective
Identify managers who are likely to launch new funds in 2024 using machine learning models. This is a classification problem where the target variable is imbalanced due to the rarity of fund launches by managers.

## Challenges
- Imbalanced dataset: Few instances of fund launches compared to non-launches.
- Real-world data complexities.
- Precision-recall tradeoff: Focus on maximizing recall while maintaining acceptable precision.

## Approach
### Model Selection
Ensemble methods were employed:
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier
- Voting Classifier combining the above models.

### Feature Selection
Top N features selected based on feature importances from ensemble models.

### Metrics
- **Accuracy**: 75.63%
- **Precision**: 59.45%
- **Recall**: 62.14%
- **F1-Score**: 60.76%
- **ROC-AUC**: 71.83%

### Hyperparameter Tuning
Two approaches were utilized:
1. **RandomizedSearchCV**:
   - Best Threshold: 0.496
   - Accuracy: 73.61%
   - Precision: 55.39%
   - Recall: 69.68%
   - F1-Score: 61.72%
   - ROC-AUC: 80.75%
   
2. **GridSearchCV**:
   - Accuracy: 72.91%
   - Precision: 54.30%
   - Recall: 71.16%
   - F1-Score: 61.60%
   - ROC-AUC: 80.47%

## Implementation
The project includes an end-to-end Python script:
1. **Data Preprocessing**: Loading and preparing the dataset.
2. **Model Training**: Training ensemble models and optimizing hyperparameters.
3. **Prediction**: Generating predictions for 2024 with associated metrics.
4. **Output**: Excel file download containing predictions and metrics.

## Further Enhancements
1. Introducing Additional Features to the dataset
2. Adding more instances and increasing the size of the data
3. Using Deep Learning techniques to learn indepth data patterns and identify data complexities

## Usage
To run the script:
1. Install required Python packages.
2. Provide the dataset as input.
3. Execute the script to train models and obtain 2024 predictions.

## Conclusion
This README provides an overview of our approach, challenges faced, and outcomes achieved in predicting fund-launching managers for 2024. For detailed implementation and results, refer to the provided Python script and associated files.

