import os
import subprocess
import sys

try:
    import openpyxl
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from lightgbm import LGBMClassifier
import numpy as np
import warnings
import traceback

warnings.filterwarnings("ignore")

print("All libraries are imported")

def calculate_odds_ratios(model, feature_names):
    """Calculate odds ratios for logistic regression coefficients"""
    odds_ratios = pd.Series(np.exp(model.coef_[0]), index=feature_names)
    return odds_ratios

def main():
    try:
        df = pd.read_csv(r"C:\Users\Mohammed Aftab\Desktop\Desktop\Predictive Modeling Convergence\Likely to Launch_Auditor\Predictive_Likely_to_launch.csv")
        print("Data loaded successfully")

        df['AsofDate'] = pd.to_datetime(df['AsofDate'])
        filtered_df = df[df['AsofDate'].dt.year != 2024]  #Training data till year 2023
        test_2024_df = df[df['AsofDate'].dt.year == 2024] # Test/predict for 2024 data
        print("Filtered out data for the year 2024")

        sig_cols = []
        numerical_columns = [i for i in filtered_df.select_dtypes(include=np.number).columns if i != 'NewFundLaunch']

        for column in numerical_columns:
            launched = filtered_df.loc[filtered_df['NewFundLaunch'] == 1, column]
            no_lauched = filtered_df.loc[filtered_df['NewFundLaunch'] == 0, column]
            
            _, p_value = stats.ttest_ind(launched, no_lauched)
            if p_value < 0.05:
                print(f"{column}: Significant (p-value = {p_value})")
                sig_cols.append(column)
            else:
                print(f"{column}: Not significant (p-value = {p_value})")

        new_data = filtered_df[sig_cols + ['NewFundLaunch']]        

        X = new_data.drop('NewFundLaunch', axis=1)
        y = new_data['NewFundLaunch']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=12)
        
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        top_features = feature_importances.head(100).index  # Select top 100 features/ Can be Changed.
        X_selected = X[top_features]

        X_new = filtered_df[top_features]
        y_new = filtered_df['NewFundLaunch']
        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, train_size=0.8, random_state=12)

        lgb = LGBMClassifier(class_weight='balanced', random_state=42)
        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        xgb = XGBClassifier(scale_pos_weight=len(y_train) / sum(y_train), random_state=42)

        ensemble_model = VotingClassifier(estimators=[
            ('lgb', lgb),
            ('rf', rf),
            ('xgb', xgb)
        ], voting='soft')

        ensemble_model.fit(X_train, y_train)
        print("Model trained successfully")

        y_pred = ensemble_model.predict(X_test)
        y_prob = ensemble_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        print(f'Accuracy: {accuracy*100:.2f}%')
        print(f'Precision: {precision*100:.2f}%')
        print(f'Recall: {recall*100:.2f}%')
        print(f'F1-Score: {f1*100:.2f}%')
        print(f'ROC-AUC: {roc_auc*100:.2f}%')

        # Include the first 4 columns from the original df in the test results
        ref_columns = df.columns[:4].tolist()
        test_results = X_test.copy()
        test_results[ref_columns] = df.loc[X_test.index, ref_columns]
        test_results['Probability'] = y_prob
        test_results['Predicted'] = y_pred
        test_results['Actual'] = y_test

        # Use 2024 data as test data
        test_2024_results = test_2024_df.copy()
        X_test_2024 = test_2024_results[top_features]
        y_prob_2024 = ensemble_model.predict_proba(X_test_2024)[:, 1]
        y_pred_2024 = ensemble_model.predict(X_test_2024)
        test_2024_results['Probability'] = y_prob_2024
        test_2024_results['Predicted'] = y_pred_2024

        with pd.ExcelWriter('Final_Base_Likely_to_launch_Model_Results.xlsx', engine='openpyxl') as writer:
            test_results.to_excel(writer, sheet_name='Test_Results', index=False)
            pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Score': [accuracy, precision, recall, f1, roc_auc]
            }).to_excel(writer, sheet_name='Metrics', index=False)
            feature_importances.to_frame('Importance').reset_index().rename(columns={'index': 'Feature'}).to_excel(writer, sheet_name='Feature_Importance', index=False)
            test_2024_results.to_excel(writer, sheet_name='2024_Predictions', index=False)

        print("Results saved to 'Final_Base_Likely_to_launch_Model_Results.xlsx'")

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
