import os
import subprocess
import sys

try:
    import openpyxl
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import scipy.stats as stats
import warnings
import traceback

warnings.filterwarnings("ignore")

def main():
    try:
        df = pd.read_csv(r"C:\Users\Mohammed Aftab\Desktop\Desktop\Predictive Modeling Convergence\Auditor\Likely to Launch_Auditor\Predictive_Likely_to_launch.csv")
        print("Data loaded successfully")

        df['AsofDate'] = pd.to_datetime(df['AsofDate'], format='%m/%d/%Y')

        df_launched = df[df['NewFundLaunch'] == 1]
        ### Feature Engineering- Create a new feature "MonthsBetweenLaunches"
        launch_counts = df_launched.groupby(['manager_id', 'AsofDate']).size().reset_index(name='LaunchCount')
        
        launch_counts = launch_counts.sort_values(by=['manager_id', 'AsofDate'])
        
        launch_counts['PreviousLaunchDate'] = launch_counts.groupby('manager_id')['AsofDate'].shift(1)
        
        launch_counts['MonthsBetweenLaunches'] = launch_counts.apply(
            lambda row: (row['AsofDate'] - row['PreviousLaunchDate']).days // 30 if pd.notnull(row['PreviousLaunchDate']) else None, axis=1
        )
        
        df = df.merge(launch_counts[['manager_id', 'AsofDate', 'MonthsBetweenLaunches']], on=['manager_id', 'AsofDate'], how='left')


        filtered_df = df[df['AsofDate'].dt.year != 2024]  #Training data till year 2023
        test_2024_df = df[df['AsofDate'].dt.year == 2024] # Test/predict for 2024 data
        print("Filtered out data for the year 2024")
        
        X = filtered_df.drop(['NewFundLaunch','NewManager','ExistingAuditors','AsofDate','manager_id'], axis=1)
        y = filtered_df['NewFundLaunch']

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=12)      
        xgb_params = {'scale_pos_weight': len(y_train) / sum(y_train), 'random_state': 42}
        xgb = XGBClassifier(**xgb_params)

        xgb.fit(X_train,y_train)
        y_pred = xgb.predict(X_test)
        y_prob = xgb.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

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
        X_test_2024 = test_2024_results[X_train.columns]  
        y_prob_2024 = xgb.predict_proba(X_test_2024)[:, 1]
        y_pred_2024 = xgb.predict(X_test_2024)
        test_2024_results['Probability'] = y_prob_2024
        test_2024_results['Predicted'] = y_pred_2024

        with pd.ExcelWriter('Likely_to_Launch_prediction_results.xlsx', engine='openpyxl') as writer:
            test_results.to_excel(writer, sheet_name='Test_Results', index=False)
            pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Score': [accuracy, precision, recall, f1, roc_auc]
            }).to_excel(writer, sheet_name='Metrics', index=False)
            test_2024_results.to_excel(writer, sheet_name='2024_Predictions', index=False)

        print("Results saved to 'Likely_to_Launch_prediction_results.xlsx'")

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
