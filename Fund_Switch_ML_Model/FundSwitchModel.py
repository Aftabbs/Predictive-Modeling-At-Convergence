from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import numpy as np
import warnings
warnings.filterwarnings("ignore")
print("All libraries are imported")

df=pd.read_csv(r"C:\Users\Mohammed Aftab\Downloads\final_prediction_fund_switch.csv")
filtered_df=df[pd.to_datetime(df.AsofDate).dt.year!=2024]
X=filtered_df.drop(['NewFund', 'PrevAuditor', 'CurrentAuditor', 'AsofDate', 'crd', 'FundID',
       'manager_id', 'Auditor Group','Switched'],axis=1)
y=filtered_df.Switched
## Select Significant Features
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=.8,random_state=12)
lgb = LGBMClassifier(class_weight='balanced', random_state=42)

lgb.fit(X_train, y_train)

feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': lgb.feature_importances_})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

top_features = feature_importances['feature'].head(70).tolist()  # Top 70, Can be Changed

X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
xgb = XGBClassifier(scale_pos_weight=len(y_train) / sum(y_train), random_state=42)

ensemble_model = VotingClassifier(estimators=[
    ('lgb', lgb),
    ('rf', rf),
    ('xgb', xgb)
], voting='soft')

ensemble_model.fit(X_train_selected, y_train)

y_pred_proba = ensemble_model.predict_proba(X_test_selected)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

y_pred_adjusted = (y_pred_proba >= .5859).astype(int) ## After Fine tuniing .58 is set as optimal threshhold to stabilise precision and recall

accuracy = accuracy_score(y_test, y_pred_adjusted)
precision = precision_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)
f1 = f1_score(y_test, y_pred_adjusted)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy:',accuracy*100)
print(f'Precision:',precision*100)
print(f'Recall:',recall*100)
print(f'F1-Score:',f1*100)
print(f'ROC-AUC:',roc_auc*100)

test_data = filtered_df.loc[X_test.index]

test_data['Predicted_Switch'] = y_pred_adjusted

result_columns = ['NewFund', 'PrevAuditor', 'CurrentAuditor', 'AsofDate', 'crd', 'FundID', 'manager_id', 'Auditor Group', 'Predicted_Switch']
results_df = test_data[result_columns]
results_df.to_csv('Pred_results.csv', index=False)
results_df.head(4)

