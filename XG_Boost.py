#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install xgboost
# !pip install catboost
# !pip install category_encoders


# In[2]:


import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import category_encoders as ce
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


# In[3]:


# 1. Load the Dataset
file_path = 'league_of_legends.csv'
df = pd.read_csv(file_path)


# In[4]:


# 2. Split the Dataset (90-10)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)


# In[5]:


# 3. One-Hot Encoding
train_df_onehot = pd.get_dummies(train_df, columns=['League', 'Season', 'Type'])
test_df_onehot = pd.get_dummies(test_df, columns=['League', 'Season', 'Type'])
missing_cols = set(train_df_onehot.columns) - set(test_df_onehot.columns)
for c in missing_cols:
    test_df_onehot[c] = 0
test_df_onehot = test_df_onehot[train_df_onehot.columns]


# In[6]:


# 4. Target Encoding
target_cols = [
    'blueTop', 'blueJungle', 'blueMiddle', 'blueADC', 'blueSupport',
    'redTop', 'redJungle', 'redMiddle', 'redADC', 'redSupport',
    'blueTopChamp', 'blueJungleChamp', 'blueMiddleChamp', 'blueADCChamp', 'blueSupportChamp',
    'redTopChamp', 'redJungleChamp', 'redMiddleChamp', 'redADCChamp', 'redSupportChamp',
    'blueTeamTag', 'redTeamTag'
]
target_variable = 'bResult'
encoder = ce.TargetEncoder(cols=target_cols)
encoder.fit(train_df, train_df[target_variable])

train_df_target_encoded = encoder.transform(train_df)
test_df_target_encoded = encoder.transform(test_df)
train_df_onehot = train_df_onehot.drop(columns=target_cols, axis=1)
test_df_onehot = test_df_onehot.drop(columns=target_cols, axis=1)
train_df_encoded = pd.concat([train_df_onehot, train_df_target_encoded[target_cols]], axis=1)
test_df_encoded = pd.concat([test_df_onehot, test_df_target_encoded[target_cols]], axis=1)


# In[7]:


# 5. Feature Selection using Averaged Importance Scores
# 5.1 Separate features and target variable
X_train = train_df_encoded.drop([target_variable], axis=1)
y_train = train_df_encoded[target_variable]

# 5.2 Calculate Mutual Information scores
mi_scores = mutual_info_classif(X_train, y_train)
mi_scores = pd.Series(mi_scores, name='MI_Scores', index=X_train.columns)

# 5.3 CatBoost Importance
catboost_model = CatBoostClassifier(iterations=100, verbose=0)
catboost_model.fit(X_train, y_train)
catboost_importances = pd.Series(catboost_model.get_feature_importance(), name='CatBoost_Importance', index=X_train.columns)

# 5.4 Combine and Normalize
importance_df = pd.concat([mi_scores, catboost_importances], axis=1)
importance_df['MI_Scores'] = (importance_df['MI_Scores'] - importance_df['MI_Scores'].min()) / (importance_df['MI_Scores'].max() - importance_df['MI_Scores'].min())
importance_df['CatBoost_Importance'] = (importance_df['CatBoost_Importance'] - importance_df['CatBoost_Importance'].min()) / (importance_df['CatBoost_Importance'].max() - importance_df['CatBoost_Importance'].min())
importance_df['Combined_Importance'] = (importance_df['MI_Scores'] + importance_df['CatBoost_Importance']) / 2

# 5.5 Sort and Select Features
sorted_features = importance_df.sort_values(by='Combined_Importance', ascending=False).index
N = 22
selected_features = sorted_features[:N]
X_train_selected = X_train[selected_features]
X_test_selected = test_df_encoded.drop([target_variable], axis=1)[selected_features]

print("Selected features based on average combined importance:")
print(selected_features)

# 5.6 Visualize the Final Selected Features
selected_importance = importance_df.loc[selected_features, 'Combined_Importance']
selected_importance = selected_importance.sort_values(ascending=True)

plt.figure(figsize=(10, 12))
plt.barh(selected_importance.index, selected_importance.values, color='skyblue')
plt.xlabel('Average Combined Importance Score')
plt.title('Top Selected Features Based on Average Combined Importance')
plt.show()


# In[8]:


# 6. Train XGBoost Model with Best Parameters obtained from Bayesian Optimization
best_params = {
    'learning_rate': 0.9,    
    'max_depth': 0.1,
    'n_estimators': 400,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
}

xgb_clf = xgb.XGBClassifier(random_state=42, **best_params)
xgb_clf.fit(X_train_selected, y_train)


# In[9]:


# 7. Model Validation (Cross-Validation)
cv_scores = cross_val_score(xgb_clf, X_train_selected, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")


# In[10]:


# 8. Model Testing and Evaluation
X_test_selected = test_df_encoded.drop([target_variable], axis=1)[selected_features]
y_test = test_df_encoded[target_variable]
y_pred = xgb_clf.predict(X_test_selected)
y_prob = xgb_clf.predict_proba(X_test_selected)[:, 1]

# Evaluation Metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_prob)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Test ROC AUC: {test_roc_auc:.4f}")

# 9. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {test_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[11]:


# 10. Save the Model
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_clf, f)

with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

print("XGBoost model saved.")

