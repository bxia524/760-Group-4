#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# In[76]:


df = pd.read_csv("/Users/becki/Desktop/diabetes_012_health_indicators_BRFSS2015.csv")


# In[77]:


df.shape


# In[78]:


df.head()


# In[79]:


df.info()


# In[80]:


df["Diabetes_012"].value_counts()


# In[81]:


df["Diabetes_012"].value_counts(normalize=True)


# The dataset shows class imbalance across the three categories.

# In[82]:


df.isnull().sum()


# In[83]:


df.duplicated().sum()


# In[84]:


df = df.drop_duplicates()


# In[85]:


X = df.drop(columns=["Diabetes_012"])
y = df["Diabetes_012"]


# In[86]:


sensitive_features = ["Sex", "Age", "Income"]


# In[87]:


df["Sex"].value_counts()
df["Age"].value_counts().sort_index()
df["Income"].value_counts().sort_index()


# In[88]:


pd.crosstab(df["Sex"], df["Diabetes_012"], normalize="index")


# In[89]:


pd.crosstab(df["Age"], df["Diabetes_012"], normalize="index")


# In[90]:


df.isnull().sum()


# In[91]:


df.duplicated().sum()


# In[92]:


X = df.drop(columns=["Diabetes_012"])
y = df["Diabetes_012"]


# In[93]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[94]:


sensitive_features = ["Sex", "Age", "Income"]

sensitive_test = X_test[sensitive_features].copy()


# In[95]:


y.value_counts(normalize=True)


# In[96]:


y_train.value_counts(normalize=True)


# In[97]:


y_test.value_counts(normalize=True)


# In[98]:


import numpy as np
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))
class_weights


# Without handling class imbalance, the model tends to bias toward the majority class, leading to poor performance on minority classes.

# In[ ]:





# In[99]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# #  Logistics Regression

# In[100]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


log_reg = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    random_state=42,
    class_weight=class_weights
)


# In[101]:


log_reg.fit(X_train_scaled, y_train)

y_pred_log = log_reg.predict(X_test_scaled)


# In[102]:


print("Accuracy:", accuracy_score(y_test, y_pred_log))


# In[103]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))


# In[104]:


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))


# ## Fairness 

# In[105]:


from sklearn.metrics import accuracy_score, classification_report

def evaluate_group(mask, y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
        "f1": classification_report(y_true[mask], y_pred[mask], output_dict=True)["macro avg"]["f1-score"]
    }


# In[106]:


male = sensitive_test["Sex"] == 1
female = sensitive_test["Sex"] == 0

print("Male:", evaluate_group(male, y_test, y_pred_log))
print("Female:", evaluate_group(female, y_test, y_pred_log))


# In[107]:


young = sensitive_test["Age"] <= 6
old = sensitive_test["Age"] > 6

print("Young:", evaluate_group(young, y_test, y_pred_log))
print("Old:", evaluate_group(old, y_test, y_pred_log))


# In[108]:


low = sensitive_test["Income"] <= 4
high = sensitive_test["Income"] > 4

print("Low income:", evaluate_group(low, y_test, y_pred_log))
print("High income:", evaluate_group(high, y_test, y_pred_log))


# Fairness Evaluation of Logistic Regression
# 
# The fairness evaluation shows that the Logistic Regression model performs similarly across sex groups, with only a small difference in both accuracy and macro F1-score between males and females. This suggests that there is no strong evidence of sex-based disparity in model performance.
# 
# However, larger performance gaps are observed across age and income groups. In particular, the model achieves substantially higher accuracy for younger individuals than for older individuals, and also performs better for high-income groups than for low-income groups. These differences indicate potential fairness concerns, suggesting that the model may not generalize equally well across all demographic and socioeconomic subgroups.

# # Random Forest

# In[109]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[110]:


rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)


# In[111]:


rf_model.fit(X_train, y_train)


# In[112]:


y_pred_rf = rf_model.predict(X_test)


# In[113]:


print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# In[114]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


# In[115]:


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


# ## Fairness

# In[116]:


def evaluate_group(mask, y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
        "f1": classification_report(y_true[mask], y_pred[mask], output_dict=True)["macro avg"]["f1-score"]
    }


# In[117]:


male = sensitive_test["Sex"] == 1
female = sensitive_test["Sex"] == 0

print("Male:", evaluate_group(male, y_test, y_pred_rf))
print("Female:", evaluate_group(female, y_test, y_pred_rf))


# In[118]:


young = sensitive_test["Age"] <= 6
old = sensitive_test["Age"] > 6

print("Young:", evaluate_group(young, y_test, y_pred_rf))
print("Old:", evaluate_group(old, y_test, y_pred_rf))


# In[119]:


low = sensitive_test["Income"] <= 4
high = sensitive_test["Income"] > 4

print("Low income:", evaluate_group(low, y_test, y_pred_rf))
print("High income:", evaluate_group(high, y_test, y_pred_rf))


# Compared to Logistic Regression, Random Forest shows similar overall performance and slightly improved results for minority classes.
# 
# However, fairness analysis indicates that both models exhibit similar disparities across age and income groups.
# 
# This suggests that increasing model complexity does not necessarily improve fairness.

# # XGBoost

# In[120]:


import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_group(mask, y_true, y_pred):
    report = classification_report(
        y_true[mask],
        y_pred[mask],
        output_dict=True,
        zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
        "f1": report["macro avg"]["f1-score"]
    }


# In[121]:


xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42
)


# In[122]:


xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)


# In[123]:


print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))


# In[124]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))


# In[125]:


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))


# ## Fairness

# In[126]:


male = sensitive_test["Sex"] == 1
female = sensitive_test["Sex"] == 0

young = sensitive_test["Age"] <= 6
old = sensitive_test["Age"] > 6

low = sensitive_test["Income"] <= 4
high = sensitive_test["Income"] > 4


# In[127]:


print("\nFairness - Sex")
print("Male:", evaluate_group(male, y_test, y_pred_xgb))
print("Female:", evaluate_group(female, y_test, y_pred_xgb))


# In[128]:


print("\nFairness - Age")
print("Young:", evaluate_group(young, y_test, y_pred_xgb))
print("Old:", evaluate_group(old, y_test, y_pred_xgb))


# In[129]:


print("\nFairness - Income")
print("Low income:", evaluate_group(low, y_test, y_pred_xgb))
print("High income:", evaluate_group(high, y_test, y_pred_xgb))


# # compare

# In[130]:


import pandas as pd

comparison_table = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.629, 0.65, 0.838],
    "Macro F1": [0.43, 0.42, 0.40],
    "Sex Fairness": ["Small gap", "Small gap", "Small gap"],
    "Age Fairness": ["Large gap", "Large gap", "Smaller F1 gap"],
    "Income Fairness": ["Large gap", "Large gap", "Mixed result"]
})

comparison_table


# XGBoost improves accuracy but reduces balanced performance across classes.

# Increasing model complexity does not eliminate fairness issues.

# Fairness conclusions vary depending on whether accuracy or macro F1 is used.

# XGBoost is the most accurate model, but it is not the most balanced or fair model.

# # Shap

# In[134]:


X_sample = X_test.sample(1000, random_state=42)


# In[135]:


import shap

explainer = shap.TreeExplainer(xgb_model)


# In[136]:


shap_values = explainer.shap_values(X_sample)


# In[138]:


import numpy as np

print(type(shap_values))

if hasattr(shap_values, "shape"):
    print("shap_values.shape =", shap_values.shape)

print("X_sample.shape =", X_sample.shape)


# In[140]:


shap_class2 = shap_values[:, :, 2]


# In[141]:


shap.summary_plot(shap_class2, X_sample)


# In[142]:


shap.summary_plot(shap_class2, X_sample, plot_type="bar")


# Lifestyle-related variables have relatively low importance in the model, suggesting that the model prioritizes physiological indicators over behavioral factors.

# The model mainly relies on strong medical risk factors (BMI, age, health condition), which improves accuracy but limits its ability to detect subtle cases such as pre-diabetes.

# In[ ]:




