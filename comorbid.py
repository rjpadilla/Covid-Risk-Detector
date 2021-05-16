"""
comorbidity model: uses Logistic Regression to create a prediction model
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


df = pd.read_csv("data/conditions.csv")

# Tidying data
df = df.drop(columns=['Organ_transplant', 'Healthcare_worker', 'Pregnancy', 'Cachexia', 'Autoimm_disorder'])
df.columns = ['age', 'sex', 'smoking', 'alcohol', 'hypertension',
              'diabetes', 'rheuma', 'dementia', 'cancer', 'copd',
              'asthma', 'chd', 'ccd', 'cnd', 'cld',
              'ckd', 'aids', 'death']

feature_col_names = ['age', 'sex', 'smoking', 'alcohol', 'hypertension',
                     'diabetes', 'rheuma', 'dementia', 'cancer', 'copd',
                     'asthma', 'chd', 'ccd', 'cnd', 'cld',
                     'ckd', 'aids']
predicted_class_names = ['death']

X = df[feature_col_names].values
y = df[predicted_class_names].values
SPLIT_TEST_SIZE = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_TEST_SIZE, random_state=42)

lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42, Cs=3, cv=10, refit=False, class_weight="balanced", max_iter=10000)  # set number of jobs to -1 which uses all cores to parallelize
lr_cv_model.fit(X_train, y_train.ravel())

joblib.dump(lr_cv_model, "data/comorbid-trained-model.pkl")
