"""
comorbidity model: uses Logistic Regression to create a prediction model
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


df = pd.read_csv("data/conditions.csv")
feature_col_names = ['age', 'sex', 'smoking', 'healthcare_worker',
                     'hypertension', 'diabetes',
                     'dementia', 'cancer', 'copd', 'asthma', 'chd', 'ccd', 'cnd', 'cld',
                     'ckd']
predicted_class_names = ['death']

X = df[feature_col_names].values
y = df[predicted_class_names].values
SPLIT_TEST_SIZE = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_TEST_SIZE, random_state=42)

lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42, Cs=3, cv=10, refit=False, class_weight="balanced", max_iter=10000)  # set number of jobs to -1 which uses all cores to parallelize
lr_cv_model.fit(X_train, y_train.ravel())

joblib.dump(lr_cv_model, "data/comorbid-trained-model.pkl")


df_predict = pd.read_csv("data/comorbid-predict.csv")

print(df_predict.shape)

X_predict = df_predict
del X_predict['death']

print(lr_cv_model.predict(X_predict))
