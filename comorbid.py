"""
comorbidity model: uses Logistic Regression to create a prediction model
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import plot_confusion_matrix


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

# Predict accuracy of our model
lr_cv_accuracy_test = lr_cv_model.predict(X_test)
print("Accuracy: {0:.4f}".format(metrics.recall_score(y_test, lr_cv_accuracy_test)))

# Saving charts
disp = plot_confusion_matrix(lr_cv_model, X_test, y_test)
disp.ax_.set_title("Confusion Matrix")
plt.plot(disp.confusion_matrix)
plt.savefig('static/images/confusion.png')

sns_bar = sns.displot(df['age'], kde=False, bins=10)
sns_bar.savefig('static/images/barplot.png')

sns_joint = sns.jointplot(x='death',y='age',data=df,kind="reg",size=8)
sns_joint.savefig('static/images/jointplot.png')

joblib.dump(lr_cv_model, "data/comorbid-trained-model.pkl")
