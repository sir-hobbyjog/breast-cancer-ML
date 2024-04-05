import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from classifier_test import classifier_test


training_data = pd.read_csv("Resources/training_data.csv")
y_train = training_data.pop("malignant")
X_train = training_data

testing_data = pd.read_csv("Resources/testing_data.csv")
y_test = testing_data.pop("malignant")
X_test = testing_data

rf_model = RandomForestClassifier(n_estimators=100,random_state=1)
classifier_test("Random Forest", rf_model, X_train, y_train, X_test, y_test)