import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from itertools import cycle

import plot_roc_auc

# from sklearn.multiclass import OneVsRestClassifier


df = pd.read_csv('data/data.csv')
df['Star color'] = df['Star color'].apply(lambda x: x.strip())

# preprocessor = preprocess()

X = df[df.columns[:-1]]
y = df['Spectral Class'].tolist()

le = LabelEncoder()
le.fit(y)
y = le.transform(y) 

# y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6])
# n_classes = y.shape[1]

lb = LabelBinarizer()
y = lb.fit_transform(y)
n_classes = y.shape[1]

# convert categorical column to dummy variable
numeric_features = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)',
                   'Absolute magnitude(Mv)', 'Star type']
numeric_transformer = StandardScaler()

categorical_features = ['Star color']
categorical_transformer = OneHotEncoder(handle_unknown='ignore') #sparse=False, drop="first"

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, selector(dtype_include="number")),#numeric_features),
        ('cat', categorical_transformer,  selector(dtype_include=object))],#categorical_features)])
    remainder='passthrough')

# preprocessor = make_column_transformer((OneHotEncoder(), ['Star color']),#selector(dtype_include=object)),
#                                         remainder='passthrough')

# Append classifier to preprocessing pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', OneVsRestClassifier(LogisticRegression()))])#LogisticRegression())]) RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

# clf.fit(X_train, y_train)

param_grid = {
    'classifier__estimator__C': [0.1, 1.0, 10],
}
# param_grid = {
#     'classifier__n_estimators': [50, 100],
#     'classifier__max_features' :['sqrt', 'log2'],
#     'classifier__max_depth' : [4,6,8],
#     }

grid_search = GridSearchCV(clf, param_grid, cv=10)

grid_search.fit(X_train, y_train)

# print("model score: %.3f" % clf.score(X_test, y_test))
# print(grid_search.best_params_)
# print(grid_search.score(X_test, y_test))   


# # # ROC AUC score
predictions = grid_search.predict_proba(X_test)

# Binarize the output
# y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
# n_classes = y_test_binarized.shape[1]
# print(roc_auc_score(y_test_binarized, predictions, multi_class='ovr'))

# lb = LabelBinarizer()
# y_test_binarized = lb.fit_transform(y_test)
# print(y_test_binarized.shape)
print(roc_auc_score(y_test, predictions, multi_class='ovr'))


# plot_roc_auc.roc_auc_multiclass(grid_search, X_test, n_classes, y_test)

# # print(metrics.classification_report(y_test, predictions, labels=[0,1,2,3,4,5,6]))
# import csv

# with open('predicted.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(["actual", "predicted"])
#     writer.writerows(zip(y_test, predictions)) 
