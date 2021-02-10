import sys
import os
import pickle
import json
import yaml
import csv

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import plot_confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

import plot_roc_auc


params = yaml.safe_load(open('params.yaml'))['evaluate']
seed = params['seed']
split = params['split']
model_type = params['model']

model_file = os.path.join('model', sys.argv[1]) #model.pkl
input = os.path.join(sys.argv[2], 'data.pkl') #data/prepared
scores_file = os.path.join('metrics', sys.argv[3]) #scores.json
plots_file = os.path.join('plots', sys.argv[4]) #predicted.csv
roc_auc_plot_file = os.path.join('plots', sys.argv[5]) #ROC_AUC_curve.png

with open(model_file, 'rb') as fd:
    model = pickle.load(fd)

with open(input, 'rb') as fd:
    df = pickle.load(fd)

X = df[df.columns[:-1]]
y = df['Spectral Class']

# convert classes to binaries
lb = LabelBinarizer()
y = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)


# ROC AUC score
def return_roc_auc(X_test, y_test):
    predictions = model.predict_proba(X_test)
    # # Binarize the output
    # y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
    roc_auc = roc_auc_score(y_test, predictions, multi_class='ovr')
    return roc_auc

os.makedirs('metrics', exist_ok=True)

with open(scores_file, 'w') as fd:
    json.dump({
        'model type': model_type,
        'ROC AUC': return_roc_auc(X_test, y_test)}, fd)


## Plot confusion matrix

os.makedirs('plots', exist_ok=True)

# confusion_matrix = plot_confusion_matrix(
#                             model, 
#                             X_test, y_test,
#                             cmap=plt.cm.Blues
#                             )
# plt.savefig("figures/confusion_matrix.png")

predictions = model.predict(X_test)

with open(plots_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["actual", "predicted"])
    writer.writerows(zip(y_test, predictions)) 

## ROC AUC Curve
with open(roc_auc_plot_file, 'w') as fd:
    plot_roc_auc.roc_auc_multiclass(model, X_test, y_test)


# python src/evaluate.py model.pkl data/prepared scores.json predicted.csv ROC_AUC_curve.png
