import sys
import os
import pickle
import json
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import plots


params = yaml.safe_load(open('params.yaml'))['evaluate']
seed = params['seed']
split = params['split']
model_type = params['model']

model_file = os.path.join('model', sys.argv[1]) #model.pkl
input = os.path.join(sys.argv[2], 'data.pkl') #data/prepared
scores_file = os.path.join('metrics', '{}_{}'.format(model_type, sys.argv[3])) #scores.json
confusion_matrix_plots_file = os.path.join('plots', '{}_{}'.format(model_type, sys.argv[4])) #confusion_matrix.png
roc_auc_plots_file = os.path.join('plots', '{}_{}'.format(model_type, sys.argv[5])) #ROC_AUC_curve.png


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


# ROC AUC meric
def return_roc_auc(X_test, y_test):
    predictions = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, predictions, multi_class='ovr')
    return roc_auc

os.makedirs('metrics', exist_ok=True)

with open(scores_file, 'w') as fd:
    json.dump({
        'model type': model_type,
        'ROC AUC': return_roc_auc(X_test, y_test),
        'best parameters': model.best_params_
    }, 
    fd
    )

## Create metric plots

# ROC AUC Curve
with open(roc_auc_plots_file, 'w') as fd:
    fig = plots.roc_auc_multiclass(X_test, y_test, model_type, model)
    plt.savefig(roc_auc_plots_file)
    plt.close()

## Confusion matrix
predictions = model.predict(X_test)

with open(confusion_matrix_plots_file, 'w') as fd:
    fig = plots.confusion_matrix_plot(
                lb.inverse_transform(y_test), 
                lb.inverse_transform(predictions), 
                lb.classes_)
    plt.savefig(confusion_matrix_plots_file)
    plt.close()

# python src/evaluate.py model.pkl data/prepared scores.json confusion_matrix.png ROC_AUC_curve.png
