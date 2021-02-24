import sys
import os
import pickle
import json
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import plots


params = yaml.safe_load(open('params.yaml'))['evaluate']
seed = params['seed']
split = params['split']
model_type = params['model']

model_file = os.path.join('model', sys.argv[1]) #model.pkl
input = os.path.join(sys.argv[2], 'data.pkl') #data/prepared
scores_file = os.path.join('reports', 'metrics', '{}_{}'.format(model_type, sys.argv[3])) #scores.json
confusion_matrix_plots_file = os.path.join('reports', 'plots', '{}_{}'.format(model_type, sys.argv[4])) #confusion_matrix.png
roc_auc_plots_file = os.path.join('reports', 'plots', '{}_{}'.format(model_type, sys.argv[5])) #ROC_AUC_curve.png


def split_dataset(df):
    X = df[df.columns[:-1]]
    y = df['Star type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    return X_train, X_test, y_train, y_test


# Confusion matrix
def plot_confusion_matrix(model, X_test, confusion_matrix_plots_file, y_test):
    predictions = model.predict(X_test)
    with open(confusion_matrix_plots_file, 'w') as fd:
        fig = plots.confusion_matrix_plot(y_test, predictions, [0,1,2,3,4,5])
        plt.savefig(confusion_matrix_plots_file)
        plt.close()


# ROC AUC Curve
def plot_roc_auc(y_test, roc_auc_plots_file, X_test, model_type, model):
    # convert classes to binaries
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    with open(roc_auc_plots_file, 'w') as fd:
        fig = plots.roc_auc_multiclass(X_test, y_test, model_type, model)
        plt.savefig(roc_auc_plots_file)
        plt.close()

    
## Metrics

# # ROC AUC metric
# def return_roc_auc(X_test, y_test):
#     predictions = model.predict_proba(X_test)
#     roc_auc = roc_auc_score(y_test, predictions, multi_class='ovr')
#     return roc_auc

os.makedirs('reports', exist_ok=True)
os.makedirs('reports/metrics', exist_ok=True)
os.makedirs('reports/plots', exist_ok=True)

with open(model_file, 'rb') as fd:
    model = pickle.load(fd)

with open(input, 'rb') as fd:
    df = pickle.load(fd)

X_train, X_test, y_train, y_test = split_dataset(df)

with open(scores_file, 'w') as fd:
    json.dump({
        'model type': model_type,
        'Accuracy Score': accuracy_score(y_test, model.predict(X_test)),
        'best parameters': model.best_params_
    }, 
    fd
    )

# Create metric plots
plot_confusion_matrix(model, X_test, confusion_matrix_plots_file, y_test)
plot_roc_auc(y_test, roc_auc_plots_file, X_test, model_type, model)



# python src/evaluate.py model.pkl data/prepared scores.json confusion_matrix.png ROC_AUC_curve.png
