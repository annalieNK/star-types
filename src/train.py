import os
import sys
import pickle
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from transform import preprocess
from parameters import parameters


params = yaml.safe_load(open('params.yaml'))['train']
seed = params['seed']
split = params['split']
model_type = params['model']

preprocessor = preprocess()

if model_type == 'random forest':
    ml_model = RandomForestClassifier()
elif model_type == 'support vector machine':
    ml_model = SVC()
elif model_type == 'logistic regression':
    ml_model = LogisticRegression()
elif model_type == 'kneighbors':
    ml_model = KNeighborsClassifier()

input = sys.argv[1]  #data/prepared
output = os.path.join('model', sys.argv[2]) #model.pkl


def split_dataset(df):
    X = df[df.columns[:-1]]
    y = df['Star type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    return X_train, X_test, y_train, y_test


def fit(ml_model, model_type):

    clf = Pipeline(
            steps=[
                ('preparation', preprocessor),
                ('classifier', OneVsRestClassifier(ml_model)) 
            ]
        )

    # print(clf.get_params().keys())  
    param_grid = parameters(model_type)
    grid_search = GridSearchCV(clf, param_grid, cv=10)
    model = grid_search.fit(X_train, y_train)

    return model


os.makedirs('model', exist_ok=True)

with open(os.path.join(input, 'data.pkl'), 'rb') as fd:
    df = pickle.load(fd)

with open(output, 'wb') as fd:
    X_train, X_test, y_train, y_test = split_dataset(df)
    pickle.dump(fit(ml_model, model_type), fd)


# python src/train.py data/prepared model.pkl