import os
import sys
import pickle
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

from transform import preprocess


params = yaml.safe_load(open('params.yaml'))['train']
seed = params['seed']
split = params['split']
model_type = params['model']

preprocessor = preprocess()

if model_type == 'random forest':
    ml_model = RandomForestClassifier()
else:
    ml_model = LogisticRegression()

input = sys.argv[1]  #"data/prepared"
output = sys.argv[2] #model.pkl #"model/saved_model.sav" 

with open(os.path.join(input, 'data.pkl'), 'rb') as fd:
    df = pickle.load(fd)

X = df[df.columns[:-1]]
y = df['Spectral Class']

# convert classes to integers
# le = LabelEncoder()
# le.fit(y)
# y = le.transform(y)

lb = LabelBinarizer()
y = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

clf = Pipeline(
        steps=[
            ('preparation', preprocessor),
            ('classifier', OneVsRestClassifier(ml_model)) 
            ]
        )
     
if model_type == 'random forest':
            param_grid = {
                'classifier__estimator__n_estimators': [50, 100],
                'classifier__estimator__max_features' :['sqrt', 'log2'],
                'classifier__estimator__max_depth' : [4,6,8],
                }
else:    
    param_grid = {
        'classifier__estimator__C': [0.1, 1.0, 10],
        }

grid_search = GridSearchCV(clf, param_grid, cv=10)

grid_search.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)

with open(os.path.join('model', output), 'wb') as fd:
    pickle.dump(grid_search, fd)

# python src/train.py data/prepared model.pkl