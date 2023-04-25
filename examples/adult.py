import copy
import pandas as pd
from pmc import MultiCalibrator, Auditor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from pmlb import pmlb   
from pmc.metrics import (
    proportional_multicalibration_loss,
    multicalibration_loss
)

dataset = pmlb.fetch_data('adult')
X = dataset.drop('target',axis=1)
y = dataset['target']
Xtrain,Xtest, ytrain,ytest = train_test_split(
    X,
    y,
                                                    stratify=y,
                                                    random_state=42,
                                                    test_size=0.2
                                                   )

groups = ['race','sex','workclass']

metric = 'PMC'

est = MultiCalibrator(
    estimator = LogisticRegression(),
    auditor_type = Auditor(groups=groups,grouping='intersectional'),
    eta = 0.1,
    gamma=0.1,
    alpha=0.01,
    rho=0.2,
    max_iters=1000,
    verbosity=0,
    n_bins=10,
    metric=metric,
    bin_scaling='log',
)

print('fitting')

est.fit(Xtrain,ytrain)

print(est.stats_)