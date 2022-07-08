import copy
import pandas as pd
from multicalibrator import MultiCalibrator
from auditor import Auditor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from pmlb import pmlb   
import ipdb
import utils
import metrics

dataset = pmlb.fetch_data('adult', 
                          local_cache_dir='/home/bill/projects/pmlb'
)
X = dataset.drop('target',axis=1)
y = dataset['target']
Xtrainval,Xtest, ytrainval,ytest = train_test_split(X,y,
                                                    stratify=y,
                                                    random_state=42,
                                                    test_size=0.2
                                                   )
Xtrain,Xval, ytrain,yval = train_test_split(Xtrainval,ytrainval,
                                            stratify=ytrainval, 
                                            random_state=42,
                                            test_size=0.5
                                           )

# groups = ['age','workclass','race','sex','native-country']
groups = ['race','sex','native-country']
# groups = ['race', 'sex']
# groups = list(X.columns)

est = LogisticRegression().fit(Xtrain,ytrain)

grouping = 'marginal'

print(f'y balance: {y.sum()/len(y)}')
MC_params = dict(
    estimator = est,
    auditor_type = Auditor(groups=groups,grouping=grouping),
    eta = 0.3,
    gamma=0.1,
    alpha=0.1,
    rho=0.2,
    max_iters=1000,
    verbosity=2,
    n_bins=7,
    # iter_sample='bootstrap'
)
PMC_params = copy.deepcopy(MC_params)

MC = MultiCalibrator(**MC_params, metric='MC')
PMC = MultiCalibrator(**PMC_params, metric='PMC')

MC.fit(Xval,yval)
PMC.fit(Xval,yval)

bins = MC.auditor_.bins_
assert all(bins == PMC.auditor_.bins_)

print('model\tfold\tAUROC\tAUPRC\tMC\tPMC')
for model,name in [(est,'base'), (MC,'MC'), (PMC,'PMC')]:
    for x,y_true,fold in [(Xtrain, ytrain,'train'), 
                          (Xval, yval,'val'),
                          (Xtest, ytest,'test')]:
    # for x,y_true,fold in [(Xtest, ytest,'test')]:
        y_pred = pd.Series(model.predict_proba(x)[:,1], index=x.index)
        print(name,end='\t')
        print(fold,end='\t')
        for metric in [roc_auc_score, average_precision_score]: 
            # print(f'{metric.__name__}: {metric(y_true, y_pred):.3f}')
            print(f'{metric(y_true, y_pred):.3f}',end='\t')
        mc = metrics.multicalibration_loss(
            model,
            x, 
            y_true, 
            groups,
            grouping=grouping,
            alpha=MC_params['alpha'], 
            gamma=MC_params['gamma'],
            bins=bins,
            rho=MC_params['rho']
        )
        pmc = metrics.proportional_multicalibration_loss(
            model,
            x, 
            y_true, 
            groups,
            grouping=grouping,
            proportional=True,
            alpha=MC_params['alpha'], 
            gamma=MC_params['gamma'],
            # n_bins=MC_params['n_bins'],
            bins=bins,
            rho=MC_params['rho']
        )
        # if name == 'PMC':
        #     assert pmc == model.auditor_.loss(y_true, y_pred, x)
        # elif name == 'MC':
        #     assert mc == model.auditor_.loss(y_true, y_pred, x)
        print(f'{mc:.3f}',end='\t')
        print(f'{pmc:.3f}')
        # print('-----------')

