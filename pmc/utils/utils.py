import numpy as np
import pandas as pd
import ipdb

def squash_array(x):
    x[x<0.0] == 0.0
    x[x>1.0] == 1.0
    return x

def squash_series(x):
    return x.apply(lambda x: max(x,0.0)).apply(lambda x: min(x,1.0))

def category_diff(cat1, cat2):
    different=False
    for k1,v1 in cat1.items():
        if k1 not in cat2.keys():
            print(f'{k1} not in cat2')
            different=True
        else:
            if not v1.equals(cat2[k1]):
                print(f'indices for {k1} different in cat2')
                different=True
    for k2,v2 in cat2.items():
        if k2 not in cat1.keys():
            print(f'{k1} not in cat1')
            different=True
        else:
            if not v2.equals(cat1[k2]):
                print(f'indices for {k2} different in cat1')
                different=True
    if not different:
        # print('categories match.')
        return True
    else:
        return False

def categorize(X, y, groups, grouping,
               n_bins=10,
               bins=None,
               bin_scaling='linear',
               alpha=0.01,
               gamma=0.01
              ):
    """Map data to an existing set of categories."""
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"

    categories = None 

    # match sklearn.calibration_curve:
    # bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    # https://github.com/scikit-learn/scikit-learn/blob/baf828ca1/sklearn/calibration.py#L869
    if bins is None:
        bins = make_bins(n_bins, bin_scaling)
    else:
        n_bins=len(bins)


    df = X[groups].copy()
    df.loc[:,'interval'], retbins = pd.cut(y, bins, 
                                           include_lowest=True,
                                           retbins=True
                                          )
    categories = {}
    if grouping=='intersectional':
        group_ids = df.groupby(groups).groups
    elif grouping=='marginal':
        group_ids = df[groups].groupby(groups).groups
        group_ids = {}
        for g in groups:
            grp = df.groupby(g).groups
            for k,v in grp.items():
                group_ids[(g,k)] = v

    min_grp_size = gamma*len(X) 
    min_cat_size = min_grp_size*alpha/n_bins
    for group, i in group_ids.items():
        # filter groups smaller than gamma*len(X)
        if len(i) <= min_grp_size:
            continue
        for interval, j in df.loc[i].groupby('interval').groups.items():
            # filter categories smaller than alpha*gamma*len(X)/n_bins
            if len(j) > min_cat_size:
                categories[group + (interval,)] = j
    return categories

def make_bins(n_bins: int, bin_scaling='linear'):
    """Makes bins."""
    if bin_scaling == 'log':
        return np.geomspace(0.0, 1.0+1e-8, n_bins+1)
    else:
        return np.linspace(0.0, 1.0+1e-8, n_bins+1)
