import ipdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import itertools as it
logger = logging.getLogger(__name__)
from pmc.auditor import categorize_fn 

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def stratify_groups(X, y, groups,
               n_bins=10,
               bins=None,
               alpha=0.0,
               gamma=0.0
              ):
    """Map data to an existing set of groups, stratified by risk interval."""
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"


    if bins is None:
        bins = np.linspace(float(1.0/n_bins), 1.0, n_bins)
        bins[0] = 0.0
    else:
        n_bins=len(bins)


    df = X[groups].copy()
    df.loc[:,'interval'], retbins = pd.cut(y, bins, 
                                           include_lowest=True,
                                           retbins=True
                                          )
    stratified_categories = {}
    min_size = gamma*alpha*len(X)/n_bins
    for group, dfg in df.groupby(groups):
        # ipdb.set_trace()
        # filter groups smaller than gamma*len(X)
        if len(dfg)/len(X) <= gamma:
            continue
        
        for interval, j in dfg.groupby('interval').groups.items():
            if len(j) > min_size:
                if interval not in stratified_categories.keys():
                    stratified_categories[interval] = {}

                stratified_categories[interval][group] = j
                # ipdb.set_trace()
    # now we have categories where, for each interval, there is a dict of groups.
    return stratified_categories

def multicalibration_loss(
    estimator,
    X,
    y_true,
    groups,
    grouping='intersectional',
    n_bins=None,
    bins=None,
    categories=None,
    proportional=False,
    alpha=0.0,
    gamma=0.0,
    rho=0.0
):
    """custom scoring function for multicalibration.
       calculate current loss in terms of (proportional) multicalibration"""
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    y_pred = estimator.predict_proba(X)[:,1]
    y_pred = pd.Series(y_pred, index=y_true.index)


    assert isinstance(y_true, pd.Series)
    assert isinstance(y_pred, pd.Series)
    loss = 0.0

    assert groups is not None, "groups must be defined."

    if categories is None:
        # categories = estimator.auditor_.categorize(X, y_pred, groups, grouping,
        categories = categorize_fn(X, y_pred, groups, grouping,
                                n_bins=n_bins,
                                bins=bins,
                                alpha=alpha, 
                                gamma=gamma
                               )

    for c, idx in categories.items():
        category_loss = np.abs(y_true.loc[idx].mean() 
                               - y_pred.loc[idx].mean()
                              )
        if proportional: 
            category_loss /= max(y_true.loc[idx].mean(), rho)

        if  category_loss > loss:
            loss = category_loss

    return loss

def multicalibration_score(estimator, X, y_true, groups, **kwargs):
    return -multicalibration_loss(estimator, X, y_true, groups, **kwargs)

def proportional_multicalibration_loss(estimator, X, y_true, groups, **kwargs):
    kwargs['proportional'] = True
    return multicalibration_loss(estimator, X, y_true, groups,  **kwargs)
def proportional_multicalibration_score(estimator, X, y_true, **kwargs):
    return -proportional_multicalibration_loss(estimator, X, y_true, groups,  **kwargs)

def differential_calibration(
    estimator, 
    X, 
    y_true,
    groups,
    n_bins=None,
    bins=None,
    stratified_categories=None,
    alpha=0.0,
    gamma=0.0,
    rho=0.0
):
    """Return the differential calibration of estimator on groups."""

    assert isinstance(X, pd.DataFrame), "X needs to be a dataframe"
    assert all([g in X.columns for g in groups]), ("groups not found in"
                                                   " X.columns")
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    y_pred = estimator.predict_proba(X)[:,1]

    if stratified_categories is None:
        stratified_categories = stratify_groups(X, y_pred, groups,
                                n_bins=n_bins,
                                bins=bins,
                                alpha=alpha, 
                                gamma=gamma
                               )
    logger.info(f'# categories: {len(stratified_categories)}')
    dc_max = 0
    logger.info("calclating pairwise differential calibration...")
    for interval in stratified_categories.keys():
        for (ci,i),(cj,j) in pairwise(stratified_categories[interval].items()):

            yi = max(y_true.loc[i].mean(), rho)
            yj = max(y_true.loc[j].mean(), rho)

            dc = np.abs( np.log(yi) - np.log(yj) )

            if dc > dc_max:
                dc_max = dc

    return dc_max
