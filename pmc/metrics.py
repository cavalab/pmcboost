import ipdb
import numpy as np
import pandas as pd
from .params import groups as GROUPS
from tqdm import tqdm
import logging
import itertools as it
logger = logging.getLogger(__name__)
from .auditor import categorize_fn 

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def stratify_groups(X, y, groups,
               n_bins=10,
               bins=None,
               alpha=0.01,
               gamma=0.01
              ):
    """Map data to an existing set of groups, stratified by risk interval."""
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"


    if bins is None:
        bins = np.linspace(float(1.0/n_bins), 1.0, n_bins)
        bins[0] = 0.0
    else:
        n_bins=len(bins)

    min_size = gamma*alpha*len(X)/n_bins

    df = X[groups].copy()
    df.loc[:,'interval'], retbins = pd.cut(y, bins, 
                                           include_lowest=True,
                                           retbins=True
                                          )
    stratified_categories = {}
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
    groups=GROUPS,
    n_bins=None,
    bins=None,
    categories=None,
    proportional=False,
    alpha=0.01,
    gamma=0.1,
    rho=0.1
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
        categories = categorize_fn(X, y_pred, groups,
                                n_bins=n_bins,
                                bins=bins,
                                alpha=alpha, 
                                gamma=gamma
                               )

    for c, idx in categories.items():
        category_loss = np.abs(y_true.loc[idx].mean() 
                               - y_pred.loc[idx].mean()
                              )
        # print(c,len(idx),category_loss)
        if proportional: 
            category_loss /= max(y_true.loc[idx].mean(), rho)

        if  category_loss > loss:
            loss = category_loss

    return loss

def multicalibration_score(estimator, X, y_true, **kwargs):
    return -multicalibation_loss(estimator, X, y_true, **kwargs)

def proportional_multicalibration_loss(estimator, X, y_true, **kwargs):
    kwargs['proportional'] = True
    return multicalibration_loss(estimator, X, y_true, **kwargs)
def proportional_multicalibration_score(estimator, X, y_true, **kwargs):
    return -proportional_multicalibration_loss(estimator, X, y_true, **kwargs)

def differential_calibration(
    estimator, 
    X, 
    y_true,
    groups=GROUPS,
    n_bins=None,
    bins=None,
    stratified_categories=None,
    alpha=0.01,
    gamma=0.1,
    rho=0.01
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

# def smoothed_empirical_differential_fairness(self, concentration=1.0):
#     """Smoothed EDF 
#     References:
#         .. [#foulds18] J. R. Foulds, R. Islam, K. N. Keya, and S. Pan,
#            "An Intersectional Definition of Fairness," arXiv preprint
#            arXiv:1807.08362, 2018.
#     """
#     sbr = self._smoothed_base_rates(self.dataset.labels, concentration)

#     def pos_ratio(i, j):
#         return abs(np.log(sbr[i]) - np.log(sbr[j]))

#     def neg_ratio(i, j):
#         return abs(np.log(1 - sbr[i]) - np.log(1 - sbr[j]))

#     # overall DF of the mechanism
#     return max(max(pos_ratio(i, j), neg_ratio(i, j))
#                for i in range(len(sbr)) for j in range(len(sbr)) if i != j)
