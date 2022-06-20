import pandas as pd
import numpy as np
import ipdb
import logging
logger = logging.getLogger(__name__)

def categorize_fn(X, y, groups,
               n_bins=10,
               bins=None,
               alpha=0.01,
               gamma=0.01
              ):
    """Map data to an existing set of categories."""
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"

    categories = None 

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
    categories = {}
    for group, i in df.groupby(groups).groups.items():
        # filter groups smaller than gamma*len(X)
        if len(i)/len(X) <= gamma:
            continue
        for interval, j in df.loc[i].groupby('interval').groups.items():
            if len(j) > min_size:
                categories[group + (interval,)] = j
                # ipdb.set_trace()
    return categories

class Auditor():
    """A class that determines and manages group membership over which to assess
    multicalibration.

    Parameters
    ----------
    estimator : Regessor or None, default: None 
        Optionally specify an ML method to determine which group to return.
    groups: list of str, default: None
        Specify a list of sensitive attributes to use as groups, instead of
        using an estimator. 
    metric: 'MC' or 'PMC', default: PMC
    alpha: float, default: 0.01
        tolerance for calibration error per group. 
    n_bins: int, default: 10
        used to discretize probabilities. 
    gamma: float, default: 0.1
        the minimum probability of a group occuring in the data. 
    random_state: int, default: 0
        random seed.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, 
                 estimator=None,
                 groups=None,
                 alpha=0.01,
                 n_bins=10,
                 bins=None,
                 gamma=0.1,
                 rho=0.1,
                 metric=None,
                 random_state=0,
                 verbosity=0
                ):
        self.estimator=estimator
        self.groups = groups
        self.alpha=alpha
        self.n_bins=n_bins
        self.bins=bins
        self.gamma=gamma
        self.rho=rho
        self.metric=metric
        self.random_state=random_state
        self.verbosity=verbosity

    def categorize(self, X, y):
        """Map data to an existing set of categories."""

        # return categories
        return categorize_fn(X, y, self.groups, 
                          bins=self.bins_,
                          alpha=self.alpha,
                          gamma=self.gamma
                         )


    def make_categories(self, X, y):
        """Define categories on data. 

        group:
            a subset of individuals in the population.
        interval:
            a risk prediction interval in (0, 1]. 
        category: a category is a pair of (subgroup, risk interval). 
            we represent these as a pair lambda functions that return True
            if a given sample is in the cateogry. 
        """
        assert isinstance(X, pd.DataFrame), "X should be a dataframe"
        # group or estimator must be specified 
        assert len(self.groups)>0 or self.estimator != None, \
                "Auditor must either have an estimator or groups specified."

        # self.categories_ = None 
        # df = X.copy()

        if self.bins is None:
            self.n_bins_ = self.n_bins
            self.bins_ = np.linspace(1/self.n_bins, 1.0, self.n_bins)
            self.bins_ = np.insert(self.bins_, 0, 0.0)
        else:
            self.bins_ = self.bins
            if self.bins_[0] > 0.0:
                self.bins_[0] = 0.0
                self.bins_ = np.insert(self.bins_, 0, 0.0)
            if self.bins_[-1] < 1.0:
                self.bins_ = np.concatenate((self.bins_, 1.0))
            self.n_bins_ = len(self.bins_)-1


        logger.info(f'self.bins_: {self.bins_}')
        min_size = self.gamma*self.alpha*len(X)/self.n_bins_
        logger.info(f'group size limit: {round(self.gamma*len(X))}')
        logger.info(f'category size limit: {round(min_size)}')


        # whether this auditor is a grouper or estimator
        self.is_grouper_ =  len(self.groups) > 0
        
        return self.categorize(X,y)

    def loss(self, y_true, y_pred, X, return_cat=False,
             metric=None):
        """calculate current loss in terms of multicalibration or PMC"""
        metric = self.metric if metric == None else metric
        alpha = 0.0
        worst = None 
        categories = self.categorize(X, y_pred)

        for c, idx in categories.items():
            category_loss = np.abs(y_true.loc[idx].mean() 
                                   - y_pred.loc[idx].mean()
                                  )
            if metric=='PMC': 
                category_loss /= max(y_true.loc[idx].mean(), self.rho)

            if  category_loss > alpha:
                alpha = category_loss
                worst = (c, idx)
                worstc = c
                worstidx = idx

        if return_cat:
            return alpha, worstc, worstidx, categories
        else:
            return alpha, worstc, worstidx

