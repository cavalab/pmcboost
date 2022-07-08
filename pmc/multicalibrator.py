"""
Proportional Multicalibration Post-processor
copyright William La Cava
License: GNU GPL3
"""
import ipdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from copy import copy
# from utils import squash_array, squash_series
import utils #as utils
import logging
# logging.basicConfig(format='%(message)s',
#                     datefmt='%m/%d/%Y %I:%M:%S %p',
#                    )
logger = logging.getLogger(__name__)

class MultiCalibrator(ClassifierMixin, BaseEstimator):
    """ A classifier post-processor that updates a model to satisfy different
    notions of fairness.

    Parameters
    ----------
    estimator : Probabilistic Classifier 
        A pre-trained classifier that outputs probabilities. 
    auditor_type: Classifier or callable
        Method that returns a subset of sample from the data, belonging to a 
        specific group.
    metric: 'MC' or 'PMC', default: PMC
    alpha: float, default: 0.01
        tolerance for calibration error per group. 
    n_bins: int, default: 10
        used to discretize probabilities. 
    gamma: float, default: 0.1
        the minimum probability of a group occuring in the data. 
    rho: float, default: 0.1
        the minimum risk prediction to attempt to adjust. 
        relevant for proportional multicalibration.
    max_iters: int, default: None
        maximum iterations. Will terminate whether or not alpha is achieved.
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
                 auditor_type=None,
                 metric='PMC',
                 alpha=0.01,
                 n_bins=10,
                 gamma=0.01,
                 rho=0.1,
                 eta=1.0,
                 max_iters=100,
                 random_state=0,
                 verbosity=0,
                 iter_sample=None,
                 split=0.5
                ):
        self.estimator=estimator
        self.auditor_type=auditor_type
        self.metric=metric
        self.alpha=alpha
        self.n_bins=n_bins
        self.gamma=gamma
        self.rho=rho
        self.eta=eta
        self.max_iters=max_iters
        self.random_state=random_state
        self.verbosity=verbosity
        self.iter_sample=iter_sample
        self.split=split

    def __name__(self):
        if self.metric=='PMC':
            return 'Proportional Multicalibrator'
        return 'MultiCalibrator' 

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        logger.setLevel({0:logging.WARN, 
                         1:logging.INFO, 
                         2:logging.DEBUG}
                        [self.verbosity]
                       )

        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        assert len(self.classes_) == 2, "Only binary classification supported"
        # assert self.split > 0.0 and self.split <= 1.0
        if self.split == 0.0 or self.split == 1.0:
            train_X = X
            test_X = X
            train_y = y
            test_y = y
        else:
            train_X,test_X,train_y,test_y = \
                    train_test_split(X, 
                                     y,
                                     train_size=self.split,
                                     test_size=1-self.split,
                                     shuffle=False,
                                     random_state=self.random_state
                                    )


        self.est_ = self.estimator.fit(train_X, train_y)

        self.X_ = test_X
        self.y_ = test_y.astype(float)

        if not isinstance(self.X_, pd.DataFrame):
            self.X_ = pd.DataFrame(self.X_)
        if not isinstance(self.y_, pd.Series):
            self.y_ = pd.Series(self.y_)
        self.X_ = self.X_.set_index(self.y_.index)


        assert hasattr(self.est_, 'predict_proba'), ("Classifier has no"
                                                    "'predict_proba' method")

        self.auditor_ = copy(self.auditor_type)
        for att in vars(self):
            if hasattr(self.auditor_, att):
                setattr(self.auditor_, att, getattr(self,att))
        
        # map groups to adjustments
        self.adjustments_ = [] 
        iters, n_updates = 0, 0 
        updated = True
        # predictions
        y_init = self.est_.predict_proba(self.X_)[:,1]
        y_init = pd.Series(y_init, index=self.X_.index)
        y_adjusted = copy(y_init)
        MSE = mse(self.y_, y_init)
        log = dict(
            iteration=[],
            r=[],
            ybar=[],
            delta=[],
            alpha=[],
            category=[] 
        )

        ######################################## 
        # initialize categories and loss metric
        categories = self.auditor_.make_categories(self.X_, y_init)
        # bootstrap sample self.X_,y
        Xs, ys = self.X_, self.y_
        init_cal_loss, _, _ = self.auditor_.loss(
                                                 self.y_, 
                                                 y_adjusted,
                                                 self.X_
                                                )
        smallest_cat = len(Xs)
        ######################################## 
        # main boosting loop
        for i in range(self.max_iters):
            if self.iter_sample == 'bootstrap':
                Xs, ys, ys_pred = resample(self.X_, self.y_, y_adjusted,
                                           random_state=self.random_state
                                          )
            else:
                Xs, ys, ys_pred = self.X_, self.y_, y_adjusted

            MSE = mse(ys, ys_pred)
            cal_loss, p_worst_c, p_worst_idx, cats =  \
                    self.auditor_.loss(ys, ys_pred, Xs, return_cat=True)

            logger.info(
                        f'# categories:{len(categories)}, '
                        f'smallest cat: {smallest_cat},'
                        f'# updates:{n_updates}, '
                        f'{self.metric}:{cal_loss:.3f}, '
                        f'MSE:{MSE:.3f} '
            )
            # make an iterable over groups, intervals
            categories = self.auditor_.categorize(Xs, ys_pred)
            if self.iter_sample == None:
                assert utils.category_diff(categories, cats), \
                        "categories don't match"

                assert p_worst_c in categories.keys()

            Mworst_delta = 0
            pmc_adjust = 1
            smallest_cat = len(Xs)
            if self.verbosity > 0:
                iterator = tqdm(categories.items(), 
                                      desc='updating categories', 
                                      leave=False)
            else:
                iterator = categories.items()

            updated=False
            ########################################  
            # loop through categories (group, interval pairs)
            for category, idx in iterator:
                if len(idx) < smallest_cat:
                    smallest_cat = len(idx)

                # calc average predicted risk for the group
                rbar = ys_pred.loc[idx].mean()
                # calc actual average risk for the group
                ybar = ys.loc[idx].mean()

                # delta 
                delta = ybar - rbar
                
                # set alpha 
                alpha = self.alpha  

                # set the PMC adjustment if needed
                if self.metric=='PMC':
                    pmc_adjust = max(ybar,self.rho)
                    alpha *= pmc_adjust

                logger.debug(
                      f'category:{category}, '
                      f'rbar:{rbar:3f}, '
                      f'ybar:{ybar:3f}, '
                      f'delta:{delta:3f}, '
                      f'alpha:{alpha:3f}, '
                      f'delta/pmc_adjust:{np.abs(delta)/pmc_adjust:.3f}'
                     )

                if ((self.metric=='MC' and np.abs(delta) > Mworst_delta)
                    or (self.metric=='PMC' 
                        and np.abs(delta)/pmc_adjust > Mworst_delta)
                   ):
                    Mworst_delta=np.abs(delta) 
                    Mworst_c = category
                    Mworst_idx = idx
                    if self.metric=='PMC':
                        Mworst_delta /= pmc_adjust

                if np.abs(delta) > alpha:
                    update = self.eta*delta

                    logger.debug(f'Updating category:{category}')
                    # update estimates 
                    y_adjusted.loc[idx] += update

                    if updated == False:
                        # initialize adjustment list
                        self.adjustments_.append({})

                    # store adjustment
                    self.adjustments_[-1][category] = update

                    updated=True
                    n_updates += 1

                    # make sure update was good
                    assert not any(y_adjusted.isna())

                iters += 1
                if iters >= self.max_iters: 
                    logger.info('max iters reached')
                    break

            # constrain adjusted output between 0 and 1
            y_adjusted = utils.squash_series(y_adjusted)
            assert y_adjusted.max() <= 1.0 and y_adjusted.min() >= 0.0

            new_cal_loss, worst_c, worst_idx = self.auditor_.loss(
                                                     ys, 
                                                     ys_pred,
                                                     Xs
                                                    )
            logger.debug(f'worst category from multicalibrator: '
                         f'{Mworst_c}, alpha = {Mworst_delta}')
            logger.debug(f'worst category from auditor: '
                         f'{worst_c}, alpha = {new_cal_loss}')

            if iters >= self.max_iters: 
                logger.warn('max_iters was reached before alpha termination'
                            ' criterion was satisfied.')
                break

            if self.iter_sample=='bootstrap' and not updated:
                total_cal_loss, _, _ = self.auditor_.loss(
                                                     self.y_, 
                                                     y_adjusted,
                                                     self.X_
                                                    )
                if total_cal_loss < self.alpha:
                    break
            elif not updated:
                logger.info('no updates this round. breaking')
                break
            else:
                cal_diff = cal_loss - new_cal_loss
        ## end for loop
        ######################################## 
        logger.info(f'finished. updates: {n_updates}')
        y_end = pd.Series(self.predict_proba(self.X_)[:,1], index=self.X_.index)
        np.testing.assert_allclose(y_adjusted, y_end, rtol=1e-04)

        init_MC = self.auditor_.loss(self.y_, y_init, self.X_, metric='MC')[0]
        final_MC = self.auditor_.loss(self.y_, y_end, self.X_, metric='MC')[0]
        init_PMC = self.auditor_.loss(self.y_, y_init, self.X_, metric='PMC')[0]
        final_PMC = self.auditor_.loss(self.y_, y_end, self.X_, metric='PMC')[0]
        logger.info(f'initial multicalibration: {init_MC:.3f}')
        logger.info(f'final multicalibration: {final_MC:.3f}')
        logger.info(f'initial proportional multicalibration: {init_PMC:.3f}')
        logger.info(f'final proportional multicalibration: {final_PMC:.3f}')
        # Return the classifier
        return self
    

    def predict_proba(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check if fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        # X = check_array(X)

        # y_pred = self.est_.predict_proba(X)[:,1]

        y_pred = pd.Series(self.est_.predict_proba(X)[:,1],
                            index=X.index)
        

        for adjust_iter in self.adjustments_:
            if self.iter_sample == 'bootstrap':
                Xs, ys_pred = resample(X, y_pred,  
                                  random_state=self.random_state
                                 )
            else:
                Xs, ys_pred = X, y_pred

            categories = self.auditor_.categorize(Xs, ys_pred)
            for category, update in adjust_iter.items(): 
                if category in categories.keys():
                    idx = categories[category]
                    y_pred.loc[idx] += update
                    # y_pred.loc[idx] = utils.squash_series(y_pred.loc[idx])
                # else:
                #     logger.warn(f'y_pred missing category {category}')
            y_pred = utils.squash_series(y_pred)

        # y_pred = utils.squash_series(y_pred)
        # ipdb.set_trace()
        rety = np.vstack((1-y_pred, y_pred)).T
        return rety

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        # X = check_array(X)

        return self.predict_proba(X)[:,1] > 0.5

    def score(self, X, y):
        """Return auditor score"""
        y_pred = self.predict_proba(X)[:,1]
        return self.auditor_.loss(y, y_pred, X)[0]
