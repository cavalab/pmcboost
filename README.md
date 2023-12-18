# PMCBoost

PMCBoost is a sklearn-compatible package for **P**roportional **M**ulti**C**alibration Boosting. 
PMCBoost is a post-processing method that improves the calibration of a given classifier among subpopulations in the data.
PMCBoost supports both Multicalibration (Hébert-Johnson et al 2018) and Proportional Mulicalibration Boosting. 

"Proportional" multicalibration extends multicalibration by controlling both the *absolute* calibration error in each (group,bin) category, as well as the *proportion* of calibration error relative to the true (group,bin) risk. 
This makes it control both multicalibration and *differential calibration* (Foulds et al 2019) simultaneously. 

# Cite 

The following paper describes PMC in more detail: 


La Cava, W., Lett, E., and Wan, G. (2023).
Fair admission risk prediction with proportional multicalibration.
*Conference on Health, Inference, and Learning.*
(Best Paper Award!)
[PMLR](https://proceedings.mlr.press/v209/la-cava23a.html)  |  [arXiv](https://doi.org/10.48550/arXiv.2209.14613)  |  [experiments](https://github.com/cavalab/proportional-multicalibration) 
 
# Installation 
```
pip install git+https://github.com/cavalab/pmcboost
```

# Usage
```python
from pmc import MultiCalibrator, Auditor

# start with a baseline estimator
from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression()

# setup your data
X = pd.DataFrame()
y = pd.Series()
# groups correspond to columns in X that should be audited for fairness. 
groups = ['race','gender','income']
# create a Mutlicabrator
est = MultiCalibrator(
    estimator = estimator,
    auditor_type = Auditor(groups=groups)
    )
# train
est.fit(X,y)
```

Type `help(pmc.MultiCalibrator)` in Python to see additional parameters, or look [here](https://github.com/cavalab/pmcboost/blob/main/pmc/multicalibrator.py#L28). 

# Related Work

- La Cava, W., Lett, E., & Wan, G. (2022). Proportional Multicalibration. https://doi.org/10.48550/arXiv.2209.14613

- Hébert-Johnson, Ú., Kim, M. P., Reingold, O., & Rothblum, G. N. (2018). Calibration for the (Computationally-Identifiable) Masses. ArXiv:1711.08513 [Cs, Stat]. http://arxiv.org/abs/1711.08513

- Foulds, J., Islam, R., Keya, K. N., & Pan, S. (2019). An Intersectional Definition of Fairness. ArXiv:1807.08362 [Cs, Stat]. http://arxiv.org/abs/1807.08362

- For post-processing just for multicalibration or multiaccuracy, see the MCBoost package (in R) 
    - https://github.com/mlr-org/mcboost

# Contact

- William La Cava
    - @lacava
    - william lacava at gmail dot com
    - cavalab.org
