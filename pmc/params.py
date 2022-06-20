from ml.pmc.auditor import Auditor
import scipy.stats #import randint
import numpy as np

groups = ['ethnicity','gender'] #,'anchor_year_group']

params = dict(
    auditor_type = Auditor(groups=groups),
    eta=0.25,
    gamma=0.05,
    alpha=0.05,
    rho=0.1,
    max_iters=10**5,
    verbosity=1,
    n_bins=5,
    split=0,
    # iter_sample='bootstrap'
)
# Etas = [0.1, 0.5, 1.0]
Etas = np.linspace(0.01,1,10)
eta_dist = scipy.stats.uniform(0.01,1)

Gammas = [0.01, 0.05, 0.1]
gamma_dist = scipy.stats.uniform(0.05,.2)

Alphas = [0.01, 0.05, 0.1]
alpha_dist = scipy.stats.uniform(0.01,.15)

N_binses = [2, 5, 10]
N_binses = scipy.stats.randint(2,10)

Rhos = [0.01, 0.05, 0.1, 0.2]
rho_dist = scipy.stats.uniform(0.05,.15)

Iter_samples = [None, 'bootstrap']

Splits = [.25, 0.5, 0.75]
split_dist = scipy.stats.uniform(0.25,.75)

mc_hyper_params = dict(
    eta=Etas,
    # gamma=Gammas,
    # alpha=Alphas,
    # n_bins=N_binses,
    # iter_sample=Iter_samples,
    # split=Splits
)
# pmc_hyper_params = dict(rho=Rhos)
# pmc_hyper_params.update(mc_hyper_params)
pmc_hyper_params = mc_hyper_params.copy()
