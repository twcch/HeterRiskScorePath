import numpy as np


def mckelvey_zavoina_r2(model, X):
    params = model.params[:-1]  # beta
    sigma = model.params[-1]  # sigma
    mu = np.dot(X, params)  # latent Xβ

    var_xbeta = np.var(mu)
    var_error = sigma ** 2

    return var_xbeta / (var_xbeta + var_error)


