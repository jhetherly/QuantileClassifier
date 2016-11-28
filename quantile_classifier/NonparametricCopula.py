import numpy as np
from sklearn.base import BaseEstimator


def remove_bad_weights(X, sample_weight):
    # handle bad weights by grouping nearby points until a net
    # positive/negative weight is found, then averaging over the coordinates of
    # those points
    from scipy.spatial import cKDTree

    sum_weights = np.sum(sample_weight)
    if sum_weights == 0.0:
        raise ValueError(("The sum of weights cannot be zero. "
                          "Please send better data."))
    sum_weights_is_positive = sum_weights > 0.0
    bad_point_indices = sample_weight < 0.0
    worst_points_indices = np.argsort(
                                sample_weight[bad_point_indices])

    tree_ = cKDTree(X)
    # this holds the newly computed averages coords
    new_X = []
    # this holds the new sample weight
    new_sample_weight = []
    # this holds the indices to remove in both the X and sample_weight
    # arrays
    # indices_to_remove = []
    indices_to_remove_set = set()
    n_samples_total = X.shape[0]

    X_bad = X[bad_point_indices][worst_points_indices]
    for row_i in range(X_bad.shape[0]):
        coord = X_bad[row_i]
        for n_neighbors in range(1, n_samples_total):
            dd, ii = tree_.query([coord], k=n_neighbors)
            if n_neighbors != 1:
                ii = ii[0]
            ii = [x for x in ii if x not in indices_to_remove_set]
            if len(ii) == 0:
                break
            elif len(ii) == 1:
                continue
            ws = sample_weight[ii]
            sum_ws = np.sum(ws)
            if (sum_weights_is_positive and sum_ws > 0.0) or \
                    sum_ws < 0.0:
                for iii in ii:
                    indices_to_remove_set.add(iii)
                new_sample_weight.append(np.sum(ws))
                new_X.append(np.average(X[ii],
                             weights=np.abs(ws), axis=0))
                break
    X = np.delete(X, list(indices_to_remove_set), axis=0)
    sample_weight = np.delete(sample_weight,
                              list(indices_to_remove_set), axis=0)
    X = np.concatenate((X, np.array(new_X)), axis=0)
    sample_weight = np.concatenate((sample_weight,
                                    new_sample_weight), axis=0)
    return X, sample_weight


class _EmpiricalMarginalDistributions:
    """
    Compute the empirical marginal distributions given a set of data
    """

    def __init__(self, X, sample_weight, trust=True):
        from scipy.interpolate import PchipInterpolator

        # store the marginal (per feature) empirical cumulative
        # distributions (assumes X has dimensions (n_samples, n_features))
        # trust determines whether or not to trust the normalization of the
        #   data or estimate it from a fit to the empirical distributions

        # check for monotonic weights
        sum_weights = np.sum(sample_weight)
        if (sum_weights > 0 and
            not np.all(sample_weight >= 0.0)) or \
            (sum_weights < 0 and
             not np.all(sample_weight <= 0.0)) or \
             sum_weights == 0.0:
            raise ValueError(("Weights must be either all positive or "
                              "all negative and cannot sum to zero. Please "
                              "send better data (consider using the"
                              "remove_bad_weights function)."))
            # X, sample_weight = remove_bad_weights(X, sample_weight)

        xs_ = []
        weights_ = []
        ecdfs_ = []
        X_T = X.T
        sorted_indices = np.argsort(X, axis=0)
        xs__ = X_T[np.arange(X_T.shape[0])[:, None],
                   sorted_indices.T]
        weights__ = sample_weight[sorted_indices].T
        ecdfs__ = np.cumsum(sample_weight[sorted_indices], axis=0).T
        for i in range(X_T.shape[0]):
            xs_.append(xs__[i])
            weights_.append(weights__[i])
            ecdfs_.append(ecdfs__[i])
            if ecdfs_[-1][-1] <= 0.0:
                raise ValueError(("Sum of weights is zero or negative! "
                                  "Please send better data."))

        # tediously run through points and find duplicates
        for i in range(len(ecdfs_)):
            xs_i = xs_[i]
            unique_x = np.unique(xs_i)
            if unique_x.size != xs_i.size:
                weights_i = weights_[i]
                new_x = []
                new_weights = []
                previous_x = None
                previous_weight = 0.0
                for ii in range(xs_i.size):
                    xs_ii = xs_i[ii]
                    if previous_x is None:
                        previous_x = xs_ii
                        previous_weight = weights_i[ii]
                        continue
                    if ii == xs_i.size - 1:
                        if xs_ii == previous_x:
                            new_x.append(previous_x)
                            new_weights.append(previous_weight + weights_i[ii])
                        else:
                            new_x.append(previous_x)
                            new_x.append(xs_ii)
                            new_weights.append(previous_weight)
                            new_weights.append(weights_i[ii])
                        continue
                    if xs_ii == previous_x:
                        previous_weight += weights_i[ii]
                    else:
                        new_x.append(previous_x)
                        previous_x = xs_ii
                        new_weights.append(previous_weight)
                        previous_weight = weights_i[ii]
                xs_[i] = np.array(new_x)
                weights_[i] = np.array(new_weights)
                ecdfs_[i] = np.cumsum(weights_[i])

        # save normalizations
        self.norms_ = []
        for row in ecdfs_:
            self.norms_.append(row[-1])

        # now to "normalize" the ecdfs
        if not trust:
            from scipy.optimize import curve_fit

            def generalize_logistic(x, K, B, M, nu):
                exp = np.exp(B*(M - x))
                return K*np.power(1.0 + exp, -1.0/nu)

            def generalize_logistic_gradient(x, K, B, M, nu):
                dif = x - M
                exp = np.exp(-B*(dif))
                power_arg = 1.0 + exp
                denom = 1.0/nu*np.power(power_arg, -1.0/nu - 1.0)
                dK = np.power(power_arg, -1.0/nu)
                dB = K*dif*exp*denom
                dM = -K*B*exp*denom
                dnu = K*np.log(power_arg)/(nu*nu)*dK
                return np.array([dK, dB, dM, dnu]).T

            initial_params = np.array([1.0, 1.0, 0.0, 1.0])
            param_bounds = ([0., 0., -np.inf, 0.],
                            [np.inf, np.inf, np.inf, np.inf])
            for i in range(len(ecdfs_)):
                row = ecdfs_[i]
                peak = row[-1]
                initial_params[0] = peak*1.01
                param_bounds[0][0] = peak
                pars, pcov = curve_fit(generalize_logistic,
                                       xs_[i], row,
                                       p0=initial_params,
                                       bounds=param_bounds,
                                       jac=generalize_logistic_gradient,
                                       method='trf',
                                       xtol=1e-6,
                                       max_nfev=15000,
                                    #    verbose=2,
                                       )
                # param_names = {0: 'norm', 1: 'exp coef',
                #                2: 'offset', 3: 'exp'}
                # self._print_fit_results(pars, pcov, row.size,
                #                         95, param_names)

                # if new peak is less than 5% higher than old one, use it
                if (pars[0] - peak)/peak <= 0.05:
                    self.norms_[i] = pars[0]
                    row /= pars[0]
                else:
                    row /= peak
        else:
            for i in range(len(ecdfs_)):
                peak = ecdfs_[i][-1]
                ecdfs_[i] /= peak

        # now safe to construct monotonic interpolants
        self.ecdfs_ = []
        for i in range(len(ecdfs_)):
            self.ecdfs_.append(PchipInterpolator(xs_[i],
                               ecdfs_[i],
                               extrapolate=True))

    def __call__(self, X):
        # calculate the marginal (per feature) empirical cumulative
        # distributions (assumes X has dimensions (n_samples, n_features))
        return self.evaluate(X)

    def evaluate(self, X):
        # calculate the marginal (per feature) empirical cumulative
        # distributions (assumes X has dimensions (n_samples, n_features))
        result = X.copy()
        result_T = result.T
        X_T = X.T
        for i in range(X.shape[1]):
            result_T[i] = np.clip(self.ecdfs_[i](X_T[i]), 0., 1.)
        return result

    def _print_fit_results(self, pars, pcov, n, ci, param_names):
        from scipy.stats.distributions import t

        alpha = 1. - ci/100.  # 95% confidence interval = 100*(1 - alpha)
        p = len(pars)
        dof = max(0, n - p)

        tval = t.ppf(1.0 - alpha/2., dof)
        print 'Best fit parameters with {}% error interval:'.format(ci)
        for ii, p, var in zip(range(n), pars, np.diag(pcov)):
            sigma = np.sqrt(var)
            print '{0}: {1} [{2} {3}]'.format(param_names[ii], p,
                                              p - sigma*tval,
                                              p + sigma*tval)
        print "Full covariance matrix = \n", pcov


class NonparametricCopula (BaseEstimator):

    def __init__(self, estimator=None, trust_ecdfs=True):
        """Store all values of parameters and nothing else

        Keyword arguments:
        estimator -- estimator to use for 1- and 2-D density estimation
        """
        import inspect

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, sample_weight=None):
        """
        This function determines the structure of the copulae and creates the
        respective pdfs for the various marginal and copula densities
        X has dimensions (n_samples, n_features)
        """
        from sklearn.utils import check_array

        X = check_array(X, order='C')

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        print 'Calculating the marginal empirical distributions'
        self.emd_ = _EmpiricalMarginalDistributions(X,
                                                    sample_weight,
                                                    self.trust_ecdfs)
        print 'Calculating the marginal densities'
