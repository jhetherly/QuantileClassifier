import numpy as np
import scipy.stats as stat
from sklearn.base import BaseEstimator


# TODO: write unit tests
def rdc(x, y, k=20, s=1/6., f=np.sin):
    """
    Compute the randomized dependence coefficient
    """
    # this is based on the paper titled (https://arxiv.org/abs/1304.7717):
    # "The Randomized Dependence Coefficient"
    # very good at finding nonlinear correlations among pairs of variables
    # the original was written in R (just 5 lines!), this is my translation
    # to numpy/scipy/scikit-learn (the original code is in the comments)
    from sklearn.cross_decomposition import CCA

    # x <- cbind(apply(as.matrix(x),2,function(u)rank(u)/length(u)),1)
    # y <- cbind(apply(as.matrix(y),2,function(u)rank(u)/length(u)),1)
    x = stat.rankdata(x)/x.size
    y = stat.rankdata(y)/y.size
    x = np.insert(x[:, np.newaxis], 1, 1, axis=1)
    y = np.insert(y[:, np.newaxis], 1, 1, axis=1)
    # x <- s/ncol(x)*x%*%matrix(rnorm(ncol(x)*k),ncol(x))
    # y <- s/ncol(y)*y%*%matrix(rnorm(ncol(y)*k),ncol(y))
    x = np.dot(s/x.shape[1]*x,
               np.random.normal(size=x.shape[1]*k).reshape((x.shape[1], -1)))
    y = np.dot(s/y.shape[1]*y,
               np.random.normal(size=y.shape[1]*k).reshape((y.shape[1], -1)))
    # cancor(cbind(f(x),1),cbind(f(y),1))$cor[1]
    x = np.insert(f(x), x.shape[1], 1, axis=1)
    y = np.insert(f(y), y.shape[1], 1, axis=1)
    # the following is taken from:
    # http://stackoverflow.com/questions/37398856/
    # how-to-get-the-first-canonical-correlation-from-sklearns-cca-module
    cca = CCA(n_components=1)
    x_c, y_c = cca.fit_transform(x, y)
    return np.corrcoef(x_c.T, y_c.T)[0, 1]


# TODO: generalize this to allow for a lower (upper) bound, 
#       for specification of overall sign of weights,
#       and for equality or strict gt/lt
# TODO: write unit tests
def remove_bad_weights(X, sample_weight, limited_features=None):
    # handle bad weights by grouping nearby points until a net
    # positive/negative weight is found, then averaging over the coordinates of
    # those points
    # limited_features is a list of feature indices to find the nearest
    # neighbors along, otherwise all features are used

    if np.all(sample_weight >= 0.0) or np.all(sample_weight <= 0.0):
        # no bad points found, exiting
        return X, sample_weight

    from scipy.spatial import cKDTree

    reduced_feature_search = limited_features is not None
    sum_weights = np.sum(sample_weight)
    if sum_weights == 0.0:
        raise ValueError(("The sum of weights cannot be zero. "
                          "Please send better data."))
    sum_weights_is_positive = sum_weights > 0.0

    # sort points by the worst offenders
    if sum_weights_is_positive:
        bad_point_indices = sample_weight < 0.0
    else:
        bad_point_indices = sample_weight > 0.0
    worst_points_indices = np.argsort(
                                sample_weight[bad_point_indices])

    if not reduced_feature_search:
        tree_ = cKDTree(X)
    else:
        tree_ = cKDTree(X[:, limited_features])
    # this holds the newly computed averaged coords
    new_X = []
    # this holds the new sample weight
    new_sample_weight = []
    # this holds the indices to remove in both the X and sample_weight
    # arrays
    indices_to_remove_set = set()
    n_samples_total = X.shape[0]

    X_bad = X[bad_point_indices][worst_points_indices]
    for row_i in range(X_bad.shape[0]):
        coord = X_bad[row_i]
        for n_neighbors in range(1, n_samples_total):
            if not reduced_feature_search:
                dd, ii = tree_.query([coord], k=n_neighbors)
            else:
                dd, ii = tree_.query([coord[limited_features]], k=n_neighbors)
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


# TODO: write unit tests (with plots)
class _EmpiricalMarginalDistributions:
    """
    Compute the empirical marginal distributions given a set of data
    """

    def __init__(self, X, sample_weight, trust=True, reduction_factor=None):
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
        if reduction_factor is None:
            ecdfs__ = np.cumsum(sample_weight[sorted_indices], axis=0).T
        for i in range(X_T.shape[0]):
            if reduction_factor is None:
                xs_.append(xs__[i])
                weights_.append(weights__[i])
                ecdfs_.append(ecdfs__[i])
            else:
                columns = int(X_T.shape[1]/float(reduction_factor))
                ii = columns*reduction_factor
                xs_.append(np.average(xs__[i][:ii].reshape((columns,
                                                            reduction_factor)),
                                      weights=
                                        np.abs(weights__[i][:ii]).reshape((
                                                  columns, reduction_factor)),
                                      axis=1))
                weights_.append(np.sum(weights__[i][:ii].reshape((
                                        columns, reduction_factor)), axis=1))
                if X_T.shape[1] % reduction_factor != 0:
                    np.concatenate((xs_[-1], [np.average(
                                                        xs__[i][ii:],
                                                        weights=
                                                np.abs(weights__[i][ii:]))]))
                    np.concatenate((weights_[-1], [np.sum(weights__[i][ii:])]))
                ecdfs_.append(np.cumsum(weights_[-1]))

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
        norms_ = np.empty(X_T.shape[0], dtype=np.dtype('Float64'))
        for i, row in enumerate(ecdfs_):
            norms_[i] = row[-1]

        # now to "normalize" the ecdfs
        if not trust:
            from scipy.optimize import curve_fit

            def generalize_logistic(x, K, A, B, M, nu):
                exp = np.exp(B*(M - x))
                return A + (K - A)*np.power(1.0 + exp, -1.0/nu)

            def generalize_logistic_gradient(x, K, A, B, M, nu):
                dif = x - M
                exp = np.exp(-B*(dif))
                power_arg = 1.0 + exp
                denom = 1.0/nu*np.power(power_arg, -1.0/nu - 1.0)
                dK = np.power(power_arg, -1.0/nu)
                dA = 1. - dK
                dB = (K - A)*dif*exp*denom
                dM = -(K - A)*B*exp*denom
                dnu = (K - A)*np.log(power_arg)/(nu*nu)*dK
                return np.array([dK, dA, dB, dM, dnu]).T

            initial_params = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
            param_bounds = ([0., -1., 0., -np.inf, 0.],
                            [np.inf, 0., np.inf, np.inf, np.inf])
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
                # param_names = {0: 'norm', 1: 'lower bound', 2: 'exp coef',
                #                3: 'offset', 4: 'exp'}
                # self._print_fit_results(pars, pcov, row.size,
                #                         95, param_names)

                # if new peak is less than 5% higher than old one, use it
                if (pars[0] - peak)/peak <= 0.05:
                    norms_[i] = pars[0]
                    row /= pars[0]
                else:
                    row /= peak
        else:
            for i in range(len(ecdfs_)):
                peak = ecdfs_[i][-1]
                ecdfs_[i] /= peak

        # now safe to construct monotonic interpolants
        self.ecdfs_ = []
        self.pdfs_ = []
        for i in range(len(ecdfs_)):
            self.ecdfs_.append(PchipInterpolator(xs_[i],
                               ecdfs_[i],
                               extrapolate=True))
            self.pdfs_.append(self.ecdfs_[-1].derivative())
        self.norm_ = np.max(norms_)

    def __call__(self, X):
        # calculate the marginal (per feature) empirical cumulative
        # distributions (assumes X has dimensions (n_samples, n_features))
        return self.cdf(X)

    def cdf(self, X):
        # calculate the marginal (per feature) empirical cumulative
        # distributions (assumes X has dimensions (n_samples, n_features))
        result = X.copy()
        result_T = result.T
        X_T = X.T
        for i in range(X.shape[1]):
            result_T[i] = np.clip(self.ecdfs_[i](X_T[i]), 0., 1.)
        return result

    def pdf(self, X, cdfs=None):
        if cdfs is None:
            cdfs = self.cdf(X)
        result = X.copy()
        result_T = result.T
        X_T = X.T
        cdfs_T = cdfs.T
        for i in range(X.shape[1]):
            result_T[i] = self.pdfs_[i](X_T[i])
            result_T[i][cdfs_T[i] == 1.] = 0.
            result_T[i][cdfs_T[i] == 0.] = 0.
        return result

    def spdf(self, X):
        """Return the pdfs scaled by the overall normalization"""
        return self.pdf(X)*self.norm_

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

    def __init__(self, trust_ecdfs=True, reduction_factor=None):
        """Store all values of parameters and nothing else

        Keyword arguments:
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
        from sklearn.neighbors import KernelDensity
        from scipy.sparse.csgraph import minimum_spanning_tree
        import itertools

        X = check_array(X, order='C')

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        self.emds_ = _EmpiricalMarginalDistributions(X,
                                                     sample_weight,
                                                     trust=self.trust_ecdfs,
                                                     reduction_factor=
                                                     self.reduction_factor)

        abs_weights = np.abs(sample_weight)
        reduced_total = int(X.shape[0]/float(self.reduction_factor))
        # randomly sample points, but weigh them by sample weights
        reduced_indices = np.random.choice(X.shape[0], size=reduced_total,
                                           replace=False,
                                           p=abs_weights /
                                           np.sum(np.abs(sample_weight)))
        U_ = self.emds_.cdf(X)
        reduced_U_ = U_[reduced_indices]
        reduced_weights_ = sample_weight[reduced_indices]
        if True:
            U_rebinned = self._make_2D_linear_binning_for_copula(
                            U=reduced_U_[:, [0, 1]],
                            weights=reduced_weights_)
            print U_rebinned
            print U_rebinned.sum()
            print U_rebinned.shape
        exit()
        # Z_ = self._gaussian_coord_transform(X)
        # self._first_tree_reduced_Z = Z_[reduced_indices]
        # self._first_tree_kdes = []
        # graph = np.zeros((X.shape[1], X.shape[1]))
        # for pair in itertools.combinations(np.arange(X.shape[1]), 2):
        #     graph[pair] = rdc(self._first_tree_reduced_U[:, pair[0]],
        #                       self._first_tree_reduced_U[:, pair[1]])
        # for indices in minimum_spanning_tree(-1*graph).nonzero():
        #     Z_pair = self._first_tree_reduced_Z[:, indices]
        #     # denom = np.product(stat.norm.pdf(Z_pair), axis=1)
        #
        #     self._first_tree_kdes.append(
        #         [indices,
        #          lambda u,
        #          kde=KernelDensity(kernel='gaussian',
        #                            bandwidth=0.1).fit(Z_pair):
        #          np.exp(kde.score_samples(u))])

    def score_samples(self, X):
        from sklearn.utils import check_array

        X = check_array(X, order='C')

    # def _gaussian_transform(self, U):
    #     return stat.norm.pdf(stat.norm.ppf(U))

    def _gaussian_coord_transform(self, X):
        return stat.norm.ppf(self.emds_(X))

    # TODO: possibly jit this or implement in cython
    def _make_2D_linear_binning_for_copula(self, U, weights, size=(50, 50)):
        """
        Linear binning is used for down-sampling data while retaining much
        higher fidelity (in terms of asymptotic behavior) than nearest-neighbor
        binning (the usual type of binning).
        A-----------------------------------B
         |       |                         |
         |                                 |
         |       |                         |
         |- - - -P- - - - - - - - - - - - -|
         |       |                         |
        D-----------------------------------C
        For a 2D point P with weight wP:
        Assign a weight to corner A of the proportion of area (times wP)
            between P and C
        Assign a weight to corner B of the proportion of area (times wP)
            between P and D
        Assign a weight to corner C of the proportion of area (times wP)
            between P and A
        Assign a weight to corner D of the proportion of area (times wP)
            between P and B
        """
        result = np.zeros(size)
        bin_sizes = [1.0/float(size[0]), 1.0/float(size[1])]
        bin_area = bin_sizes[0]*bin_sizes[1]
        max_bins = np.array(size) - 1
        for i in range(U.shape[0]):
            row = U[i]
            weight = weights[i]
            low_bins = np.floor(row/bin_sizes).astype(int)
            if np.array_equal(max_bins, low_bins):
                result[low_bins[0], low_bins[1]] += weight
                continue
            low_remainders = row - low_bins.astype(float)*bin_sizes
            high_remainders = bin_sizes - low_remainders
            low_high_remainders = low_remainders.copy()
            low_high_remainders[1] = high_remainders[1]
            low_area = weight*low_remainders[0]*low_remainders[1]/bin_area
            low_high_area =\
                weight*low_high_remainders[0]*low_high_remainders[1]/bin_area
            if low_bins[1] + 1 == size[1]:
                low_area = low_area + low_high_area
                result[low_bins[0], low_bins[1]] += weight - low_area
                result[low_bins[0] + 1, low_bins[1]] += low_area
                continue
            high_low_remainders = high_remainders.copy()
            high_low_remainders[1] = low_remainders[1]
            high_area = weight*high_remainders[0]*high_remainders[1]/bin_area
            high_low_area =\
                weight*high_low_remainders[0]*high_low_remainders[1]/bin_area
            if low_bins[0] + 1 == size[0]:
                low_area = low_area + high_low_area
                result[low_bins[0], low_bins[1]] += weight - low_area
                result[low_bins[0], low_bins[1] + 1] += low_area
                continue
            result[low_bins[0], low_bins[1]] += high_area
            result[low_bins[0], low_bins[1] + 1] += high_low_area
            result[low_bins[0] + 1, low_bins[1]] += low_high_area
            result[low_bins[0] + 1, low_bins[1] + 1] += low_area
        return result
