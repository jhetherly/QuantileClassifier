import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from fastkde import fastKDE


def _LLR_metric(s, b):
    """
    This is the standard significance metric used for data analysis at the
    LHC's ATLAS experiment at CERN (the part in the middle). Taken from
    equation 97 in https://arxiv.org/abs/1007.1727
    """
    # default result is zero
    result = np.zeros(s.size, dtype=np.dtype('Float64'))

    # as long as s >= 0 and b > 0, this function is well behaved
    valid_sb_indices_ = np.logical_and(s >= 0, b > 0)
    if np.count_nonzero(valid_sb_indices_) > 0:
        # this operation can be numerically unstable (adding very small
        #   numbers to large numbers (s+b) is always risky), thus the absolute
        #   value of the argument to the sqrt must be taken
        # the form of this calculation is written this way to maximize
        #   numerical stability, not efficiency
        s_, b_ = s[valid_sb_indices_], b[valid_sb_indices_]
        sum_ = np.add(s_, b_, dtype=np.dtype('Float64'))
        log_sum_ = np.empty(sum_.size, np.dtype('Float64'))
        np.log(sum_, out=log_sum_)
        log_b_ = np.empty(b_.size, np.dtype('Float64'))
        np.log(b_, out=log_b_)
        result[valid_sb_indices_] = \
            np.sqrt(np.absolute(2.0*(sum_*(log_sum_ - log_b_) - s_)))

    # else, just return a Poisson error in the case of zero background
    valid_s_indices_ = np.logical_and(s > 0, np.logical_not(valid_sb_indices_))
    if np.count_nonzero(valid_s_indices_) > 0:
        s_ = s[valid_s_indices_]
        div = np.empty(s_.size, np.dtype('Float64'))
        np.divide(s_, np.sqrt(s_), div)
        result[valid_s_indices_] = div

    return result


# TODO: implement pair-copula, currently only works for 2D data
class FastKDE:
    """
    Nicer interface for fastkde evaluations
    Uses a custom pair-copula construction for dimension > 2
    """

    def __init__(self, X, sample_weight,
                 use_copulas=True,
                 interp_method='grid',
                 **kw):
        """
        X must have dimensions            (n_samples, dimension)
        sample_weight must have dimension (n_samples)
        use_copulas is a boolean that should improve computational performance
        for high-dimensional data at the expense of some accuracy
        interp_method dictates how the estimated pdfs are evaluated between the
        grid points (look at scipy.interpolate.griddata for more detail)

        kw are arguments passed directly to the fastKDE constructor - the
        documentation for the fastKDE constructor is quoted (almost) verbatim
        below along with the names and default values for the parameters

        From the fastKDE constructor:
        Estimates the density function of a given dataset using the
        self-consistent method of Bernacchia and Pigolotti (2011, J. R.
        Statistic Soc. B.).  Prior to estimating the PDF, the data are
        standardized to have a mean of 0 and a variance of 1. Standardization
        is done so that PDFs of varying widths can be calculated on a unified
        grid; the original PDF can be re-obtained by scaling, offsetting,
        and renormalizing the calculated PDF.  Assuming the PDF is reasonably
        narrow, then most of the information in the PDF should be contained in
        the returned domain.  The width of the domain is set in terms of
        multiples of unit standard deviations of the data; the default is
        20-sigma.

        ...

        beVerbose (False)   : print debugging information <for 'FastKDE' as
                              well>

        axes (None)         : the axis-values of the estimated PDF.  They must
                              be evenly spaced and they should have a length
                              that is a power of two
                              plus one (e.g., 33). <a function called
                              nextHighestPowerOfTwo in the fastKDE namespace
                              helps with this>

        logAxes (False)     : Flags whether axes should be log spaced (i.e.,
                              the PDF is calculated based on log(data) and
                              then transformed back to sample space).  Should
                              be a logical value (True or False) or a list of
                              logical values with an item for each variable
                              (i.e, len(logAxes) == shape(data)[0]) specifying
                              which axes should use log spacing.  If only True
                              or False is given, that value is used for all
                              variables.

        numPointsPerSigma (10): the number of points on the data grid per
                                standard deviation; this influences the total
                                size of the axes that are automatically
                                calculated if no other aspects of the grid are
                                specified.

        numPoints (None)    : the number of points to use for the pdf grid. If
                              provided as a scalar, each axis will have the
                              same number of points. Otherwise, it should be an
                              iterable with a value for each axis length.
                              Axis lengths should be a power
                              of two plus one (e.g., 33)

        doApproximateECF (True): flags whether to approximate the ECF using a
                                 (much faster) FFT. In tests, this is accurate
                                 to ~1e-14 over low frequencies, but is
                                 inaccurate to ~1e-2 for the highest ~5% of
                                 frequencies.

        ecfPrecision (1)    : sets the precision of the approximate ECF.  If
                              set to 2, it uses double precision accuracy; 1
                              otherwise

        doFFT (True)        : flags whether to calculate phiSC and its FFT to
                              obtain pdf

        doSaveMarginals (True): flags whether to calculate and save the
                                marginal distributions

        fracContiguousHyperVolumes (1): the fraction of contiguous hypervolumes
                                        of the ECF, that are above the ECF
                                        threshold, to use in the density
                                        estimate

        numContiguousHyperVolumes (None): like fracContiguousHyperVolumes, but
                                          specify an integer number to use.
                                          fracContiguousHyperVolumes will be
                                          ignored if this is provided as an
                                          argument.

        positiveShift (True): translate the PDF vertically such that the
                              estimate is positive or 0 everywhere

        axisExpansionFactor (1.0): sets the amount by which the KDE grid will
                                   be expanded relative to the original
                                   min-max spread for each variable: 1.0 means
                                   a 100% (2x) expansion in the range.  Such
                                   an expansion is necessary to avoid kernel
                                   power from one end of the grid leaking into
                                   the opposite end due to the perioidicity of
                                   the Fourier transform.

        doSaveTransformedKernel (False): <undocumented, effects the 'kSC'
                                          and 'kappaSC' properties of the
                                          fastKDE object - likely worthless
                                          for our usage>
        """
        from scipy.spatial import KDTree
        from scipy.interpolate import griddata, interpn

        self.use_copulas_ = use_copulas

        if 'beVerbose' in kw:
            self.beVerbose = kw['beVerbose']
        else:
            self.beVerbose = False

        # pdf_obj = fastKDE.fastKDE(X)
        pdf, axes = fastKDE.pdf(*X.T, **kw)
        # print "sanity check: ", (pdf_obj.pdf == pdf)
        # TODO: make sure this works with regards to the comment below
        # regarding the flipped nature of the pdf
        _interp_points = FastKDE.expand_axes(axes)
        self.interp_points_tree = KDTree(_interp_points)

        # TODO: contact fastKDE author after inspecting source code
        # After inspecting the dimensions of the output with the following
        # inputs:
        # X, y = make_classification(n_samples=1000, n_features=2,
        #                            n_informative=2,
        #                            n_classes=3, n_redundant=0,
        #                            n_clusters_per_class=1, random_state=0)
        # it seems like the x-y axes are flippped(!) in the pdf grid. Thus,
        # I need to take the transpose of the pdf grid. The following
        # statements demonstrate this below.
        # print np.max(X.T, axis=1)
        # print np.min(X.T, axis=1)
        # print axes[0][-1], axes[1][-1]
        # print axes[0][0], axes[1][0]
        # print pdf.shape, axes[0].shape, axes[1].shape
        # return

        if interp_method == 'grid':
            # faster method that relies on data being placed on a grid
            self.interp = lambda x: interpn(points=axes,
                                            values=pdf.T,
                                            xi=x,
                                            method='splinef2d',
                                            bounds_error=False,
                                            fill_value=np.nan
                                            )
        else:
            # slower, more generic method
            self.interp = lambda x: griddata(points=_interp_points,
                                             values=np.ravel(pdf.T),
                                             xi=x,
                                             method='cubic',
                                             rescale=True,
                                             fill_value=np.nan
                                             )

    @staticmethod
    def expand_axes(X):
        """
        Utility method for making cartesian product (could just use
        sklearn.utils.extmath, but this is slightly faster)
        """
        # check for "list of list"
        if all(type(elem).__module__ == np.__name__ for elem in X):
            return np.array(np.meshgrid(*X)).T.reshape(-1, len(X))
        # otherwise, for just the single list
        return X

    def __call__(self, X):
        """
        X must have dimensions (n_samples, dimension)
        """
        _result = self.interp(X)
        # np.nan's show up whenever you try to interpolate beyond the input
        # grid (except when the interpolation method is 'nearest', not
        # recommended). To circumvent this issue this method uses a KDTree
        # built from the original input grid to find the nearest grid points
        # in the original grid and evaluates those points instead.
        # WARNING: evaluations can be small negative numbers due to
        #          interpolation effects (thus, absolute values must be taken)
        _global_nan_indices = np.where(np.isnan(_result))
        if len(_global_nan_indices[0]) == 0:
            return np.absolute(_result)
        _coordinates_of_nan_indices = X[_global_nan_indices]
        _dists, _indices = self.interp_points_tree.query(
                                                _coordinates_of_nan_indices)
        _result[_global_nan_indices] = self.__call__(
                                        self.interp_points_tree.data[_indices])
        return np.absolute(_result)


class QuantileClassifier (BaseEstimator, ClassifierMixin):
    """
    Classifier that selects signal through construction of the full,
    multi-dimensional PDFs for each class and comparing quantiles of metric
    distributions.

    interface:
        default constructor
        fit
        predict
        predict_proba
        score
        get_sub_distributions
    public members:
        all scikit-learn classifier standard stuff
        pdfs_
        pdf_weights_
        class_distributions_
        metric_distributions_
    """

    def __init__(self, use_copulas=True, metric=_LLR_metric):
        """Store all values of parameters and nothing else

        Keyword arguments:
        use_copulas -- Use pairwise copula deconstruction to build
                       multi-dimensional PDF (default True)
        metric -- Callable used to calculate the significance of a point given
                  given the signal and background yields (default CERN ATLAS
                  LLR). Should operate on numpy arrays and the convention is
                  that larger values of the metric are more "signal" like.
        """
        import inspect

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y, sample_weight=None):
        """
        This function creates the pdf's (one for each class) and stores them
        for later use along with the distributions of metrics per class
        """
        from sklearn.utils.multiclass import unique_labels
        from sklearn.utils.validation import check_X_y

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        #X, y = check_X_y(X, y, multi_output=True)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        #self.classes_, y = np.unique(y, return_inverse=True)

        self.X_ = X
        self.y_ = y

        use_copulas = self.use_copulas
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if self.metric is None:
            metric = _LLR_metric
        else:
            metric = self.metric

        # index is the class label
        self.pdfs_ = {}
        self.pdf_weights_ = {}
        self.class_distributions_ = {}
        for class_label in self.classes_:  # pdf index
            _X = X[y == class_label]
            _sample_weight = sample_weight[y == class_label]
            self.pdfs_[class_label] = FastKDE(_X,
                                              _sample_weight,
                                              use_copulas,
                                              interp_method='grid')
            self.pdf_weights_[class_label] = _sample_weight.sum()
            self.class_distributions_[class_label] = \
                self.pdf_weights_[class_label]*self.pdfs_[class_label](X)

        self.metric_distributions_ = {}
        for class_label in self.classes_:  # signal index
            _signal_total = self.class_distributions_[class_label]
            _background_total = None

            for k, v in self.class_distributions_.items():  # background index
                if class_label == k:
                    continue
                _background_total = v if _background_total is None else\
                    _background_total + v

            self.metric_distributions_[class_label] = \
                metric(_signal_total, _background_total)

        # Return the classifier for chaining
        return self

    def predict(self, X):
        from sklearn.utils.validation import check_array, check_is_fitted

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        if self.metric is None:
            metric = _LLR_metric
        else:
            metric = self.metric

        # logic here
        percentile_ratios_ = self._compute_percentile_ratio_dict(X, metric)
        # find maximum
        best_percentile_ratios_ = \
            np.array(percentile_ratios_.keys())[np.argmax(np.array(
                                        percentile_ratios_.values()), axis=0)]

        return best_percentile_ratios_

    def predict_proba(self, X):
        from sklearn.preprocessing import normalize

        percentile_ratios_probabilities_ = self.decision_function(X)

        return normalize(percentile_ratios_probabilities_, axis=1, norm='l1')

    def decision_function(self, X):
        from sklearn.utils.validation import check_array, check_is_fitted

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        if self.metric is None:
            metric = _LLR_metric
        else:
            metric = self.metric

        # logic here
        percentile_ratios_ = self._compute_percentile_ratio_dict(X, metric)
        percentile_ratios_probabilities_ = \
            np.array([percentile_ratios_[k] for k in self.classes_]).T

        return percentile_ratios_probabilities_

    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a
        harsh metric since you require for each sample that each label set be
        correctly predicted.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def get_sub_distributions(self, class_label):
        """
        Helper function that returns a dictionary of metric distributions for
        the training data given a class index (must be run after a call to the
        fit function)
        """
        sub_dists = {}
        m_dist = self.metric_distributions_[class_label]
        for subclass_label in self.classes_:
            sub_dists[subclass_label] = \
                np.sort(m_dist[self.y_ == subclass_label])
        return sub_dists

    def _compute_percentile_ratio_dict(self, X, metric):
        """
        Internal helper function that returns a dictionaries of arrays of
        percentile ratios given a class index (must be run after a call to the
        fit function)
        """
        from scipy.stats import percentileofscore

        class_distributions_ = {}
        for class_label in self.classes_:  # pdf index
            class_distributions_[class_label] = \
                self.pdf_weights_[class_label]*self.pdfs_[class_label](X)

        metric_distributions_ = {}
        for class_label in self.classes_:  # signal index
            _signal_total = class_distributions_[class_label]
            _background_total = None

            for k, v in class_distributions_.items():  # background index
                if class_label == k:
                    continue
                _background_total = v if _background_total is None else\
                    _background_total + v

            metric_distributions_[class_label] = \
                metric(_signal_total, _background_total)

        result = {}
        for k, v in metric_distributions_.items():
            training_sub_dists = self.get_sub_distributions(k)
            result_ = np.empty(v.size, dtype=np.dtype('Float64'))

            for i, m in enumerate(v):
                _signal_total = None
                _background_total = None
                for kk, training_dist in training_sub_dists.items():
                    frac_ = 1.0 - 0.01*percentileofscore(training_dist, m)
                    if kk == k:  # signal
                        _signal_total = frac_ if _signal_total is None else\
                                                        _signal_total + frac_
                    else:  # background
                        _background_total = frac_ if _background_total is\
                                        None else _background_total + frac_
                if _background_total + _signal_total == 0:
                    result_[i] = 0.0
                else:
                    result_[i] = float(_signal_total) / \
                            float(_background_total + _signal_total)

            result[k] = result_

        return result
