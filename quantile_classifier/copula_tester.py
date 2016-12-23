import cProfile
import numpy as np
from NonparametricCopula import NonparametricCopula


def main():
    run_profiler = False
    data_low_end = 0.0
    data_high_end = 10.0

    if run_profiler:
        pr = cProfile.Profile()

    test_data = (data_high_end - data_low_end) * \
        np.random.logistic(size=(10000, 4)) + data_low_end

    # np.random.seed(0)
    data = (data_high_end - data_low_end) * \
        np.random.logistic(size=(1000000, 4)) + data_low_end

    pdf = NonparametricCopula(trust_ecdfs=False, reduction_factor=100)
    if run_profiler:
        # pr.enable()
        pr.runcall(pdf.fit, data)
        pr.dump_stats('nonparametric_copula.prof')
        exit()
    pdf.fit(data)

    # TODO: make plots of these to make sure thing look okay
    ecdf = pdf.emds_(test_data)
    pdfs = pdf.emds_.pdf(test_data)
    spdfs = pdf.emds_.spdf(test_data)
    hist, bin_edges = np.histogram(ecdf.ravel(), range=(0., 1.), density=True)
    print 'histogram of ecdf (each bin should be about 0.1):\n', 0.1*hist
    print 'empirical dist.: ', ecdf
    print 'probability dist.: ', pdfs
    print 'scaled probability dist.: ', spdfs
    print 'gaus. trans.: ', pdf._gaussian_coord_transform(test_data)

if __name__ == "__main__":
    main()
