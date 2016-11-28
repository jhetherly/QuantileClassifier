import numpy as np
from NonparametricCopula import NonparametricCopula


def main():
    # np.random.seed(0)
    data_low_end = -20.0
    data_high_end = 10.0

    test_data = (data_high_end - data_low_end) * \
        np.random.logistic(size=(40, 2)) + data_low_end

    data = (data_high_end - data_low_end) * \
        np.random.logistic(size=(100, 2)) + data_low_end

    pdf = NonparametricCopula(trust_ecdfs=False)
    pdf.fit(data)

    ecdf = pdf.emd_(test_data)
    print 'empirical dist.: ', ecdf

if __name__ == "__main__":
    main()
