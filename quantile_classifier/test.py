import cProfile, pstats, StringIO
from sklearn.datasets import load_iris, make_classification
# from sklearn.utils.estimator_checks import check_estimator
import quantile_classifier


def main():

    pr = cProfile.Profile()

    # # currently fails
    # check_estimator(quantile_classifier.QuantileClassifier)

    pdf_classifier = quantile_classifier.QuantileClassifier()

    # X, y = load_iris(True)
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                               n_classes=3, n_redundant=0,
                               n_clusters_per_class=1, random_state=0)

    # pr.enable()
    pr.runcall(pdf_classifier.fit, X, y)
    pdf_classifier.fit(X, y)
    print pdf_classifier.score(X, y)
    # print zip(y, pdf_classifier.predict_proba(X))

    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()
    pr.dump_stats('quantile_classifier.prof')

if __name__ == "__main__":
    main()
