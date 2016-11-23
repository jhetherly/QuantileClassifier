from sklearn.datasets import load_iris, make_classification
# from sklearn.utils.estimator_checks import check_estimator
import quantile_classifier


def main():

    # # currently fails
    # check_estimator(quantile_classifier.QuantileClassifier)

    pdf_classifier = quantile_classifier.QuantileClassifier()

    # X, y = load_iris(True)
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                               n_classes=3, n_redundant=0,
                               n_clusters_per_class=1, random_state=0)

    pdf_classifier.fit(X, y)
    print pdf_classifier.score(X, y)
    # print zip(y, pdf_classifier.predict_proba(X))

if __name__ == "__main__":
    main()
