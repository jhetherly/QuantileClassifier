import cProfile  # , pstats, StringIO
import numpy as np
# classifier validation
# from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import make_classification
# data handling
import sklearn
if int(sklearn.__version__.split('.')[1]) >= 18:
    from sklearn.model_selection import train_test_split
else:
    from sklearn.cross_validation import train_test_split
# classifiers
from quantile_classifier import QuantileClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
# classifier performance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score


def main():
    # # currently fails
    # check_estimator(QuantileClassifier)

    run_profiler = False
    rand_state = None  # integer or None
    number_of_classes = 4
    number_of_dimensions = 2
    number_of_samples = 10000

    if run_profiler:
        pr = cProfile.Profile()

    X, y = make_classification(n_samples=number_of_samples,
                               n_features=number_of_dimensions,
                               n_informative=number_of_dimensions,
                               n_classes=number_of_classes, n_redundant=0,
                               n_clusters_per_class=1,
                               random_state=rand_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=rand_state
                                                        )

    pdf_classifier = QuantileClassifier()
    if run_profiler:
        # pr.enable()
        pr.runcall(pdf_classifier.fit, X, y)
    pdf_classifier.fit(X_train, y_train)
    pdf_y_predicted = pdf_classifier.predict(X_test)
    print "\nPerformance of QuantileClassifier:"
    print classification_report(y_test, pdf_y_predicted,
                                # target_names=["background", "signal"]
                                )
    print "Confusion matrix for classes: ", np.unique(y)
    print confusion_matrix(y_test, pdf_y_predicted, labels=np.unique(y))
    if number_of_classes == 2:
        print "Area under ROC curve: {:.4}".format(
                                        roc_auc_score(y_test, pdf_y_predicted))

    if run_profiler:
        # pr.disable()
        # s = StringIO.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print s.getvalue()
        pr.dump_stats('quantile_classifier.prof')

    dt = DecisionTreeClassifier(max_depth=3,
                                min_samples_leaf=0.05*len(X_train))
    bdt = AdaBoostClassifier(dt,
                             algorithm='SAMME',
                             n_estimators=800,
                             learning_rate=0.5)

    bdt.fit(X_train, y_train)
    bdt_y_predicted = bdt.predict(X_test)
    print "\n\nPerformance of BDT:"
    print classification_report(y_test, bdt_y_predicted,
                                # target_names=["background", "signal"]
                                )
    print "Confusion matrix for classes: ", np.unique(y)
    print confusion_matrix(y_test, bdt_y_predicted, labels=np.unique(y))
    if number_of_classes == 2:
        print "Area under ROC curve: {:.4}".format(
                                    roc_auc_score(y_test, bdt_y_predicted))

if __name__ == "__main__":
    main()
