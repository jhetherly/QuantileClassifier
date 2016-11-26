import cProfile  # , pstats, StringIO
import numpy as np
# classifier validation
# from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import make_classification
# data handling
import sklearn
if int(sklearn.__version__.split('.')[1]) >= 18:
    from sklearn.model_selection import train_test_split, GridSearchCV
else:
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
# classifiers
from quantile_classifier import QuantileClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# classifier performance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from qc_plotting_tools import multi_class_roc_curves, compare_train_test_all_classes, plot_decision_boundary_2D


def show_performance(n, y_test, y_predicted, classes, number_of_classes):
    print "\nPerformance of {}:".format(n)
    print classification_report(y_test, y_predicted,
                                # target_names=["background", "signal"]
                                )
    print "Confusion matrix for classes: ", classes
    print confusion_matrix(y_test, y_predicted, labels=classes)
    if number_of_classes == 2:
        print "Area under ROC curve: {:.4}".format(
                                        roc_auc_score(y_test, y_predicted))


def make_performance_plots(clf, n, number_of_dimensions,
                           X_train, X_test, y_train, y_test):
    print 'Creating performance plot for {}.'.format(n)
    if hasattr(clf, 'decision_function'):
        test_df = clf.decision_function(X_test)
        train_df = clf.decision_function(X_train)
        multi_class_roc_curves(clf.classes_, y_test, test_df, n, df=True)
        compare_train_test_all_classes(clf.classes_, train_df, test_df, n,
                                       y_train, y_test)
    else:
        test_prob = clf.predict_proba(X_test)
        multi_class_roc_curves(clf.classes_, y_test, test_prob, n)
    if number_of_dimensions == 2:
        print 'Creating 2D decision boundary plot. Will take a moment... '
        plot_decision_boundary_2D(X_train, y_train, clf, n)


def main():
    # # currently fails
    # check_estimator(QuantileClassifier)

    create_plots = True
    run_profiler = False
    rand_state = 9  # integer or None
    number_of_classes = 4
    number_of_dimensions = 2
    number_of_samples = 100

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

    print '\n\nRunning QuantileClassifier...'
    pdf_classifier = QuantileClassifier()
    if run_profiler:
        # pr.enable()
        pr.runcall(pdf_classifier.fit, X, y)
    pdf_classifier.fit(X_train, y_train)
    pdf_y_predicted = pdf_classifier.predict(X_test)
    show_performance('QuantileClassifier',
                     y_test, pdf_y_predicted,
                     np.unique(y), number_of_classes)
    if create_plots:
        make_performance_plots(pdf_classifier, 'QuantileClassifier',
                               number_of_dimensions,
                               X_train, X_test, y_train, y_test)

    if run_profiler:
        # pr.disable()
        # s = StringIO.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print s.getvalue()
        pr.dump_stats('quantile_classifier.prof')

    print '\n\nRunning BDT...'
    # NOTE: the values to the right are the 'TMVA-like' parameters
    bdt_param_grid = {'base_estimator__max_depth': [2, 3],  # 3
                      'base_estimator__min_samples_leaf':
                      [0.01*i*len(X_train) for i in [1., 5., 10.]],  # 5.
                      'n_estimators': [100, 200],  # 800
                      'learning_rate': [0.5, 0.75]}  # 0.5
    bdt = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(),
                                          algorithm='SAMME'),
                       param_grid=bdt_param_grid)

    print 'Finding best best fit for BDT parameters...'
    bdt.fit(X_train, y_train)
    bdt_y_predicted = bdt.predict(X_test)
    show_performance('BDT (best fit params {}):'.format(bdt.best_params_),
                     y_test, bdt_y_predicted,
                     np.unique(y), number_of_classes)
    if create_plots:
        make_performance_plots(bdt.best_estimator_, 'BoostedDecisionTree',
                               number_of_dimensions,
                               X_train, X_test, y_train, y_test)

    print '\n\nRunning SVC...'
    svc_param_grid = {'kernel': ('linear', 'rbf'),
                      'C': [1.0, 5.0, 10.0, 20.0],
                      'gamma':
                      [i/number_of_dimensions for i in [0.5, 1.0, 1.5]]}
    svc = GridSearchCV(SVC(decision_function_shape='ovr'),
                       param_grid=svc_param_grid)

    print 'Finding best best fit for SVC parameters...'
    svc.fit(X_train, y_train)
    svc_y_predicted = svc.predict(X_test)
    show_performance('SVC (best fit params {}):'.format(svc.best_params_),
                     y_test, svc_y_predicted,
                     np.unique(y), number_of_classes)
    if create_plots:
        make_performance_plots(svc.best_estimator_, 'SupportVectorMachine',
                               number_of_dimensions,
                               X_train, X_test, y_train, y_test)

    print '\n\nRunning KNeighborsClassifier...'
    knn_param_grid = {'n_neighbors': [1, 5, 10],
                      'algorithm': ['brute']}
    knn = GridSearchCV(KNeighborsClassifier(),
                       param_grid=knn_param_grid)
    # knn = KNeighborsClassifier()

    # print 'Finding best best fit for KNN parameters...'
    print 'Running KNN...'
    knn.fit(X_train, y_train)
    knn_y_predicted = knn.predict(X_test)
    show_performance('KNN (best fit params {}):'.format(knn.best_params_),
    # show_performance('KNeighborsClassifier',
                     y_test, knn_y_predicted,
                     np.unique(y), number_of_classes)
    if create_plots:
        # make_performance_plots(knn, 'KNeighborsClassifier',
        make_performance_plots(knn.best_estimator_, 'KNeighborsClassifier',
                               number_of_dimensions,
                               X_train, X_test, y_train, y_test)

    # Some helpful info for interpreting what printed
    print "\n\nAs a reminder:"
    print ("\n\nThe confusion matrix C_{i, j} is "
           "equal to the number of observations known to be in group i but "
           "predicted to be in group j")
    print ("Precision is the true-positive rate divided by the sum of the "
           "true-positive and false-positive rates. In the confusion matrix "
           "it's a diagonal element divided by the sum of that column.")
    print ("Recall is the true-positive rate divided by the sum of the "
           "true-positive and false-negative rates. In the confusion matrix "
           "it's a diagonal element divided by the sum of that row.")
    print ("f1-score is 2 * (precision * recall) / (precision + recall). For "
           "for multi-class data it's a weighted average of the individual "
           "f1-scores.")
    print ("Support is simply the total number of instances of a class type "
           "in the truth set of labels")

if __name__ == "__main__":
    main()
