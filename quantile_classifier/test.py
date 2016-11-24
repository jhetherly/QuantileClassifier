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
# classifier performance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc


def multi_class_roc_curves(classes, y_test, y_scores, c_name):
    # Compute ROC curve and ROC area for each class
    # classes should be something like classifier.classes_
    # y_scores should have dimensions (n_samples, n_classes) = (output of
    # decision_function)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from itertools import cycle

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_y_test = []
    for i, c in enumerate(classes):
        c_y_test = y_test.copy()
        c_y_test_signal = y_test == c
        c_y_test_background = y_test != c
        c_y_test[c_y_test_signal] = 1.0
        c_y_test[c_y_test_background] = 0.0
        fpr[c], tpr[c], _ = roc_curve(c_y_test, y_scores.T[i])
        roc_auc[c] = auc(fpr[c], tpr[c])
        all_y_test.append(c_y_test)
    all_y_test = np.array(all_y_test)
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_y_test.ravel(),
                                              y_scores.T.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='black', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for c, color in zip(classes, colors):
        plt.plot(fpr[c], tpr[c], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(c, roc_auc[c]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for {}'.format(c_name))
    plt.legend(loc="lower right")
    file_name = '{}_roc_curves.pdf'.format(c_name)
    print 'Saving ROC curves to file: {}'.format(file_name)
    plt.savefig(file_name, format='pdf')


def compare_train_test_all_classes(clf, c_name, X_train, y_train,
                                   X_test, y_test, bins=30):
    decisions = []
    for X, y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y > 0.5]).ravel()
        d2 = clf.decision_function(X[y < 0.5]).ravel()
        decisions += [d1, d2]
        # compare_train_test_single_class


def compare_train_test_single_class(decisions, c_name, class_n, bins=30):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='Class {} (train)'.format(class_n))
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='Rest (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    # width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err,
                 fmt='o', c='r', label='Class {} (test)'.format(class_n))

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='Rest (test)')

    plt.xlabel("{} output".format(c_name))
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    file_name = '{}_test_train_comparison_class _{}.pdf'.format(c_name,
                                                                class_n)
    print 'Saving test-training distributions to file: {}'.format(file_name)
    plt.savefig(file_name, format='pdf')


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

    print '\n\nRunning QuantileClassifier...'
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
    multi_class_roc_curves(pdf_classifier.classes_, y_test,
                           pdf_classifier.decision_function(X_test),
                           'QuantileClassifier')

    if run_profiler:
        # pr.disable()
        # s = StringIO.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print s.getvalue()
        pr.dump_stats('quantile_classifier.prof')

    print '\n\nRunning BDT...'
    # dt = DecisionTreeClassifier(max_depth=3,
    #                             min_samples_leaf=0.05*len(X_train))
    # bdt = AdaBoostClassifier(dt,
    #                          algorithm='SAMME',
    #                          n_estimators=800,
    #                          learning_rate=0.5)
    bdt_param_grid = {'base_estimator__max_depth': [2, 3, 4],
                      'base_estimator__min_samples_leaf':
                      [0.01*i*len(X_train) for i in [1., 5., 10.]],
                      'n_estimators': [500, 800],
                      'learning_rate': [0.5, 0.75]}
    bdt = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(),
                                          algorithm='SAMME'),
                       param_grid=bdt_param_grid)

    print 'Finding best best fit for BDT parameters...'
    bdt.fit(X_train, y_train)
    bdt_y_predicted = bdt.predict(X_test)
    print ("\nPerformance of BDT (best fit params "
           "{}):".format(bdt.best_params_))
    print classification_report(y_test, bdt_y_predicted,
                                # target_names=["background", "signal"]
                                )
    print "Confusion matrix for classes: ", np.unique(y)
    print confusion_matrix(y_test, bdt_y_predicted, labels=np.unique(y))
    if number_of_classes == 2:
        print "Area under ROC curve: {:.4}".format(
                                    roc_auc_score(y_test, bdt_y_predicted))
    # multi_class_roc_curves(bdt.classes_, y_test,
    multi_class_roc_curves(bdt.best_estimator_.classes_, y_test,
                           bdt.decision_function(X_test),
                           'BoostedDecisionTree')

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
    print ("\nPerformance of SVC (best fit params "
           "{}):".format(svc.best_params_))
    print classification_report(y_test, svc_y_predicted,
                                # target_names=["background", "signal"]
                                )
    print "Confusion matrix for classes: ", np.unique(y)
    print confusion_matrix(y_test, svc_y_predicted, labels=np.unique(y))
    if number_of_classes == 2:
        print "Area under ROC curve: {:.4}".format(
                                    roc_auc_score(y_test, svc_y_predicted))
    # multi_class_roc_curves(bdt.classes_, y_test,
    multi_class_roc_curves(svc.best_estimator_.classes_, y_test,
                           svc.decision_function(X_test),
                           'SupportVectorMachine')

    # Some helpful info for interpreting what printed
    print ("\n\nAs a reminder, the confusion matrix is defined as C_{i, j} is "
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
