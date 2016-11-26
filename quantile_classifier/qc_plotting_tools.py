import numpy as np
# classifier performance
from sklearn.metrics import roc_curve, auc


def histogram_integral(hist, bin_edges):
    return np.sum(hist*np.diff(bin_edges))


def multi_class_roc_curves(classes, y_test, y_scores, c_name, df=False):
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
    if len(classes) == 2 and df:
        i = 1
        c = classes[i]
        c_y_test = y_test.copy()
        c_y_test_signal = y_test == c
        c_y_test_background = y_test != c
        c_y_test[c_y_test_signal] = 1.0
        c_y_test[c_y_test_background] = 0.0
        fpr[c], tpr[c], _ = roc_curve(c_y_test, y_scores)
        roc_auc[c] = auc(fpr[c], tpr[c])
    else:
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
    colors = cycle(['red', 'aqua', 'darkorange', 'cornflowerblue'])
    if len(classes) == 2 and df:
        i = 1
        c = classes[i]
        color = colors.next()
        plt.plot(fpr[c], tpr[c], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(c, roc_auc[c]))
    else:
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='black', linestyle=':', linewidth=4)

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
    plt.clf()


def compare_train_test_all_classes(classes, train_scores, test_scores, c_name,
                                   y_train, y_test, nbins=30):
    # Compute training and test dataset scores distributions for each class
    # (useful for checking over-/under-training)
    # classes should be something like classifier.classes_
    # *_scores should have dimensions (n_samples, n_classes) = (output of
    # decision_function)
    if len(classes) == 2:
        i = 1
        c = classes[i]
        decisions = []
        cs = []
        for scores, y in ((train_scores, y_train), (test_scores, y_test)):
            d1 = scores[y == c]
            decisions += [d1]
            cs += [c]
            for bi, bc in enumerate(classes):
                if bc == c:
                    continue
                d2 = scores[y == bc]
                decisions += [d2]
                cs += [bc]
        compare_train_test_single_class(decisions, c_name, cs, nbins)
    else:
        for i, c in enumerate(classes):
            decisions = []
            cs = []
            for scores, y in ((train_scores, y_train), (test_scores, y_test)):
                d1 = scores[y == c].T[i]
                decisions += [d1]
                cs += [c]
                for bi, bc in enumerate(classes):
                    if bc == c:
                        continue
                    d2 = scores[y == bc].T[i]
                    decisions += [d2]
                    cs += [bc]
            compare_train_test_single_class(decisions, c_name, cs, nbins)


def compare_train_test_single_class(decisions, c_name, class_ns, nbins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    num_classes = len(decisions)/2
    bk_colors = ['#6C3483', '#1F618D', '#117A65',
                 '#B7950B', '#A04000', '#283747']
    # this is to get rid of very low scores on the left side of the
    # distributions that just act to artificially shift the low bin edge
    dist_fraction_low_edge = 0.005

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    # train dataset
    hist, bins = np.histogram(decisions[0],
                              bins=nbins, range=low_high
                              )
    total_sig_norm = histogram_integral(hist, bins)
    practical_low_edge = bins[
        np.argmax(dist_fraction_low_edge*np.sum(hist) <= hist)]
    plt.hist(decisions[0],
             color='r', alpha=0.75, range=low_high, bins=nbins,
             histtype='stepfilled',
             label='{} (training data)'.format(class_ns[0]))
    data_to_histogram = []
    data_to_histogram_flat = np.empty((0))
    colors_to_plot = []
    label_to_show = []
    for i in range(1, num_classes):
        data_to_histogram.append(decisions[i])
        data_to_histogram_flat = np.concatenate((data_to_histogram_flat,
                                                 decisions[i]))
        colors_to_plot.append(bk_colors[i % num_classes])
        label_to_show.append('{} (training data)'.format(class_ns[i]))
    hist, bins = np.histogram(data_to_histogram_flat,
                              bins=nbins, range=low_high
                              )
    total_bkg_norm = histogram_integral(hist, bins)
    practical_low_edge = min(practical_low_edge,
                             bins[np.argmax(
                                dist_fraction_low_edge*np.sum(hist) <= hist)])
    plt.hist(data_to_histogram,
             color=colors_to_plot, alpha=0.5,
             range=low_high, bins=nbins, stacked=True,
             histtype='stepfilled',
             label=label_to_show)

    # test dataset histograms with error bars
    hist, bins = np.histogram(decisions[num_classes],
                              bins=nbins, range=low_high
                              )
    # rescale
    hist = total_sig_norm*hist/histogram_integral(hist, bins)
    practical_low_edge = min(practical_low_edge,
                             bins[np.argmax(
                                dist_fraction_low_edge*np.sum(hist) <= hist)])
    scale = float(len(decisions[num_classes])) / np.sum(hist)
    err = np.sqrt(hist * scale) / scale

    # width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err,
                 fmt='o', c='r',
                 label='{} (scaled test data)'.format(class_ns[0]))

    data_to_histogram_flat = np.empty((0))
    for i in range(num_classes + 1, 2*num_classes):
        data_to_histogram_flat = np.concatenate((data_to_histogram_flat,
                                                 decisions[i]))
    hist, bins = np.histogram(data_to_histogram_flat,
                              bins=nbins, range=low_high
                              )
    hist = total_bkg_norm*hist/histogram_integral(hist, bins)
    practical_low_edge = min(practical_low_edge,
                             bins[np.argmax(
                                dist_fraction_low_edge*np.sum(hist) <= hist)])
    scale = float(len(decisions[num_classes + 1])) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err,
                 fmt='o', c='b',
                 label='Rest (scaled test data)')

    plt.xlabel("{} output".format(c_name))
    plt.ylabel("Arbitrary units")
    # plt.legend(loc='best')
    lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                     ncol=2, mode="expand", borderaxespad=0.)
    file_name = '{}_test_train_comparison_class_{}.pdf'.format(c_name,
                                                               class_ns[0])
    plt.plot()
    plt.xlim([practical_low_edge, high])
    print 'Saving test-training distributions to file: {}'.format(file_name)
    plt.savefig(file_name, format='pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()


def plot_decision_boundary_2D(X_train, y_train, clf, c_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # create a mesh to plot in
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    # TODO: optimize this stepsize!!!!
    # step size in the mesh
    hx = (x_max - x_min)/40.0
    hy = (y_max - y_min)/40.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx),
                         np.arange(y_min, y_max, hy))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # reshape for contourf plot
    Z = Z.reshape(xx.shape)

    colors = ['#6C3483', '#1F618D', '#117A65', '#B7950B', '#A04000', '#283747']
    # color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}
    color_map = {}
    for i, c in enumerate(clf.classes_):
        color_map[c] = colors[i % len(colors)]

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    for c in clf.classes_:
        c_X_train = X_train[y_train == c]
        c_y_train = y_train[y_train == c]
        colors = [color_map[y] for y in c_y_train]
        plt.scatter(c_X_train[:, 0], c_X_train[:, 1],
                    c=colors, cmap=plt.cm.Paired,
                    label='{}'.format(c))

    # file_name = '{}_test_train_comparison_class_{}.pdf'.format(c_name,
    #                                                            class_ns[0])
    # plt.plot()
    # plt.xlim([practical_low_edge, high])
    # print 'Saving test-training distributions to file: {}'.format(file_name)
    # plt.savefig(file_name, format='pdf',
    #             bbox_extra_artists=(lgd,), bbox_inches='tight')
    lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                     ncol=2, mode="expand", borderaxespad=0.)
    plt.plot()
    file_name = '{}_decision_boundary.pdf'.format(c_name)
    print 'Saving 2D decision boundary plot to file: {}'.format(file_name)
    plt.savefig(file_name, format='pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
