#TODO: 
# 1. Get rid of sklearn.model import // check if it's ok
# 2. check added imports sum, delete, pi...
# 3. added "kernel" input to evalute c - check that it's ok

from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array, sum, delete, pi, linspace, power
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 

SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the labels - for test
    """
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]
    train_data = array([])
    train_labels = array([])
    test_data = array([])
    test_labels = array([])

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    shuffled = permutation(len(data))
    data = data[shuffled]
    labels = labels[shuffled]
    train_index = int(0.8 * len(data))
    train_data, test_data = data[:train_index], data[train_index:]
    train_labels, test_labels = labels[:train_index], labels[train_index:]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    sample_size = len(prediction)
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for i in range(sample_size):
        if prediction[i] == 1:
            if labels[i] == 1:
                accuracy += 1
                tp += 1
            else:
                fn += 1
        else:
            if labels[i] == 0:
                accuracy += 1
                tn += 1
            else:
                fp += 1        
    fpr = fn / (fn + tn)
    tpr = tp / (tp + fp)
    accuracy /= sample_size
    ###########################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    for i in range(len(folds_array)):
        predictions = []
        tr_data = concatenate(delete(folds_array,i,0))
        tr_labels = concatenate(delete(labels_array,i,0))
        clf.fit(tr_data, tr_labels)
        for row in folds_array[i]:
            predict = clf.predict(row.reshape(1, -1))
            predictions.append(predict)
        _tpr, _fpr, _accuracy = get_stats(predictions, labels_array[i])
        tpr.append(_tpr)
        fpr.append(_fpr)
        accuracy.append(_accuracy)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly',
                               'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = None
    svm_df['fpr'] = None
    svm_df['accuracy'] = None

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    folds_array = array_split(data_array, folds_count)
    labels_array = array_split(labels_array, folds_count)
    tpr = []
    fpr = []
    accuracy = []
    for i in range(len(kernels_list)):
        clf = SVC(kernel=kernels_list[i])
        clf.set_params(**kernel_params[i])
        _tpr, _fpr, _accuracy = get_k_fold_stats(folds_array, labels_array, clf)
        tpr.append(_tpr)
        fpr.append(_fpr)
        accuracy.append(_accuracy)
    svm_df['tpr'] = tpr
    svm_df['fpr'] = fpr
    svm_df['accuracy'] = accuracy  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return svm_df


def get_most_accurate_kernel(accuracies):
    """
    :return: integer representing the row number of the most accurate kernel
    """
    
    best_kernel = 0
    best_accuracy = 0
    index = 0
    for accuracy in accuracies:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_kernel = index
        index += 1
    return best_kernel


def get_kernel_with_highest_score(scores):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    best_kernel = 0
    best_score = 0
    index = 0
    for score in scores:
        if score > best_score:
            best_score = score
            best_kernel = index
        index += 1
    return best_kernel


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    a = linspace(0,1,100)
    b = y[5] - x[5]*alpha_slope
    f = alpha_slope*a + b
    plt.title('FPR Vs. TPR')
    area = pi*20
    colors = ["Red", "Green" , "Blue", "Black", "Yellow", "Orange"]
    names = ["poly deg 2", "poly deg 3", "poly deg 4", "RBF gamma 0.005", "RBF gamma 0.05", "RBF gamma 0.5"]
    print("fpr   tpr")
    for i in range(0,6):
        plt.scatter(x[i], y[i], area, color=colors[i], alpha=0.5, label=names[i])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.plot(a,f,'-r',label="alpha_slope line")
    plt.axis([0,1.2,0.8,1.4])
    plt.legend()
    plt.show()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def evaluate_c_param(data_array, labels_array, folds_count, kernel_type, best_kernel_params):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """
    res = pd.DataFrame()
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    res['c values'] = None
    res['tpr'] = None
    res['fpr'] = None
    res['accuracy'] = None
    c_values = []
    for i in range(-4,1):
        for j in range(1,3):
            c = power(10.0,i) * (j/3)
            c_values.append(c)
    res['c values'] = c_values
    folds_array = array_split(data_array, folds_count)
    labels_array = array_split(labels_array, folds_count)
    tpr = []
    fpr = []
    accuracy = []
    for c_value in res['c values']:
        clf = SVC(C=c_value, kernel=kernel_type)
        clf.set_params(**best_kernel_params)
        _tpr, _fpr, _accuracy = get_k_fold_stats(folds_array, labels_array, clf)
        tpr.append(_tpr)
        fpr.append(_fpr)
        accuracy.append(_accuracy)
    res['tpr'] = tpr
    res['fpr'] = fpr
    res['accuracy'] = accuracy           
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels, best_kernel_params):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    kernel_type = ''
    kernel_params = None
    clf = SVC(class_weight='balanced')  # TODO: set the right kernel
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
