import csv
import time
import os
import itertools
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from classifier_evaluation_ML import evaluate, ROC_curve, PR_curve, threshold_tuning

'''
Cross-validation and ...
'''
ratios = [3,5,10,100,500]


kernels = ['linear', 'poly', 'rbf']
Cs = [0.1, 1, 10]


nums_estimators = [10]
criteria = ['gini','log_loss']
methods_max_features = ['sqrt','log2']

svm_metric_lists = [[], [], [], [], [], [], [], [], []]
rf_metric_lists = [[], [], [], [], [], [], [], []]
svm_summary_file = open('../ML_results/SVM summary.txt', 'w')
rf_summary_file = open('../ML_results/RF summary.txt', 'w')

for ratio in ratios:
    print('---------------------- Ratio : 1 to {} ----------------------'.format(ratio))
    ratioed_X = np.load('../Data/ML/' + str(ratio) + '-to-1 X.npz')['arr_0']
    ratioed_y = np.load('../Data/ML/' + str(ratio) + '-to-1 y.npz')['arr_0']

    num_folds = 10
    kfolder = StratifiedKFold(n_splits=num_folds, shuffle=True)
    folds = kfolder.split(ratioed_X, ratioed_y)

    for kernel in kernels:
        for C in Cs:
            print('{} - {}'.format(kernel, C))
            directory = '../ML_results/SVM/' + str(ratio)+ ' ' + str(kernel) + ' - ' + str(C)
            os.mkdir(directory)
            log_file = open(directory + '/log.txt', 'w')
            
            metric_lists = [[], [], [], [], [], [], [], [], []]
            y_hat, pos_class_probs = [], []
            folds, folds_temp = itertools.tee(folds)
            fold_num = 1
            for train_indices, test_indices in folds_temp:
                directory2 = directory + '/fold' + str(fold_num)
                os.mkdir(directory2)
                log_file2 = open(directory2 + '/log.txt', 'w')
                np.savetxt(directory2 + '/train_indices.txt', train_indices)
                np.savetxt(directory2 + '/test_indices.txt', test_indices)
                X_train = ratioed_X[train_indices]
                y_train = ratioed_y[train_indices]
                X_test = ratioed_X[test_indices]
                y_test = ratioed_y[test_indices]

                if ratio == 1:
                    w = None
                else:
                    w = 'balanced'
                classifier = SVC(C=C, kernel=kernel, probability=True, cache_size=1000, class_weight=w)
                t1 = time.time()
                classifier.fit(X_train, y_train)
                t2 = time.time()
                y_hat = classifier.predict(X_test)
                class_probs = classifier.predict_proba(X_test)
                decision_func = classifier.decision_function(X_test)
                t3 = time.time()
                pos_class_probs_0 = np.array([prob[0] for prob in class_probs])
                pos_class_probs_1 = np.array([prob[1] for prob in class_probs])
                metrics = evaluate(y_test, y_hat, pos_class_probs_1, decision_func, "SVM")
                np.savetxt(directory2 + '/y_test.txt', y_test)
                np.savetxt(directory2 + '/y_hat.txt', y_hat)
                np.savetxt(directory2 + '/decision_function.txt', decision_func)
                np.savetxt(directory2 + '/pos_class_probs.txt', class_probs)
                np.savetxt(directory2 + '/pos_class_probs_0.txt', pos_class_probs_0)
                np.savetxt(directory2 + '/pos_class_probs_1.txt', pos_class_probs_1)
                log_file2.write('\Accuracy => {}'.format(metrics[0]))
                log_file2.write('\nPrecision => {}'.format(metrics[1]))
                log_file2.write('\nRecall => {}'.format(metrics[2]))
                log_file2.write('\nSpecificity => {}'.format(metrics[3]))
                log_file2.write('\nF1-score => {}'.format(metrics[4]))
                log_file2.write('\nMCC => {}'.format(metrics[5]))
                log_file2.write('\nAUROC => {}'.format(metrics[6]))
                log_file2.write('\nAUPR => {}'.format(metrics[7]))
                log_file2.write('\nLoss => {}'.format(metrics[8]))
                log_file2.close()
                for i in range(len(metric_lists)):
                    metric_lists[i].append(metrics[i])
                    svm_metric_lists[i].append(metrics[i])
                    
                log_file.write('\nFold: {}, Training time: {:.2f}, Inference time: {:.2f}'.format(fold_num, t2-t1, t3-t2))
                fold_num += 1
            np.savetxt(directory + '/y_test.txt', y_test)
            np.savetxt(directory + '/y_hat.txt', y_hat)
            np.savetxt(directory + '/decision_function.txt', decision_func)
            np.savetxt(directory + '/pos_class_probs.txt', class_probs)
            np.savetxt(directory + '/pos_class_probs_0.txt', pos_class_probs_0)
            np.savetxt(directory + '/pos_class_probs_1.txt', pos_class_probs_1)
            
            
            log_file.write('\nAccuracy => AVG : {}  STDEV : {}'.format(np.average(metric_lists[0]), np.std(metric_lists[0])))
            log_file.write('\nPrecision => AVG : {}  STDEV : {}'.format(np.average(metric_lists[1]), np.std(metric_lists[1])))
            log_file.write('\nRecall => AVG : {}  STDEV : {}'.format(np.average(metric_lists[2]), np.std(metric_lists[2])))
            log_file.write('\nSpecificity => AVG : {}  STDEV : {}'.format(np.average(metric_lists[3]), np.std(metric_lists[3])))
            log_file.write('\nF1-score => AVG : {}  STDEV : {}'.format(np.average(metric_lists[4]), np.std(metric_lists[4])))
            log_file.write('\nMCC => AVG : {}  STDEV : {}'.format(np.average(metric_lists[5]), np.std(metric_lists[5])))
            log_file.write('\nAUROC => AVG : {}  STDEV : {}'.format(np.average(metric_lists[6]), np.std(metric_lists[6])))
            log_file.write('\nAUPR => AVG : {}  STDEV : {}'.format(np.average(metric_lists[7]), np.std(metric_lists[7])))
            log_file.write('\nLoss => AVG : {}  STDEV : {}'.format(np.average(metric_lists[8]), np.std(metric_lists[8])))
            log_file.close()

    for num_estimators in nums_estimators:
        for criterion in criteria:
            for method in methods_max_features:
                print('{} - {} - {}'.format(num_estimators, criterion, method))
                directory = '../ML_results/RF/' + str(ratio)+ ' ' + str(num_estimators) + ' - ' + str(criterion) + ' - ' + str(method)
                os.mkdir(directory)
                log_file = open(directory + '/log.txt', 'w')
                
                metric_lists = [[], [], [], [], [], [], [], []]
                y_hat, pos_class_probs = [], []
                folds, folds_temp = itertools.tee(folds)
                fold_num = 1
                for train_indices, test_indices in folds_temp:
                    directory2 = directory + '/fold' + str(fold_num)
                    os.mkdir(directory2)
                    log_file2 = open(directory2 + '/log.txt', 'w')
                    np.savetxt(directory2 + '/train_indices.txt', train_indices)
                    np.savetxt(directory2 + '/test_indices.txt', test_indices)
                    X_train = ratioed_X[train_indices]
                    y_train = ratioed_y[train_indices]
                    X_test = ratioed_X[test_indices]
                    y_test = ratioed_y[test_indices]

                    classifier = RandomForestClassifier(n_estimators=num_estimators, criterion=criterion, max_features=method)
                    t1 = time.time()
                    classifier.fit(X_train, y_train)
                    t2 = time.time()
                    y_hat = classifier.predict(X_test)
                    class_probs = classifier.predict_proba(X_test)
                    t3 = time.time()
                    pos_class_probs_0 = np.array([prob[0] for prob in class_probs])
                    pos_class_probs_1 = np.array([prob[1] for prob in class_probs])
                    metrics = evaluate(y_test, y_hat, pos_class_probs_1,None,"RF")
                    np.savetxt(directory2 + '/y_test.txt', y_test)
                    np.savetxt(directory2 + '/y_hat.txt', y_hat)
                    np.savetxt(directory2 + '/decision_function.txt', decision_func)
                    np.savetxt(directory2 + '/pos_class_probs.txt', class_probs)
                    np.savetxt(directory2 + '/pos_class_probs_0.txt', pos_class_probs_0)
                    np.savetxt(directory2 + '/pos_class_probs_1.txt', pos_class_probs_1)
                    log_file2.write('\Accuracy => {}'.format(metrics[0]))
                    log_file2.write('\nPrecision => {}'.format(metrics[1]))
                    log_file2.write('\nRecall => {}'.format(metrics[2]))
                    log_file2.write('\nSpecificity => {}'.format(metrics[3]))
                    log_file2.write('\nF1-score => {}'.format(metrics[4]))
                    log_file2.write('\nMCC => {}'.format(metrics[5]))
                    log_file2.write('\nAUROC => {}'.format(metrics[6]))
                    log_file2.write('\nAUPR => {}'.format(metrics[7]))
                    for i in range(len(metric_lists)):
                        metric_lists[i].append(metrics[i])
                        rf_metric_lists[i].append(metrics[i])
                        
                    log_file.write('\nFold: {}, Training time: {:.2f}, Inference time: {:.2f}'.format(fold_num, t2-t1, t3-t2))
                    fold_num += 1
                np.savetxt(directory + '/y_test.txt', y_test)
                np.savetxt(directory + '/y_hat.txt', y_hat)
                np.savetxt(directory + '/pos_class_probs.txt', class_probs)
                np.savetxt(directory + '/pos_class_probs_0.txt', pos_class_probs_0)
                np.savetxt(directory + '/pos_class_probs_1.txt', pos_class_probs_1)
                
                log_file.write('\nAccuracy => AVG : {}  STDEV : {}'.format(np.average(metric_lists[0]), np.std(metric_lists[0])))
                log_file.write('\nPrecision => AVG : {}  STDEV : {}'.format(np.average(metric_lists[1]), np.std(metric_lists[1])))
                log_file.write('\nRecall => AVG : {}  STDEV : {}'.format(np.average(metric_lists[2]), np.std(metric_lists[2])))
                log_file.write('\nSpecificity => AVG : {}  STDEV : {}'.format(np.average(metric_lists[3]), np.std(metric_lists[3])))
                log_file.write('\nF1-score => AVG : {}  STDEV : {}'.format(np.average(metric_lists[4]), np.std(metric_lists[4])))
                log_file.write('\nMCC => AVG : {}  STDEV : {}'.format(np.average(metric_lists[5]), np.std(metric_lists[5])))
                log_file.write('\nAUROC => AVG : {}  STDEV : {}'.format(np.average(metric_lists[6]), np.std(metric_lists[6])))
                log_file.write('\nAUPR => AVG : {}  STDEV : {}'.format(np.average(metric_lists[7]), np.std(metric_lists[7])))
                log_file.close()

svm_summary_file.write('\nAccuracy => AVG : {}  STDEV : {}'.format(np.average(svm_metric_lists[0]), np.std(svm_metric_lists[0])))
svm_summary_file.write('\nPrecision => AVG : {}  STDEV : {}'.format(np.average(svm_metric_lists[1]), np.std(svm_metric_lists[1])))
svm_summary_file.write('\nRecall => AVG : {}  STDEV : {}'.format(np.average(svm_metric_lists[2]), np.std(svm_metric_lists[2])))
svm_summary_file.write('\nSpecificity => AVG : {}  STDEV : {}'.format(np.average(svm_metric_lists[3]), np.std(svm_metric_lists[3])))
svm_summary_file.write('\nF1-score => AVG : {}  STDEV : {}'.format(np.average(svm_metric_lists[4]), np.std(svm_metric_lists[4])))
svm_summary_file.write('\nMCC => AVG : {}  STDEV : {}'.format(np.average(svm_metric_lists[5]), np.std(svm_metric_lists[5])))
svm_summary_file.write('\nAUROC => AVG : {}  STDEV : {}'.format(np.average(svm_metric_lists[6]), np.std(svm_metric_lists[6])))
svm_summary_file.write('\nAUPR => AVG : {}  STDEV : {}'.format(np.average(svm_metric_lists[7]), np.std(svm_metric_lists[7])))
svm_summary_file.write('\nAUPR => AVG : {}  STDEV : {}'.format(np.average(svm_metric_lists[8]), np.std(svm_metric_lists[8])))
svm_summary_file.close()
rf_summary_file.write('\nAccuracy => AVG : {}  STDEV : {}'.format(np.average(rf_metric_lists[0]), np.std(rf_metric_lists[0])))
rf_summary_file.write('\nPrecision => AVG : {}  STDEV : {}'.format(np.average(rf_metric_lists[1]), np.std(rf_metric_lists[1])))
rf_summary_file.write('\nRecall => AVG : {}  STDEV : {}'.format(np.average(rf_metric_lists[2]), np.std(rf_metric_lists[2])))
rf_summary_file.write('\nSpecificity => AVG : {}  STDEV : {}'.format(np.average(rf_metric_lists[3]), np.std(rf_metric_lists[3])))
rf_summary_file.write('\nF1-score => AVG : {}  STDEV : {}'.format(np.average(rf_metric_lists[4]), np.std(rf_metric_lists[4])))
rf_summary_file.write('\nMCC => AVG : {}  STDEV : {}'.format(np.average(rf_metric_lists[5]), np.std(rf_metric_lists[5])))
rf_summary_file.write('\nAUROC => AVG : {}  STDEV : {}'.format(np.average(rf_metric_lists[6]), np.std(rf_metric_lists[6])))
rf_summary_file.write('\nAUPR => AVG : {}  STDEV : {}'.format(np.average(rf_metric_lists[7]), np.std(rf_metric_lists[7])))
rf_summary_file.close()
