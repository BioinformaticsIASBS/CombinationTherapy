import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#def evaluate(ground_truth, predictions, prediction_scores):
def evaluate(ground_truth, predictions):
    '''
    ground_truth: Correct labels
    predictions: Predicted labels
    predictions_scores: Either probability estimates of the positive class,
                        confidence values, or non-thresholded decision values           
    '''
    set_of_labels = set(ground_truth)
    assert False not in [label in set_of_labels for label in predictions],\
           'Predicted labels must be valid'      
    accuracy = metrics.accuracy_score(ground_truth, predictions)
    precision = metrics.precision_score(ground_truth, predictions)
    recall = metrics.recall_score(ground_truth, predictions)
    f1_score = metrics.f1_score(ground_truth, predictions)
    MCC = metrics.matthews_corrcoef(ground_truth, predictions)
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()

    # Calculate specificity
    specificity = tn / (tn + fp)
    
    #AUROC = metrics.roc_auc_score(ground_truth, prediction_scores)
    #AUPR = metrics.average_precision_score(ground_truth, prediction_scores)

    return [accuracy, precision, recall, specificity, f1_score, MCC]


def ROC_curve(ground_truth, prediction_scores):
    '''
    ground_truth: Correct labels
    predictions_scores: Either probability estimates of the positive class,
                        confidence values, or non-thresholded decision values           
    '''
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth, prediction_scores)
    plt.plot([0,1], [0,1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def PR_curve(ground_truth, prediction_scores):
    '''
    ground_truth: Correct labels
    predictions_scores: Either probability estimates of the positive class,
                        confidence values, or non-thresholded decision values           
    '''
    pre, rec, thresholds = metrics.precision_recall_curve(ground_truth, prediction_scores)
    plt.plot(rec, pre, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

def threshold_tuning(ground_truth, prediction_scores, mode='range', step=0.01):
    '''
    ground_truth: Correct labels
    predictions_scores: Either probability estimates of the positive class,
                        confidence values, or non-thresholded decision values
    mode: method of generating candidate thresholds
    '''
    if mode == 'ROC':
        fpr, tpr, thresholds = metrics.roc_curve(ground_truth, prediction_scores)
        #G-Mean = sqrt(Sensitivity * Specificity)
        gmeans = np.sqrt(tpr * (1-fpr))
        optimal_threshold = thresholds[np.argmax(gmeans)]
        #J-statistic => faster way to get the same result
        #print(thresholds)
        #print(gmeans)
    elif mode == 'PR':
        pre, rec, thresholds = metrics.precision_recall_curve(ground_truth, prediction_scores)
        f1_scores = (2 * pre * rec) / (pre + rec)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        #print(thresholds)
        #print(f1_scores)
    elif mode == 'range':
        thresholds = np.arange(min(ground_truth), max(ground_truth), step)
        scores = []
        for threshold in thresholds:
            predictions = [0 if score < threshold else 1
                           for score in prediction_scores]
            scores.append(metrics.f1_score(ground_truth, predictions))
        optimal_threshold = thresholds[np.argmax(scores)]
        #print(thresholds)
        #print(scores)
    return optimal_threshold
            
        

if __name__ == '__main__':    
    a = [1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    b = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    c = [0.1, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9,
         0.1, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1]
    print(evaluate(a, b, c))
    print(threshold_tuning(a, c, 'range'))
#display curves
#thresholding
#test on our data
#num_thresholds nmigire?
