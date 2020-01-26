import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(prediction)):
    	if prediction[i] == 1:
    		if ground_truth[i] == 0:
    			FP += 1
    		else:
    			TP += 1
    	elif prediction[i] == 0:
    		if ground_truth[i] == 1:
    			FN += 1
    		else:
    			TN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + TN + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    correct_prediction = 0
    for i in range(len(prediction)):
    	if prediction[i] == ground_truth[i]:
    		correct_prediction += 1
    return correct_prediction / len(prediction)
