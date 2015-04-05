'''
Contains a set of utilities used in the project.
'''

from sklearn import metrics

class Performance:

    @staticmethod
    def get_perf(y, y_pred):
        '''
        This method outputs several performance metrics for classification.
        '''

        # Gets Confusion Matrix
        #conf_matrix = metrics.confusion_matrix(y_true = y, y_pred = y_pred)

        # Gets Accuracy
        accuracy = metrics.accuracy_score(y_true = y, y_pred = y_pred)

        # Gets Recall
        recall = metrics.recall_score(y_true = y, y_pred = y_pred)

        # Gets Precision
        precision = metrics.precision_score(y_true = y, y_pred = y_pred)

        # F1
        f1 = metrics.f1_score(y_true = y, y_pred = y_pred)

        return {'accuracy': accuracy, 'recall': recall,
                'precision': precision, 'F1': f1}
