'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code_src.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        scores = precision_recall_fscore_support(self.data['true_y'], self.data['pred_y'], average='weighted')
        precision = scores[0]
        recall = scores[1]
        f1 = scores[2]
        return accuracy_score(self.data['true_y'], self.data['pred_y']), precision, recall, f1

