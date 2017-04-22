from tfs import *

import tensorflow as tf



class MLP(BaseModel):

    def __init__(self, dataset):

        net = [
            ['dense', {
                'units' 	: 1000,
                'act'	: 'relu'
            }],
            ['dense', {
                'units'	: 10,
                'act'	: None
            }]
        ]

        self.logits_train = sequential(dataset.train['images'], net, name ='mlp')
        self.logits_val = sequential(dataset.val['images'], net, name = 'mlp', reuse = True)
        args = {
            'y'             : dataset.train['labels'],
            'y_pred'        : self.logits_train,
            'y_val'         : dataset.val['labels'],
            'y_pred_val'    : self.logits_val,
            'rate'          : FLAGS.rate
        }
        self.optimizer, self.train_summary_op, self.val_summary_op, self.global_step = classify(**args)
        self.train_feed = self.val_feed = None
