import tensorflow as tf
config = {
          'batch_size' : 16,
          'eps' : 1e-8,
          'split':0,
          'steps_prediction':100,
          'num_epochs':5000,
          'performance_epoch_frequency':1,
          'eta_pred':1e-3,
          'neg_loss_weight':1,
          'steps_test':4,
          'max_grad_norm':10,
          'stratification_pos_ratio':0.5,
          'prepool_layers': [{'type':'dense',
                                'units':128,
                                'activation':tf.nn.relu},
                                {'type'      :'res',
                                 'hidden'   :[128,128,128],
                                 'activation':tf.nn.relu},
                                {'type':'dense',
                                'units':128,
                                'activation':tf.nn.relu},
                                     ],
          'prediction_layers': [{'type':'dense',
                                  'units':128,
                                  'activation':tf.nn.relu},
                                 {'type':'dense',
                                  'units':128,
                                  'activation':tf.nn.relu}],
          'postpool_layers': [{'type':'dense',
                                'units':128,
                                'activation':tf.nn.relu},
                                {'type'      :'res',
                                 'hidden'   :[128,128,128],
                                 'activation':tf.nn.relu},
                                {'type':'dense',
                                'units':128,
                                'activation':tf.nn.relu},
                                     ],
         'c_units':128,
                               }