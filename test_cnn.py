# -*- coding: utf-8 -*-
# * *****************************************************************
# *                                                                 *
# * - Programmed by:                                                *
# *     Su Hao, Guan TianYuan, Liu Yan                              *
# *     Computational Dynamics Group, School of Aerospace           *
# *     Engineering, Tsinghua University, 2023.10.24                *
# *                                                                 *
# * *****************************************************************
import numpy as np 
import os
import time
import cnn_model

''' Hyper-parameters  '''
hp = dict()
# model directory
hp['model_dir'] = os.path.join(os.path.abspath('.'), 'model')
hp['data_dir'] = os.path.join(os.path.abspath('.'), 'data')
hp['model_path'] = os.path.join(hp['model_dir'], '40.weights')
# training hyper-parameters
hp['batch_size'] = 1


''' Data Preparation '''
miniBatchXModel = np.load(os.path.join(hp['data_dir'],'XModel.npy'))
miniBatchXLabel = np.load(os.path.join(hp['data_dir'],'XLabel.npy'))
miniBatchYLabel = np.load(os.path.join(hp['data_dir'],'YLabel.npy'))


''' Test step  '''
def test_step(x_batch_test, materials, y_batch_test):
    y_pred = cnn([x_batch_test, materials], trainingFlag=False)
    return y_pred


''' Load model and predict '''
cnn = cnn_model.CNN3DModel(0.0)
cnn.load_weights(hp['model_path'])

time0 = time.time()

y_pred = test_step(miniBatchXModel, miniBatchXLabel, miniBatchYLabel)
y_true = miniBatchYLabel

print("============================")
print("Predicted result: ", y_pred.numpy())
print("Baseline result: ", y_true)
print("Time cost: ", time.time()-time0)