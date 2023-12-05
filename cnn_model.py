# -*- coding: utf-8 -*-
# * *****************************************************************
# *                                                                 *
# * - Programmed by:                                                *
# *     Su Hao, Guan TianYuan, Liu Yan                              *
# *     Computational Dynamics Group, School of Aerospace           *
# *     Engineering, Tsinghua University, 2023.10.24                *
# *                                                                 *
# * *****************************************************************

import tensorflow as tf

class CNN3DModel(tf.keras.Model):
    
    def __init__(self, dropout):
        super(CNN3DModel, self).__init__()
        self.dropout = dropout
        ''' Block 1: 3*3*3Conv(filters=32, strides=[2,2,2], padding=valid, act=relu) + BN '''
        ''' ImageShape 200*200*200 -> '''
        self.convBlock1Layer1 = tf.keras.layers.Conv3D(32, (3,3,3), strides=(2,2,2), padding='valid')
        self.bnBlock1Layer2 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock1Layer3 = tf.keras.layers.ReLU()
        
        ''' Block 2: 3*3*3Conv(filters=64, strides=[2,2,2], padding=valid, act=relu) + BN '''
        ''' ImageShape -> '''
        self.convBlock2Layer1 = tf.keras.layers.Conv3D(64, (3,3,3), strides=(2,2,2), padding='valid')
        self.bnBlock2Layer2 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock2Layer3 = tf.keras.layers.ReLU()
        
        ''' Block 3: 3*3*3Conv(filters=128, strides=[2,2,2], padding=valid, act=relu) + BN '''
        ''' ImageShape -> '''
        self.convBlock3Layer1 = tf.keras.layers.Conv3D(128, (3,3,3), strides=(2,2,2), padding='valid')
        self.bnBlock3Layer2 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock3Layer3 = tf.keras.layers.ReLU()
        
        ''' Block 4: 3*3*3Conv(filters=256, strides=[1,1,1], padding=valid, act=relu) + BN '''
        ''' ImageShape -> '''
        self.convBlock4Layer1 = tf.keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='valid')
        self.bnBlock4Layer2 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock4Layer3 = tf.keras.layers.ReLU()
        
        ''' Block 5: 3*3*3Conv(fliters=512, strides=[2,2,2], padding=valid, act=relu) + BN '''
        ''' ImageShape -> '''
        self.convBlock5Layer1 = tf.keras.layers.Conv3D(512, (3,3,3), strides=(2,2,2), padding='SAME')
        self.bnBlock5Layer2 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock5Layer3 = tf.keras.layers.ReLU()
        
        ''' Block 6: 3*3*3Conv(fliters=512, strides=[2,2,2], padding=valid, act=relu) + BN '''
        ''' ImageShape -> '''
        self.convBlock6Layer1 = tf.keras.layers.Conv3D(512, (3,3,3), strides=(2,2,2), padding='SAME')
        self.bnBlock6Layer2 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock6Layer3 = tf.keras.layers.ReLU()
        
        ''' Block 7: Flatten + MLP + BN + Dropout(0.2) + Concat + MLP + BN + Dropout(0.2) 
            + MLP + BN + Dropout(0.2) + MLP + BN + Dropout(0.2) + MLP '''
        self.flattenBlock7Layer1 = tf.keras.layers.Flatten()
        ''' ImageShape -> '''
        self.denseBlock7Layer2 = tf.keras.layers.Dense(1000)
        self.bnBlock7Layer3 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock7Layer4 = tf.keras.layers.ReLU()
        self.dropoutBlock7Layer5 = tf.keras.layers.Dropout(self.dropout)
        ''' ImageShape -> '''
        self.concatBlock7Layer6 = tf.keras.layers.Concatenate(axis=-1)
        ''' ImageShape -> '''
        self.denseBlock7Layer7 = tf.keras.layers.Dense(100)
        self.bnBlock7Layer8 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock7Layer9 = tf.keras.layers.ReLU()
        self.dropoutBlock7Layer10 = tf.keras.layers.Dropout(self.dropout)
        ''' ImageShape -> '''
        self.denseBlock7Layer11 = tf.keras.layers.Dense(10)
        self.bnBlock7Layer12 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock7Layer13 = tf.keras.layers.ReLU()
        self.dropoutBlock7Layer14 = tf.keras.layers.Dropout(self.dropout)
        ''' ImageShape -> '''
        self.denseBlock7Layer15 = tf.keras.layers.Dense(5)
        self.bnBlock7Layer16 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock7Layer17 = tf.keras.layers.ReLU()
        self.dropoutBlock7Layer18 = tf.keras.layers.Dropout(self.dropout)
        ''' ImageShape -> '''
        self.denseBlock7Layer19 = tf.keras.layers.Dense(5)
        self.bnBlock7Layer20 = tf.keras.layers.BatchNormalization(trainable=True)
        self.reluBlock7Layer21 = tf.keras.layers.ReLU()
        self.dropoutBlock7Layer22 = tf.keras.layers.Dropout(self.dropout)
        ''' ImageShape '''
        self.y = tf.keras.layers.Dense(3)

    @tf.function
    def call(self, inputs, trainingFlag):
        self.inputs = inputs
        self.trainingFlag = trainingFlag
        inputImagesTensor = inputs[0]
        inputMaterialsTensor = inputs[1]
        ''' Block1 '''
        x = self.convBlock1Layer1(inputImagesTensor)
        x = self.bnBlock1Layer2(x, training=trainingFlag)
        x = self.reluBlock1Layer3(x)
        ''' Block2 '''
        x = self.convBlock2Layer1(x)
        x = self.bnBlock2Layer2(x, training=trainingFlag)
        x = self.reluBlock2Layer3(x)
        ''' Block3 '''
        x = self.convBlock3Layer1(x)
        x = self.bnBlock3Layer2(x, training=trainingFlag)
        x = self.reluBlock3Layer3(x)
        ''' Block4 '''
        x = self.convBlock4Layer1(x)
        x = self.bnBlock4Layer2(x, training=trainingFlag)
        x = self.reluBlock4Layer3(x)
        ''' Block5 '''
        x = self.convBlock5Layer1(x)
        x = self.bnBlock5Layer2(x, training=trainingFlag)
        x = self.reluBlock5Layer3(x)
        ''' Block6 '''
        x = self.convBlock6Layer1(x)
        x = self.bnBlock6Layer2(x, training=trainingFlag)
        x = self.reluBlock6Layer3(x)
        ''' Block7 '''
        x = self.flattenBlock7Layer1(x)
        x = self.denseBlock7Layer2(x)
        x = self.bnBlock7Layer3(x, training=trainingFlag)
        x = self.reluBlock7Layer4(x)
        x = self.dropoutBlock7Layer5(x, training=trainingFlag)
        x = self.concatBlock7Layer6([x, inputMaterialsTensor])
        x = self.denseBlock7Layer7(x)
        x = self.bnBlock7Layer8(x, training=trainingFlag)
        x = self.reluBlock7Layer9(x)
        x = self.dropoutBlock7Layer10(x, training=trainingFlag)
        x = self.denseBlock7Layer11(x)
        x = self.bnBlock7Layer12(x, training=trainingFlag)
        x = self.reluBlock7Layer13(x)
        x = self.dropoutBlock7Layer14(x, training=trainingFlag)
        x = self.denseBlock7Layer15(x)
        x = self.bnBlock7Layer16(x, training=trainingFlag)
        x = self.reluBlock7Layer17(x)
        x = self.dropoutBlock7Layer18(x, training=trainingFlag)
        x = self.denseBlock7Layer19(x)
        x = self.bnBlock7Layer20(x, training=trainingFlag)
        x = self.reluBlock7Layer21(x)
        x = self.dropoutBlock7Layer22(x, training=trainingFlag)
        return self.y(x)
    
    def setTrainingFlag(self, trainingFlag):
        self.trainingFlag = trainingFlag

    
    
    
    
        
        
        
        
    
    
        
        
        
        
