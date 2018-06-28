# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 20:46:28 2018

@author: 北海若
"""

import tensorflow as tf

trf = tf.contrib.tensor_forest.random_forest
ttf = tf.contrib.tensor_forest.python.tensor_forest

MODEL_DIR = "D:/AI_DataSet/REMI_Model"


def get_model():
    params = ttf.ForestHParams()
    trf.TensorForestEstimator(params, model_dir=MODEL_DIR)
