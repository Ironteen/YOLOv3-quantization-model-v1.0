#!/usr/bin/env python
# -*- coding:utf-8 -*-
# title            : nn_skeleton.py
# description      : define all the modules of neural network
# author           : Zhijun Tu
# email            : tzj19970116@163.com
# date             : 2017/07/16
# version          : 1.0
# notes            : 
# python version   : 2.7.12 which is also applicable to 3.5
###############################################################  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict
from tensorflow.python.training import moving_averages

from core.config import cfg

def debug_collect(tensor):
  '''
  use to debug: add a tensor in 'debug' set
  sess.run this 'debug' set outside and check the value
  '''
  tf.add_to_collection('debug',tensor)

def numpy2tensor(data):
  '''
  convert a numpy array(or data) to tensor 
  '''
  return tf.convert_to_tensor(data,dtype=tf.float32)

def cabs(tensor,bits):
  '''
  clip the tensor to range of 2^(bits-1)-1 to -2^(bits-1)
  '''
  upper_limit = 2**(bits-1)-1
  lower_limit = -2**(bits-1)
  new_tensor = tf.clip_by_value(tensor,lower_limit,upper_limit)
  return new_tensor

def get_clip(tensor,bits=8,method='round'):
  '''
  get the max N of scaling a small float to a int range
  there are two method: round(may get a better result)
                        floor(safe)
  '''
  limit = numpy2tensor(2**(bits-1))
  max_value = tf.maximum(tf.abs(tf.reduce_min(tensor)),tf.abs(tf.reduce_max(tensor)))
  ratio = tf.div(limit,max_value)
  if method=='round':
    clip = tf.round(tf.div(tf.log(ratio),tf.log(numpy2tensor(2))))
  elif method=='floor':
    clip = tf.floor(tf.div(tf.log(ratio),numpy2tensor(2)))
  return clip

def quantize(x, k):
  '''
  skip the gradient of round(x*n)/n
  '''
  G = tf.get_default_graph()
  n = float(2**k)
  with G.gradient_override_map({"Round": "Identity"}):
    return tf.round(x * n) / n

def quantize_plus(x):
  '''
  skip the gradient of round(x)
  '''
  G = tf.get_default_graph()
  with G.gradient_override_map({"Round": "Identity"}):
    return tf.round(x) 

def handle_round(tensor):
  '''
  a new round method
  handle_round(x) = ceil(x) when the decimal part of a float point is .5
                  = round(x) others  
  '''
  tensor_2x = tf.multiply(tensor,numpy2tensor(2))
  tensor_2x_floor = tf.floor(tensor_2x)
  new_tensor = tf.where(tf.equal(tensor_2x,tensor_2x_floor),tf.ceil(tensor),tf.round(tensor))
  return new_tensor

def quant_tensor(tensor,bits=8,fix=False,scale=1,return_type='int'):
  '''
  quantify a float tensor to a quant int tensor 
  and a quant float tensor of fix bits
  '''
  if not fix:
    clip = get_clip(tensor,bits)
    scale = tf.pow(numpy2tensor(2),clip)

  quant_data = handle_round(tf.multiply(tensor,scale))
  quant_int_data = cabs(quant_data,bits)
  quant_float_data = tf.div(quant_int_data,scale)

  if not fix and return_type=='int':
    return quant_int_data,clip
  elif not fix and return_type=='float':
    return quant_float_data,clip
  elif fix and return_type=='int':
    return quant_int_data
  elif fix and return_type=='float':
    return quant_float_data
  else:
    print('=> Error in quant_tensor module !!! ')

def _w_fold(w, gama, var, epsilon):
  """fold the BN into weight"""
  return tf.div(tf.multiply(gama, w), tf.sqrt(var + epsilon))

def _bias_fold(beta, gama, mean, var, epsilon):
  """fold the batch norm layer & build a bias_fold"""
  return tf.subtract(beta, tf.div(tf.multiply(gama, mean), tf.sqrt(var + epsilon)))

def _variable_on_device(name, shape, initializer, trainable=True):
  '''
  create a new variable of tensor 
  '''
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_on_device_reuse(name, shape, initializer, trainable=True):
  '''
  create a new variable of tensor and reuse after
  '''
  dtype = tf.float32
  with tf.variable_scope('v_scope',reuse=tf.AUTO_REUSE) as scope1:
    if not callable(initializer):
      var = tf.get_variable('v_scope'+name, initializer=initializer, trainable=trainable)
    else:
      var = tf.get_variable(
          'v_scope'+name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  '''
  create a new weight variable of tensor and add L2 norm operation
  '''
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


class ModelSkeleton(object):
  """
  build basic module of all kinds of Nerual Networks models.
  """
  def __init__(self,trainable):
    self.decay                  = cfg.DECAY
    self.stddev                 = cfg.STDDEV
    self.alpha                  = cfg.ALPHA
    self.WEIGHT_DECAY           = cfg.WEIGHT_DECAY
    self.BATCH_NORM_EPSILON     = cfg.BATCH_NORM_EPSILON
    # pass parameter from other place
    self.testing                = False if trainable else True
    self.quant                  = cfg.QUANT 
    self.model_params           = []

  def bn_fusion(self,input_data,layer_name,filters_shape,strides,padding,trainable,first_layer=False):
    '''
    fusion batch normlization into convolution layer
    '''
    with tf.variable_scope(layer_name) as scope:
      channels = input_data.get_shape()[3]
      kernel_val = tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32)
      mean_val   = tf.constant_initializer(0.0)
      var_val    = tf.constant_initializer(1.0)
      gamma_val  = tf.constant_initializer(1.0)
      beta_val   = tf.constant_initializer(0)
      scale_init = tf.constant_initializer(0.5)

      weight = _variable_with_weight_decay(
          'weight', shape=filters_shape,
          wd=self.WEIGHT_DECAY, initializer=kernel_val, trainable=trainable)
      if first_layer:
        weight = tf.div(weight,numpy2tensor(255.))
      delta = tf.nn.conv2d(128.*tf.ones_like(input_data),weight,strides, padding=padding)

      conv = tf.nn.conv2d(input_data, weight, strides, padding=padding,name='convolution')
      parameter_bn_shape = conv.get_shape()[-1:]
      gamma = _variable_on_device('gamma', parameter_bn_shape, gamma_val,
                                  trainable=trainable)
      beta  = _variable_on_device('beta', parameter_bn_shape, beta_val,
                                  trainable=trainable)
      moving_mean  = _variable_on_device('moving_mean', parameter_bn_shape, mean_val, trainable=False)
      moving_variance   = _variable_on_device('moving_variance', parameter_bn_shape, var_val, trainable=False)
      
      #fold weight and bias
      mean, variance = tf.nn.moments(conv, list(range(len(conv.get_shape()) - 1)))
      update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, self.decay, zero_debias=False)
      update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, self.decay, zero_debias=False)
      def mean_var_with_update():
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
          return tf.identity(mean), tf.identity(variance)
      if self.testing:
        mean_1, var = moving_mean, moving_variance
      else:
        mean_1, var = mean_var_with_update()

      w_fold = _w_fold(weight, gamma, var, self.BATCH_NORM_EPSILON)
      bias_fold = _bias_fold(beta, gamma, mean_1, var, self.BATCH_NORM_EPSILON)
      if first_layer:
        bias_fold = tf.add(bias_fold,delta[0,0,0,:])
        # fake_bias = tf.nn.bias_add(delta,bias_fold)
        # bias_fold = fake_bias[0,0,0,:]

    return w_fold,bias_fold

  def quant_model_int(self,input_data,weight,bias,strides,padding,activate,act_fun='relu'):
    '''
    quantify a float tensor to a int tensor of given range with fixed bits
    '''
    quant_input,clip_x = quant_tensor(input_data)
    quant_weight,clip_w = quant_tensor(weight)
    clip = clip_x+clip_w
    fix_ratio = tf.pow(numpy2tensor(2),clip)
    quant_bias = quant_tensor(bias,bits=32,fix=True,scale=fix_ratio)

    quant_origin = tf.nn.conv2d(quant_input, quant_weight, strides, padding=padding, name='convolution')
    quant_conv =tf.nn.bias_add(quant_origin, quant_bias)

    if activate:
      if act_fun=='leakyrelu':
        quant_conv = handle_round(tf.nn.leaky_relu(quant_conv,alpha = self.alpha))
      elif act_fun=='relu':
        quant_conv = handle_round(tf.nn.relu(quant_conv))

    conv = tf.div(quant_conv,fix_ratio)

    tf.add_to_collection('origins',quant_origin)
    tf.add_to_collection('activations',quant_conv)
    tf.add_to_collection('weights',quant_weight)
    tf.add_to_collection('biases',quant_bias)
    tf.add_to_collection('clips_x',clip_x)
    tf.add_to_collection('clips_w',clip_w)

    return conv

  def quant_model_float(self,input_data,weight,bias,strides,padding,activate,act_fun='relu'):
    '''
    quantify a float tensor to a quant float tensor of given range with fixed bits
    '''
    quant_input,clip_x = quant_tensor(input_data,return_type='float')
    quant_weight,clip_w = quant_tensor(weight,return_type='float')
    clip = clip_x+clip_w
    fix_ratio = tf.pow(numpy2tensor(2),clip)
    quant_bias = quant_tensor(bias,bits=32,fix=True,scale=fix_ratio,return_type='float')

    quant_origin = tf.nn.conv2d(quant_input, quant_weight, strides, padding=padding, name='convolution')
    quant_conv = tf.nn.bias_add(quant_origin,quant_bias)

    if activate:
      if act_fun=='leakyrelu':
        quant_conv = tf.nn.leaky_relu(quant_conv,alpha = self.alpha)
      elif act_fun=='relu':
        quant_conv = tf.nn.relu(quant_conv)

    tf.add_to_collection('inputs',quant_input)
    tf.add_to_collection('origins',quant_origin)
    tf.add_to_collection('activations',quant_conv)
    tf.add_to_collection('weights',quant_weight)
    tf.add_to_collection('biases',quant_bias)
    tf.add_to_collection('clips_x',clip_x)
    tf.add_to_collection('clips_w',clip_w)

    return quant_conv

  def yolo_first_layer(
    self, input_data, layer_name, filters_shape, trainable,downsample=False, activate=True,bn=True):
    '''
    the special operation of first layer
    init   : init_input_image_data/255.
    present: (init_input_image_data-128)/255.
    fusion the extra term into weight and bias
    '''

    if downsample:
      input_shape = input_data.get_shape()
      pad_h = 1 if input_shape[1]%2==0 else 0
      pad_w = 1 if input_shape[2]%2==0 else 0
      paddings = tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
      input_data = tf.pad(input_data, paddings, 'CONSTANT')
      strides = [1,2,2,1]
      padding = 'VALID'
      # print(layer_name,input_shape,padding)
    else:
      strides = [1,1,1,1]
      padding = "SAME"

    # input_data = tf.div(tf.add(input_data,128.*tf.ones_like(input_data)),numpy2tensor(255))
    weight, bias = self.bn_fusion(input_data = input_data,filters_shape=filters_shape,layer_name=layer_name, strides = strides,padding = padding,trainable = trainable,first_layer=True)
    # weight = tf.div(weight,numpy2tensor(255))
    # mask = tf.multiply(numpy2tensor(128),tf.ones_like(input_data))
    # delta = tf.div(tf.nn.conv2d(mask,weight,strides,padding=padding),numpy2tensor(255))
    
    # bias +=delta[0,0,0,:]

    if self.quant:
      conv = self.quant_model_float(input_data,weight,bias,strides,padding,activate,act_fun='leakyrelu')

    else:
      conv = tf.nn.conv2d(input_data, weight, strides, padding=padding, name='convolution')
      conv = tf.nn.bias_add(conv, bias)
      if activate:
        conv = tf.nn.leaky_relu(conv,alpha = self.alpha)

    return conv

  def conv_bn_leakyrelu_layer(
      self, input_data, layer_name, filters_shape, trainable,downsample=False, activate=True,bn=True):
    '''
    convolution + batch normlization + leakyrelu(option) 
    '''

    if downsample:
      input_shape = input_data.get_shape()
      pad_h = 1 if input_shape[1]%2==0 else 0
      pad_w = 1 if input_shape[2]%2==0 else 0
      paddings = tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
      input_data = tf.pad(input_data, paddings, 'CONSTANT')
      strides = [1,2,2,1]
      padding = 'VALID'
      # print(layer_name,input_shape,padding)
    else:
      strides = [1,1,1,1]
      padding = "SAME"

    weight, bias = self.bn_fusion(input_data = input_data,filters_shape=filters_shape,layer_name=layer_name, strides = strides,padding = padding,trainable = trainable)

    if self.quant:
      conv = self.quant_model_float(input_data,weight,bias,strides,padding,activate,act_fun='leakyrelu')

    else:
      conv = tf.nn.conv2d(input_data, weight, strides, padding=padding, name='convolution')
      conv =tf.nn.bias_add(conv, bias)
      if activate:
        conv = tf.nn.leaky_relu(conv,alpha = self.alpha)

    return conv
    

  def conv_layer(
      self, input_data, layer_name, filters_shape, trainable,downsample=False, activate=True,bn=True):
    '''
    convolution + batch normlization + relu(optional) 
    '''

    if downsample:
      pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
      paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
      input_data = tf.pad(input_data, paddings, 'CONSTANT')
      strides = [1,2,2,1]
      padding = 'VALID'
    else:
      strides = [1,1,1,1]
      padding = "SAME"

    with tf.variable_scope(layer_name) as scope:
      weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=trainable,
                                   shape=filters_shape, initializer=tf.random_normal_initializer(stddev=self.stddev))
      bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=trainable,
                                     dtype=tf.float32, initializer=tf.constant_initializer(0))
    if self.quant:
      conv = self.quant_model_float(input_data,weight,bias,strides,padding,activate=False)

    else:
      conv = tf.nn.conv2d(input_data, weight, strides, padding=padding, name='convolution')
      conv =tf.nn.bias_add(conv, bias)
      if activate:
        conv = tf.nn.relu(conv)
  
    return conv

  def residual_unit_leakyrelu(
      self, input_data, name, input_channel, filter1_num, filter2_num, trainable):
    '''
    residual unit  input_data-->1x1 conv_bn_leakyrelu-->3x3 conv_bn_leakyrelu-->
                        |-------------------------------------------------------|-->add-->output_data
    '''

    short_cut = input_data

    conv1 = self.conv_bn_leakyrelu_layer(input_data,name+'_conv1_1s1',[1,1,input_channel,filter1_num],trainable = trainable)
    conv2 = self.conv_bn_leakyrelu_layer(conv1,name+'_conv2_3s1',[3,3,filter1_num,filter2_num],trainable = trainable)

    residual_output = short_cut+conv2

    tf.add_to_collection('activations',residual_output)

    return residual_output

  def depthwise_convolution():
    pass

  

  def upsample(self,input_data):
    '''
    upsample: resize_nearest_neighbor
    '''

    input_shape = tf.shape(input_data)
    output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    return output


  def pooling_layer(
      self, input_data, layer_name, ksize, stride,padding='SAME'):
    with tf.variable_scope(layer_name) as scope:
      layer_out =  tf.nn.max_pool(input_data,
                            ksize=[1,ksize,ksize,1],
                            strides=[1, stride, stride, 1],
                            padding=padding)
      return layer_out


  def full_connect_layer(
    self, input_data, layer_name, filters_num , activation):
    with tf.variable_scope(layer_name) as scope:
      flatten_data = tf.layers.flatten(input_data)
      layer_out = tf.layers.dense(flatten_data,
                            units=filters_num,
                            activation=activation)
      return layer_out

  def global_average_pooling(self, layer_name, inputs, stride, padding = 'VALID'):
    batch_num, height, width, channels = inputs.get_shape().as_list() 
    with tf.variable_scope(layer_name) as scope:
      out = tf.nn.avg_pool(inputs, 
                            ksize = [1, height, width, 1],
                            strides = [1,stride, stride,1],
                            padding = padding)
      out = tf.reduce_mean(out, [1,2])
      return out
