import os
import numpy as np
import tensorflow as tf

import core.utils as utils
from core.config import cfg 
from core.darknet53 import darknet53_model
from core.nn_skeleton import ModelSkeleton

class yolov3_model(object):
	'''
	initial YOLO v3 structure definetion
	'''
	def __init__(self,input_data,trainable):
		self.nnlib = ModelSkeleton(trainable)

		self.trainable  = trainable
		self.strides    = np.array(cfg.COCO_STRIDES)
        	self.anchors    = utils.get_anchors(cfg.COCO_ANCHORS)
        	self.classes          = utils.read_class_names(cfg.COCO_NAMES)
        	self.num_class      = len(self.classes)

		self.darknet53_model = darknet53_model(input_data,trainable)
		self.route1,self.route2,self.route3 = self.darknet53_model.route1,self.darknet53_model.route2,self.darknet53_model.route3
		self.lbbox,self.mbbox,self.sbbox = self._backbone()

		self.pred_sbbox = self.decode(self.sbbox, self.anchors[0], self.strides[0])
        	self.pred_mbbox = self.decode(self.mbbox, self.anchors[1], self.strides[1])
        	self.pred_lbbox = self.decode(self.lbbox, self.anchors[2], self.strides[2])
  
	def _backbone(self):
		# block_big_13x13
		blkb_conv1 = self.nnlib.conv_bn_leakyrelu_layer(self.route3,'blkb_conv1_1s1',[1,1,1024,512],self.trainable)
		blkb_conv2 = self.nnlib.conv_bn_leakyrelu_layer(blkb_conv1,'blkb_conv2_3s1',[3,3,512,1024],self.trainable)
		blkb_conv3 = self.nnlib.conv_bn_leakyrelu_layer(blkb_conv2,'blkb_conv3_1s1',[1,1,1024,512],self.trainable)
		blkb_conv4 = self.nnlib.conv_bn_leakyrelu_layer(blkb_conv3,'blkb_conv4_3s1',[3,3,512,1024],self.trainable)
		blkb_branch = self.nnlib.conv_bn_leakyrelu_layer(blkb_conv4,'blkb_conv5_1s1',[1,1,1024,512],self.trainable)

		blkb_conv6 = self.nnlib.conv_bn_leakyrelu_layer(blkb_branch,'blkb_conv6_3s1',[3,3,512,1024],self.trainable)
		lbbox = self.nnlib.conv_layer(blkb_conv6,'blkb_conv7_1s1p',[1,1,1024,255],self.trainable)

		# block_middle_26x26
		blkm_conv1 = self.nnlib.conv_bn_leakyrelu_layer(blkb_branch,'blkm_conv1_1s1uc',[1,1,512,256],self.trainable)
		upsample_data = self.nnlib.upsample(blkm_conv1)
		blkm_conv1 = tf.concat([self.route2,upsample_data],axis=-1)

		blkm_conv2 = self.nnlib.conv_bn_leakyrelu_layer(blkm_conv1,'blkm_conv2_1s1',[1,1,768,256],self.trainable)
		blkm_conv3 = self.nnlib.conv_bn_leakyrelu_layer(blkm_conv2,'blkm_conv3_3s1',[3,3,256,512],self.trainable)
		blkm_conv4 = self.nnlib.conv_bn_leakyrelu_layer(blkm_conv3,'blkm_conv4_1s1',[1,1,512,256],self.trainable)
		blkm_conv5 = self.nnlib.conv_bn_leakyrelu_layer(blkm_conv4,'blkm_conv5_3s1',[3,3,256,512],self.trainable)
		blkm_branch = self.nnlib.conv_bn_leakyrelu_layer(blkm_conv5,'blkm_conv6_1s1',[1,1,512,256],self.trainable)

		blkm_conv7 = self.nnlib.conv_bn_leakyrelu_layer(blkm_branch,'blkm_conv7_3s1',[3,3,256,512],self.trainable)
		mbbox = self.nnlib.conv_layer(blkm_conv7,'blkm_conv8_1s1p',[1,1,512,255],self.trainable)

		# block_small_52x52
		blks_conv1 = self.nnlib.conv_bn_leakyrelu_layer(blkm_branch,'blks_conv1_1s1uc',[1,1,256,128],self.trainable)
		upsample_data = self.nnlib.upsample(blks_conv1)
		blks_conv1 = tf.concat([self.route1,upsample_data],axis=-1)

		blks_conv2 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv1,'blks_conv2_1s1',[1,1,384,128],self.trainable)
		blks_conv3 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv2,'blks_conv3_3s1',[3,3,128,256],self.trainable)
		blks_conv4 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv3,'blks_conv4_1s1',[1,1,256,128],self.trainable)
		blks_conv5 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv4,'blks_conv5_3s1',[3,3,128,256],self.trainable)
		blks_conv6 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv5,'blks_conv6_1s1',[1,1,256,128],self.trainable)

		blks_conv7 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv6,'blks_conv7_3s1',[3,3,128,256],self.trainable)
		sbbox = self.nnlib.conv_layer(blks_conv7,'blks_conv8_1s1p',[1,1,256,255],self.trainable)

		return lbbox,mbbox,sbbox

	def decode(self, conv_output, anchors, stride):
		"""
		return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
		       contains (x, y, w, h, score, probability)
		"""
		conv_shape       = tf.shape(conv_output)
		batch_size       = conv_shape[0]
		width            = conv_shape[1]
		height           = conv_shape[2] 
 		anchor_per_scale = len(anchors)

		conv_output = tf.reshape(conv_output, (batch_size, width, height, anchor_per_scale, 5 + self.num_class))

		conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
		conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
		conv_raw_conf = conv_output[:, :, :, :, 4:5]
		conv_raw_prob = conv_output[:, :, :, :, 5: ]

		y = tf.tile(tf.range(width, dtype=tf.int32)[:, tf.newaxis], [1, height])
		x = tf.tile(tf.range(height, dtype=tf.int32)[tf.newaxis, :], [width, 1])

		xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
		xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
		xy_grid = tf.cast(xy_grid, tf.float32)

		pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
		pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
		pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

		pred_conf = tf.sigmoid(conv_raw_conf)
		pred_prob = tf.sigmoid(conv_raw_prob)

		return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

