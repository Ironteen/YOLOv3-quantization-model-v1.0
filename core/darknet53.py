import os
import numpy as np
import tensorflow as tf

import core.utils as utils
from core.config import cfg 
from core.nn_skeleton import ModelSkeleton

class darknet53_model(object):
	'''
	darknet-53 structure definition
	'''
	def __init__(self,input_data,trainable):
		self.nnlib = ModelSkeleton(trainable)

		self.trainable  = trainable
		self.strides    = np.array(cfg.COCO_STRIDES)
        	self.anchors    = utils.get_anchors(cfg.COCO_ANCHORS)
        	self.classes          = utils.read_class_names(cfg.COCO_NAMES)
        	self.num_class      = len(self.classes)

		self.darknet53_output = self._darknet53(input_data)

    	def _darknet53(self,input_data):
    	# block1
		# output_data = self.nnlib.conv_bn_leakyrelu_layer(input_data,'blk1_conv1_3s1',[3,3,3,32],self.trainable)
		output_data = self.nnlib.yolo_first_layer(input_data,'blk1_conv1_3s1',[3,3,3,32],self.trainable)
		tf.add_to_collection('debug',output_data)
		# block2
		output_data = self.nnlib.conv_bn_leakyrelu_layer(output_data,'blk2_conv1_3s2',[3,3,32,64],self.trainable,downsample=True)
		for idx in range(1,2):
			output_data = self.nnlib.residual_unit_leakyrelu(output_data, 'blk2_res%d'%idx, 64, 32, 64, self.trainable)

		# block3
		output_data = self.nnlib.conv_bn_leakyrelu_layer(output_data,'blk3_conv1_3s2',[3,3,64,128],self.trainable,downsample=True)
		for idx in range(1,3):
			output_data = self.nnlib.residual_unit_leakyrelu(output_data, 'blk3_res%d'%idx, 128, 64, 128, self.trainable)

		# block4
		output_data = self.nnlib.conv_bn_leakyrelu_layer(output_data,'blk4_conv1_3s2',[3,3,128,256],self.trainable,downsample=True)
		for idx in range(1,9):
			output_data = self.nnlib.residual_unit_leakyrelu(output_data, 'blk4_res%d'%idx, 256, 128, 256, self.trainable)
		self.route1 = output_data

		# block5
		output_data = self.nnlib.conv_bn_leakyrelu_layer(output_data,'blk5_conv1_3s2',[3,3,256,512],self.trainable,downsample=True)
		for idx in range(1,9):
			output_data = self.nnlib.residual_unit_leakyrelu(output_data, 'blk5_res%d'%idx, 512, 256, 512, self.trainable)
		self.route2 = output_data

		# block6
		output_data = self.nnlib.conv_bn_leakyrelu_layer(output_data,'blk6_conv1_3s2',[3,3,512,1024],self.trainable,downsample=True)
		for idx in range(1,5):
			output_data = self.nnlib.residual_unit_leakyrelu(output_data, 'blk6_res%d'%idx, 1024, 512, 1024, self.trainable)
		self.route3 = output_data

		return output_data    	

	def imagenet_recog(self,input_data):
		'''
		add 1x1 kernel, map the channel nums to 1000
		and use global average pool get output([batch_size,1000])
		'''
		in_channel_nums = input_data.get_shape().as_list()[3]
		output_data = self.nnlib.conv_layer(input_data,'last_layer',[1,1,in_channel_nums,1000],self.trainable)
		output_data = self.nnlib.global_average_pooling('global_average', output_data, stride=1)

		return output_data

