#!/usr/bin/env python
# -*- coding:utf-8 -*-
# title            : quant_YOLO.py
# description      : quantify the YOLO v3 structure and save
#                    the parameters and activations of one image
# author           : Zhijun Tu
# email            : tzj19970116@163.com
# date             : 2017/07/16
# version          : 1.0
# notes            : 
# python version   : 2.7.12 which is also applicable to 3.5
###############################################################    
import os
import cv2
import numpy as np
from absl import app
from PIL import Image
from absl import flags
import tensorflow as tf 

from core.YOLOv3 import yolov3_model
from core.config import cfg 
import core.utils as utils
from core.save_params import data_save

FLAGS = flags.FLAGS

flags.DEFINE_string('gpu','5,8','comma seperate list of GPU to use')

# read one image and preprocess it
img_dir = './data/plusAI/test_1024x772.jpg'
# input_size = [1024,772]
input_size = [416,416]
original_image = cv2.imread(img_dir)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size[0],input_size[1]])
image_data = image_data*255.0-128

class YOLO_train(object):
	'''
	   run the YOLO v3 and get the result of one image
	'''
	def __init__(self):
		self.saved_weight             = './log/checkpoint_transfer/'
		self.num_classes              = cfg.COCO_CLASSES
		self.layers_name              = open(cfg.LAYERS_NAME,'r').readlines()
		
		self.trainable                = False 

		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

		with tf.name_scope('define_input'):
			self.input_data = tf.placeholder(dtype=tf.float32,shape=(None,input_size[0],input_size[1],3),name='input_data')
			

		self.model = yolov3_model(self.input_data,self.trainable)

		self.net_var = tf.global_variables()

		with tf.name_scope('loader_saver'):
			variables = tf.contrib.framework.get_variables_to_restore()
			variables_to_resotre = [v for v in variables if 'fold' not in v.name and 'last_layer' not in v.name]
			self.loader = tf.train.Saver(variables_to_resotre)
			self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=3)

	def test(self):
		self.sess.run(tf.global_variables_initializer())

		model_file = tf.train.latest_checkpoint(self.saved_weight)
		checkpoint = tf.train.get_checkpoint_state(self.saved_weight)
		ckpt_flag = 0
        	if checkpoint and checkpoint.model_checkpoint_path:
        		ckpt_flag = 1
            		self.loader.restore(self.sess, checkpoint.model_checkpoint_path)
            		print('=> Restore weights from: %s successfully...'%self.saved_weight)
        	else:
            		print('=> %s does not exist !!!'%self.saved_weight)
            		print('=> Now it starts to train YOLO from scratch ...')
		if self.test:
			input_data = image_data[np.newaxis,:,:,:]
			
			lbbox,mbbox,sbbox = self.sess.run([self.model.lbbox,self.model.mbbox,self.model.sbbox],feed_dict={self.input_data:input_data})
			print('lbbox:',np.shape(lbbox),np.min(lbbox),np.max(lbbox))
			print('mbbox:',np.shape(mbbox),np.min(mbbox),np.max(mbbox))
			print('sbbox:',np.shape(sbbox),np.min(sbbox),np.max(sbbox))

			pred_lbbox,pred_mbbox,pred_sbbox = self.sess.run([self.model.pred_lbbox,self.model.pred_mbbox,self.model.pred_sbbox],feed_dict={self.input_data:input_data})
			print('pred_lbbox:',np.shape(pred_lbbox),np.min(pred_lbbox),np.max(pred_lbbox))
			print('pred_mbbox:',np.shape(pred_mbbox),np.min(pred_mbbox),np.max(pred_mbbox))
			print('pred_sbbox:',np.shape(pred_sbbox),np.min(pred_sbbox),np.max(pred_sbbox))
			pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

			bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.2)
			bboxes = utils.nms(bboxes, 0.45, method='nms')
			image = utils.draw_bbox(original_image, bboxes)
			image = Image.fromarray(image)
			image.show()

		'''
		# visualize the feature map of each layer
		inputs = self.sess.run(tf.get_collection('inputs'),feed_dict={self.input_data:input_data})
		origins = self.sess.run(tf.get_collection('origins'),feed_dict={self.input_data:input_data})
		activations = self.sess.run(tf.get_collection('activations'),feed_dict={self.input_data:input_data})
		weights = self.sess.run(tf.get_collection('weights'),feed_dict={self.input_data:input_data})
		biases = self.sess.run(tf.get_collection('biases'),feed_dict={self.input_data:input_data})
		clips_x = self.sess.run(tf.get_collection('clips_x'),feed_dict={self.input_data:input_data})
		clips_w = self.sess.run(tf.get_collection('clips_w'),feed_dict={self.input_data:input_data})

		print('load the data successfully,start saving......')
		data_save(self.layers_name,inputs,origins,activations,weights,biases,clips_x,clips_w)
		'''
		
		# debug
		# debug = self.sess.run(tf.get_collection('debug'),feed_dict={self.input_data:input_data})

		# for idx,data in enumerate(debug):
		# 	data = np.ravel(data)
		# 	print('%02d'%idx,np.shape(data),np.min(data),np.max(data))
			# print('%02d'%idx,data[0:10])

def main(argv):
	del argv
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

	YOLO_train().test()


if __name__=='__main__':
	app.run(main)
	

	
