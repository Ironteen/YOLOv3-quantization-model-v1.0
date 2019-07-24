import numpy as np
import os
import cv2
import tensorflow as tf 
from tensorflow.python import pywrap_tensorflow
from core.save_params import save_quant_param
from core.YOLOv3 import yolov3_model
from dataset.ImageNet import load_ImageNet

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

img_dir = './data/plusAI/test.jpg'
checkpoint_path = "/data/cag-stu-tuzhijun/quantization/log/checkpoint_transfer/"
model_file = tf.train.latest_checkpoint(checkpoint_path)
epsilon = 1e-5

def handle_round(data):
	data_2x = data*2
	data_2x_floor = np.floor(data_2x)
	if data_2x==data_2x_floor:
		return int(np.ceil(data))
	else:
		return int(np.round(data))

def quant_param(data):
	max_value = np.max(np.abs(data))
	init_ratio = 127.0/max_value
	max_N = np.floor(np.log2(init_ratio))
	new_ratio = 2**max_N
	data = data*new_ratio
	data = [handle_round(x) for x in data]

	return int(max_N),data

def save_param(file_name,data,bits=8):
	save_path = './quant_param'
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	bias = 2**bits
	save_file = open(os.path.join(save_path,file_name+'.txt'),'w+')
	max_N,int_data = quant_param(np.ravel(data))
	hex_data = [hex(x) if x>0 else hex(x+bias) for x in int_data]
	for idx in hex_data:
		save_file.write(str(idx)+'\n')
	print('The length of %s is %d,saved successful'%(file_name,len(hex_data)))
	save_file.close()
	return max_N

	# Read data from checkpoint file
def read_ckpt():
	reader = pywrap_tensorflow.NewCheckpointReader(model_file)
	var_to_shape_map = reader.get_variable_to_shape_map()
	count = 0
	for key in var_to_shape_map:
		if 'p' in key:
			count +=1
			temp = reader.get_tensor(key)
			print(count,key,np.shape(temp))

def cv_read(img_path):
	image_data = cv2.imread(img_path)
	image_data = np.ravel(image_data)
	print('image_data:',image_data[0:10])


def test_module():
	''' test tf.tile function
	width = 64
	height = 48
	# temp = tf.range(width, dtype=tf.int32)[:, tf.newaxis]
	y = tf.tile(tf.range(width, dtype=tf.int32)[tf.newaxis, :], [height, 1])
	x = tf.tile(tf.range(width, dtype=tf.int32)[tf.newaxis, :], [height, 1])

	xy_grid_1 = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
	temp = xy_grid_1[0,0,:]
	xy_grid_2 = tf.tile(xy_grid_1[tf.newaxis, :, :, tf.newaxis, :], [1, 1, 1, 9, 1])
	xy_grid_3 = tf.cast(xy_grid_2, tf.float32)

	with tf.Session() as sess:
		temp = sess.run(temp)
		print(temp,np.shape(temp)) 
	'''
	####################################################################################
	''' test the clip module
	save_model = save_quant_param()
	input_data = [-130,-128,-127,-100,0,100,127,128,130]
	output_data = save_model.cabs(input_data,bits=8)
	for idx in range(len(input_data)):
		print(idx+1,input_data[idx],output_data[idx])
	'''
	###################################################################################
	a = np.array(['a','b','v','n','m'])
	# a = np.array([1,2,3,4,5,6])
	np.savetxt('./save.txt',a,fmt='%s')
	x_data = open('./save.txt','r').readlines()
	print(x_data)

if __name__=='__main__':
	# read_ckpt()
	test_module()




