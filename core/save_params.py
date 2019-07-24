import os
import time
import numpy as np 
import tensorflow as tf
from core.config import cfg 


class save_quant_param(object):
	def __init__(self):
		self.save_path             = cfg.PARAM_SAVE_PATH
		self.input_path            = os.path.join(self.save_path,'input')
		self.origin_path           = os.path.join(self.save_path,'origin')
		self.activations_path      = os.path.join(self.save_path,'activations')
		self.weights_path          = os.path.join(self.save_path,'weights')
		self.biases_path           = os.path.join(self.save_path,'biases')

		if not os.path.exists(self.save_path):
			os.mkdir(self.save_path)
		if not os.path.exists(self.input_path):
			os.mkdir(self.input_path)
		if not os.path.exists(self.origin_path):
			os.mkdir(self.origin_path)
		if not os.path.exists(self.activations_path):
			os.mkdir(self.activations_path)
		if not os.path.exists(self.weights_path):
			os.mkdir(self.weights_path)
		if not os.path.exists(self.biases_path):
			os.mkdir(self.biases_path)

		self.clip_key_path         = os.path.join(self.save_path,'clip_key.txt')
		self.clip_value_path       = os.path.join(self.save_path,'clip_value.txt')

	def cabs(self,data,bits=8):
		lower_limit = -2**(bits-1)
		upper_limit = 2**(bits-1)-1
		data = np.clip(data,lower_limit,upper_limit)

		return data

	def handle_round(self,data):
		data_2x = data*2
		data_2x_floor = np.floor(data_2x)
		if data_2x==data_2x_floor:
			return int(np.ceil(data))
		else:
			return int(np.round(data))

	def quant_numpy(self,data,clip):
		scale = 2**int(clip)
		data_shape = np.shape(data)
		data_list = np.ravel(data)
		int_data = [self.handle_round(x*scale) for x in data_list]

		return np.reshape(int_data,data_shape)

	def calculate_clip(self,data,bits=8):
		limits = 2**(bits-1)-1
		max_value = np.max(np.abs(data))
		scale = float(limits)/float(max_value)
		# method 1
		clip = np.round(np.log2(scale))
		# method 2
		# clip = np.floor(np.log2(scale))
		return int(clip)


	def save_params(self,cls,file_name,data,bits=8):
		bias = 2**bits
		if cls=='input':
			save_dir = os.path.join(self.input_path,file_name+'.txt')
		elif cls=='add' or cls=='fm':
			save_dir = os.path.join(self.activations_path,file_name+'.txt')
		elif cls=='org':
			save_dir = os.path.join(self.origin_path,file_name+'.txt')
		elif cls=='weight':
			save_dir = os.path.join(self.weights_path,file_name+'.txt')
		elif cls=='bias':
			save_dir = os.path.join(self.biases_path,file_name+'.txt')
		# save_file = open(save_dir,'w')
		data_list = np.ravel(data)
		int_data = [int(float(x)) for x in data_list]
		int_data = self.cabs(int_data,bits=bits)
		hex_data = [hex(x) if x>0 else hex(x+bias) for x in int_data]
		hex_data = [str(x).strip('L') for x in hex_data]
		np.savetxt(save_dir,hex_data,fmt='%s')

	def save_clip(self,clip_info):
		clip_key = [x[0] for x in clip_info]
		clip_value = [x[1] for x in clip_info]
		np.savetxt(self.clip_key_path,clip_key,fmt='%s')
		np.savetxt(self.clip_value_path,clip_value,fmt='%d')

	def save_activation(self,layer_name,activation,clip_next_x):
		activation = self.quant_numpy(activation,clip_next_x)
		self.save_params('fm',layer_name,activation,bits=8)

def data_save(layers_name,inputs,origins,activations,weights,biases,clips_x,clips_w,method=1):
	'''
	save weight,bias and feature map of each layer
	'''
	branch_up_list = ['blk4_res8_conv2_3s1','blk5_res8_conv2_3s1','blkb_conv5_1s1','blkm_conv6_1s1']
	branch_to_list = ['blks_conv2_1s1','blkm_conv2_1s1','blkm_conv1_1s1uc','blks_conv1_1s1uc']
	branch_dict = dict(zip(branch_up_list,branch_to_list))

	save_model = save_quant_param()
	layer_num = len(layers_name)
	clips_x_dict = dict()
	clips_w_dict = dict()
	param_count = 0
	for idx in range(layer_num):
		layer_name = layers_name[idx].strip('\r\n')
		if 'add' not in layer_name:
			clips_x_dict[layer_name] = clips_x[param_count]
			clips_w_dict[layer_name] = clips_w[param_count]
			param_count +=1
	# save all the data
	clip_info = []
	param_count = 0
	for idx in range(layer_num):
		start_time = time.time()
		layer_name = layers_name[idx].strip('\r\n')
		print('%02d  : '%(idx+1)+'%s'%layer_name+' is processing......')
		activation = activations[idx]
		if 'add' in layer_name:
			next_layer_name = layers_name[idx+1].strip('\r\n')
			next_clip_x = clips_x_dict[next_layer_name]
			activation = save_model.quant_numpy(activation,next_clip_x)
			save_model.save_params('fm',layer_name,activation,bits=8)
			# save_model.save_clip(layer_name,0)
		else:
			clip_x = clips_x_dict[layer_name]
			clip_w = clips_w_dict[layer_name]

			input_data =  save_model.quant_numpy(inputs[param_count],clip_x)
			origin = save_model.quant_numpy(origins[param_count],clip_x+clip_w)
			weight = save_model.quant_numpy(weights[param_count],clip_w)
			bias = save_model.quant_numpy(biases[param_count],clip_x+clip_w)

			save_model.save_params('input',layer_name,input_data,bits=8)
			save_model.save_params('org',layer_name,origin,bits=32)
			save_model.save_params('weight',layer_name,weight,bits=8)
			save_model.save_params('bias',layer_name,bias,bits=32)
			# calculate the clip
			# the last layer
			if 'p' in layer_name:
				clip_next_x = save_model.calculate_clip(activations[idx])
				clip = clip_x+clip_w-clip_next_x
				save_model.save_activation(layer_name,activation,clip_next_x)
				clip_info.append([layer_name,clip])
			else:
				next_layer_name = layers_name[idx+1].strip('\r\n')
				if 'add' in next_layer_name:
					next_layer_name = layers_name[idx+2].strip('\r\n')
				clip_next_x = clips_x_dict[next_layer_name]
				if layer_name in branch_dict.keys():
					clip_next_x_up = clip_next_x
					clip_next_x_down = clips_x_dict[branch_dict[layer_name]]

					clip_up = clip_x+clip_w-clip_next_x_up
					clip_down = clip_x+clip_w-clip_next_x_down

					save_model.save_activation(layer_name+'_up',activation,clip_next_x_up)
					save_model.save_activation(layer_name+'_down',activation,clip_next_x_down)
					clip_info.append([layer_name+'_up',clip_up])
					clip_info.append([layer_name+'_down',clip_down])

				elif 'res' in next_layer_name and 'conv1' in next_layer_name:
					clip_next_x_up = clip_next_x
					# skip 1x1 conv,3x3 conv,res_add,then find the forth layer
					clip_next_x_down = clips_x_dict[layers_name[idx+5].strip('\r\n')]

					clip_up = clip_x+clip_w-clip_next_x_up
					clip_down = clip_x+clip_w-clip_next_x_down

					save_model.save_activation(layer_name+'_up',activation,clip_next_x_up)
					save_model.save_activation(layer_name+'_down',activation,clip_next_x_down)
					clip_info.append([layer_name+'_up',clip_up])
					clip_info.append([layer_name+'_down',clip_down])
				elif 'res' in layer_name and 'conv2' in layer_name:
					clip_next_x = clips_x_dict[layers_name[idx+2].strip('\r\n')]
					save_model.save_activation(layer_name,activation,clip_next_x)
					clip = clip_x+clip_w-clip_next_x
					clip_info.append([layer_name,clip])
				else:
					save_model.save_activation(layer_name,activation,clip_next_x)
					clip = clip_x+clip_w-clip_next_x
					clip_info.append([layer_name,clip])

			param_count +=1
			end_time = time.time()
			duration = end_time-start_time
			minute = int(np.floor(duration/60.))
			second = int(duration-60*minute)
			print('fuck, it takes %02d:%02d to save the data of one layer,unbeleivable......'%(minute,second))
		# save the clip
		save_model.save_clip(clip_info)



