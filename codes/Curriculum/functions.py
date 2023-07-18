from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Model
from scikitplot.metrics import plot_confusion_matrix
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
import matplotlib.pylab as plt
import random
import os
import re
import numpy as np
import cv2
import config


def plot_trainingLoss(H, N, plotPath):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.title("Training Loss")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)


def plot_trainingAccuracy(H, N, plotPath):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
	"""
	2D gaussian mask - should give the same result as MATLAB's
	fspecial('gaussian',[shape],[sigma])
	"""
	m,n = [(ss-1.)/2. for ss in shape]
	y,x = np.ogrid[-m:m+1,-n:n+1]
	h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
	h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
	sumh = h.sum()
	if sumh != 0:
		h /= sumh
	return h

def depthwise_layer_factory():
	return DepthwiseConv2D((3,3), use_bias=False, padding='same')

def average_layer_factory():
	return GlobalAveragePooling2D()

def densefinal_layer_factory():
	return Dense(len(config.CLASSES), activation="softmax") 


def insert_layer_nonseq2(model, layer_regex="_conv", insert_layer_name='', position='after'):
	network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
	# Set the input layers of each layer
	for layer in model.layers:
		for node in layer._outbound_nodes:
			layer_name = node.outbound_layer.name
			if layer_name not in network_dict['input_layers_of']:
				network_dict['input_layers_of'].update({layer_name: [layer.name]})
			else:
				network_dict['input_layers_of'][layer_name].append(layer.name)
	# Set the output tensor of the input layer
	network_dict['new_output_tensor_of'].update({model.layers[0].name: model.input})

	# Iterate over all layers after the input
	model_outputs = []
	num = 0
	for layer in model.layers[1:]:
		# Determine input tensors
		layer_input = [network_dict['new_output_tensor_of'][layer_aux] for layer_aux in network_dict['input_layers_of'][layer.name]]
		if len(layer_input) == 1:
			layer_input = layer_input[0]

		# Insert layer if name matches the regular expression
		if re.search(layer_regex, layer.name):
			num = num + 1
			if position == 'replace':
				x = layer_input
			elif position == 'after':
				x = layer(layer_input)
			elif position == 'before':
				pass
			else:
				raise ValueError('position must be: before, after or replace')

			if layer_regex == "_conv":
				new_layer = depthwise_layer_factory()
			if layer_regex == "post_relu" or layer_regex == "mixed10" :
				new_layer = average_layer_factory()
			if layer_regex == "_average_pooling2d":
				new_layer = densefinal_layer_factory()
			
			if insert_layer_name:
				new_layer._name = insert_layer_name+str(num)
			else:
				new_layer._name = '{}'.format(new_layer.name)
		
			x = new_layer(x)
			print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name, layer.name, position))
			if position == 'before':
				x = layer(x)
		else:
			x = layer(layer_input)
		
		# Set new output tensor (the original one, or the one of the inserted layer)
		network_dict['new_output_tensor_of'].update({layer.name: x})

		# Save tensor in output list if it is output in initial model
		if layer.name in model.output_names:
			model_outputs.append(x)
	
	return Model(inputs=model.inputs, outputs=model_outputs)



def assign_gaussian_weights(model, kernel_size, sigma):
	kernel_weights = matlab_style_gauss2D(kernel_size,sigma) 
	in_channels = [64,128,256,512,1024,2048] 
  
	kernel_weights64 = np.expand_dims(kernel_weights, axis=-1) 
	kernel_weights64 = np.repeat(kernel_weights64, in_channels[0], axis=-1) 
	kernel_weights64 = np.expand_dims(kernel_weights64, axis=-1)  

	kernel_weights128 = np.expand_dims(kernel_weights, axis=-1)
	kernel_weights128 = np.repeat(kernel_weights128, in_channels[1], axis=-1)
	kernel_weights128 = np.expand_dims(kernel_weights128, axis=-1) 

	kernel_weights256 = np.expand_dims(kernel_weights, axis=-1)
	kernel_weights256 = np.repeat(kernel_weights256, in_channels[2], axis=-1)
	kernel_weights256 = np.expand_dims(kernel_weights256, axis=-1) 

	kernel_weights512 = np.expand_dims(kernel_weights, axis=-1)
	kernel_weights512 = np.repeat(kernel_weights512, in_channels[3], axis=-1)
	kernel_weights512 = np.expand_dims(kernel_weights512, axis=-1) 

	kernel_weights1024 = np.expand_dims(kernel_weights, axis=-1)
	kernel_weights1024 = np.repeat(kernel_weights1024, in_channels[4], axis=-1)
	kernel_weights1024 = np.expand_dims(kernel_weights1024, axis=-1) 

	kernel_weights2048 = np.expand_dims(kernel_weights, axis=-1)
	kernel_weights2048 = np.repeat(kernel_weights2048, in_channels[5], axis=-1)
	kernel_weights2048 = np.expand_dims(kernel_weights2048, axis=-1) 

	dict ={
		64 : kernel_weights64,
		128 : kernel_weights128,
		256 : kernel_weights256,
		512 : kernel_weights512,
		1024 : kernel_weights1024,
		2048 : kernel_weights2048
	}


	print('Assigning weights to filter layers...')
	i = 0
	for layer in model.layers:
		if isinstance(layer, DepthwiseConv2D) : 
			output_shape= layer.output_shape
			output_channels = output_shape[-1]
			model.layers[i].set_weights([dict[output_channels]])	
		i = i+1

	return model


class PerformanceVisualizationCallback(Callback):
	def __init__(self, model, validation_data, image_dir):
		super().__init__()
		self.model = model
		self.validation_data = validation_data

		os.makedirs(image_dir, exist_ok=True)
		self.image_dir = image_dir

	def on_epoch_end(self, epoch, logs={}):
		y_pred = np.asarray(self.model.predict(self.validation_data))
		y_true = self.validation_data.classes
		y_pred_class = np.argmax(y_pred, axis=1)
		fig, ax = plt.subplots()
		plot_confusion_matrix(y_true, y_pred_class, ax=ax)
		ax.set_title('')
		ax.set_xlabel('Predicted class')
		ax.set_ylabel('True class')
		ax.xaxis.set_ticklabels(['I', 'II', 'III', 'IV'])
		ax.yaxis.set_ticklabels(['I', 'II', 'III', 'IV'])
		fig.savefig(os.path.join(self.image_dir,
								 f'confusion_matrix_epoch_{epoch}'), dpi=300)

def model_style2(filepath_model, filepath_weights, kernel_size, sigma):	
	print(filepath_model)
	full_model = load_model(filepath_model)
	inputs = Input(shape =(config.SIZE_IMAGE, config.SIZE_IMAGE, config.CHANNELS))
	full_model= assign_gaussian_weights(full_model, kernel_size, sigma)

	for layer in full_model.layers:
		if isinstance(layer, (DepthwiseConv2D) ):
			layer.trainable = False
			print(layer._name, layer.trainable)
		else:
			layer.trainable = True
		
	full_model(inputs,training = False)
	return full_model

def cutout(image):
	image = np.array(image)
	p = 1.0
	opciones = [0, 1]
	distribution = [1.0-p, p]
	opc = random.choices(opciones, distribution)
	if opc[0] == 0:
		return noise(image)
	else:
		length = 112
		h = image.shape[0]
		w = image.shape[1]
		y = np.random.randint(h)
		x = np.random.randint(w)
		y1 = np.clip(y - length // 2, 0, h)
		y2 = np.clip(y + length // 2, 0, h)
		x1 = np.clip(x - length // 2, 0, w)
		x2 = np.clip(x + length // 2, 0, w)
		image[y1:y2, x1:x2] = [0, 0, 0]
		return noise(image)

def only_plant(image):
	image = np.array(image)
	h = image.shape[0]
	w = image.shape[1]
	noisy_image = np.zeros((h, w, 3), dtype=np.int)
	noisy_image[:, :] = [0, 0, 0]
	image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	image_threshold = cv2.inRange(image_hsv, (60, 0.2, 50), (140, 1, 255))
	kernel = np.ones((3, 3), np.uint8)
	image_morph = cv2.morphologyEx(
		image_threshold, cv2.MORPH_OPEN, kernel, iterations=1)
	mask_inv = cv2.bitwise_not(image_morph)
	img1_bg = cv2.bitwise_and(image, image, mask=image_morph)
	img2_fg = cv2.bitwise_and(noisy_image, noisy_image, mask=mask_inv)
	dst = cv2.add(img1_bg, img2_fg, dtype=cv2.CV_32FC3)
	image[0:h, 0:w] = dst
	image = cutout(image)
	return image

def noise(image):
	image = np.array(image)
	h = image.shape[0]
	w = image.shape[1]
	noisy_image = np.random.random_sample(size=(h,w,3))
	mask = (image == [0.,0.,0.]).all(axis=2)
	idx = (mask == 1)
	image[ idx ] = noisy_image[idx]	
	return image

def noise_non_normalized(image):
	h = image.shape[0]
	w = image.shape[1]
	noisy_image = np.random.randint(255, size=(h,w,3))
	mask = (image == [0.,0.,0.]).all(axis=2)
	idx = (mask == 255.0)
	image[ idx ] = noisy_image[idx]
	return image


