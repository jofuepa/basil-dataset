import numpy as np
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import random
import os


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

def noise(image):
	image = np.array(image)
	h = image.shape[0]
	w = image.shape[1]
	noisy_image = np.random.random_sample(size=(h,w,3))
	mask = (image == [0.,0.,0.]).all(axis=2)
	idx = (mask == 1)
	image[ idx ] = noisy_image[idx]
	
	return image

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

		