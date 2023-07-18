from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
#Linux
from tensorflow.keras.optimizers import Adam
#Mac OS
#from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, CategoricalAccuracy, Precision, Recall
from imutils import paths
import config 
import functions
import os
	
trainPath = os.path.sep.join([config.BASE_PATH, config.TRAIN])
valPath = os.path.sep.join([config.BASE_PATH, config.VAL])
testPath = os.path.sep.join([config.BASE_PATH, config.VAL])
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))


trainAug = ImageDataGenerator(
	rescale = 1./255, 
	rotation_range = config.ROTATION, 
	horizontal_flip = config.HORIZONTAL_FLIP, 
	brightness_range = config.BRIGHTNESS_RANGE, 
	preprocessing_function = functions.cutout,
	fill_mode = config.FILL_MODE
)

valAug = ImageDataGenerator(rescale = 1./255)
testAug = ImageDataGenerator(rescale = 1./255)

trainGen = trainAug.flow_from_directory(
	trainPath, 
	class_mode="categorical", 
	target_size=(config.SIZE_IMAGE, config.SIZE_IMAGE), 
	color_mode="rgb",
	shuffle=True,
	batch_size=config.BATCH_SIZE) 

valGen = valAug.flow_from_directory(
	valPath,
	class_mode="categorical",
	target_size=(config.SIZE_IMAGE, config.SIZE_IMAGE), 
	color_mode="rgb",
	shuffle=True,
	batch_size=config.BATCH_SIZE) 

testGen = testAug.flow_from_directory(
    valPath,
    class_mode="categorical",
    target_size=(config.SIZE_IMAGE, config.SIZE_IMAGE),
    color_mode="rgb",
    shuffle=False,
    batch_size=config.BATCH_SIZE)



METRICS = [TruePositives(name='tp'), FalsePositives(name='fp'), TrueNegatives(name='tn'), FalseNegatives(name='fn'), 
			CategoricalAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall')] 
earlystopper=EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
reducel=ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, factor=0.1, min_lr=1e-10)


kernel_size = (3,3)
f=[1.0, 0.8, 0.6, 0.4, 0.2]

filepath_weights = ''
filepath_model = ''
sigma = f[0]
final_sigma = f[-1]

lr = 0.0
for i in range(0,len(f)):
	sigma = f[i]
	if sigma == 1.0:
		model = functions.model_style2(config.LOADMODEl_INI, config.LOADWEIGHT_INI, kernel_size, sigma)
	else:
		model = functions.model_style2(filepath_model, filepath_weights, kernel_size, sigma)
	
	filepath_weights = os.path.sep.join([config.OUTPUT_PATH, 'weights_sigma' + str(sigma) +'.h5'])
	filepath_model = os.path.sep.join([config.OUTPUT_PATH, 'model_sigma' + str(sigma) +'.h5'])

	performance_cbk = functions.PerformanceVisualizationCallback(model=model,validation_data=testGen,image_dir= os.path.sep.join([config.OUTPUT_PATH, "performance_vizualizations"+ str(sigma)]))
	checkpoint=ModelCheckpoint(filepath_weights, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
	csv_logger = CSVLogger(os.path.sep.join([config.OUTPUT_PATH, 'training_sigma' + str(sigma) +'.log']))
	if sigma == final_sigma:
		config.EPOCHS = 50
		callbacks=[csv_logger, checkpoint, performance_cbk] 
	else:
		callbacks=[csv_logger, checkpoint, earlystopper,performance_cbk] 

	lr = config.INITIAL_LR_UNFROZEN

	opt = Adam(learning_rate=lr)
		
	model.compile(
	loss="categorical_crossentropy", 
	optimizer=opt,
	metrics=METRICS)

	print("Training with sigma...",sigma)

	H = model.fit(
		trainGen,
		steps_per_epoch=totalTrain //  config.BATCH_SIZE, 
		epochs=config.EPOCHS, 
		validation_data=valGen,
		validation_steps=totalVal //  config.BATCH_SIZE, 
		verbose=2,
		callbacks=callbacks)
	e = len(H.history['loss'])
	
	functions.plot_trainingLoss(H, e, os.path.sep.join([config.OUTPUT_PATH, 'Loss_sigma' + str(sigma) +'.png']))
	functions.plot_trainingAccuracy(H, e, os.path.sep.join([config.OUTPUT_PATH, 'Accuracy_sigma' + str(sigma) +'.png']))
	
	trainGen.reset()
	valGen.reset()
	testGen.reset()
	
	model.save(filepath_model)
	model.save_weights(filepath_weights)
	for layer in model.layers:
		print(layer, layer.trainable)
	del model

