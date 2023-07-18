from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
#Linux
from tensorflow.keras.optimizers import Adam
#Mac OS
#from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
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

print('ResNet50V2 base model...')
baseModel = ResNet50V2(
	weights="imagenet",
	include_top=False, 
	input_tensor=Input(shape=(config.SIZE_IMAGE, config.SIZE_IMAGE, config.CHANNELS)))

print("Modifying base model...")
inputs = Input(shape =(config.SIZE_IMAGE, config.SIZE_IMAGE, config.CHANNELS))
baseModel(inputs, training = False)
baseModel = functions.insert_layer_nonseq2(baseModel, insert_layer_name='Depthwise')
baseModel.trainable = False 
baseModel.save(config.PRETRAINED_MODEL_PATH)
baseModel = load_model(config.PRETRAINED_MODEL_PATH)
baseModel = functions.insert_layer_nonseq2(baseModel, layer_regex="post_relu")
baseModel.save(config.PRETRAINED_MODEL_PATH)
baseModel = load_model(config.PRETRAINED_MODEL_PATH)
baseModel = functions.insert_layer_nonseq2(baseModel, layer_regex="_average_pooling2d")
baseModel.save(config.PRETRAINED_MODEL_PATH)
baseModel = load_model(config.PRETRAINED_MODEL_PATH)
#baseModel.summary()

print("Assigning gaussian weights...")
kernel_size = (3,3) 
sigma = 0.001
baseModel = functions.assign_gaussian_weights (baseModel, kernel_size, sigma)

print("Model compiling...")
METRICS = [ CategoricalAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall')] 
opt = Adam(learning_rate=config.INITIAL_LR_WARMUP)
baseModel.compile(loss="categorical_crossentropy", optimizer=opt, metrics=METRICS)

print("Model training...")
filepath = os.path.sep.join([config.OUTPUT_PATH, "weights_warmup-improvement-{epoch:04d}-{val_accuracy:.4f}.hdf5"])
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger(os.path.sep.join([config.OUTPUT_PATH, 'training_warmup.log']))
earlystopper=EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

performance_cbk = functions.PerformanceVisualizationCallback(
    model=baseModel,
    validation_data=testGen,
    image_dir=os.path.sep.join([config.OUTPUT_PATH, "performance_vizualizations"]))

H = baseModel.fit(
    trainGen,
    steps_per_epoch=totalTrain // config.BATCH_SIZE,
    epochs=config.EPOCHS_WARM,
    validation_data=valGen,
    validation_steps=totalVal // config.BATCH_SIZE,
    callbacks=[csv_logger, earlystopper, checkpoint,performance_cbk])

e = len(H.history['loss'])

functions.plot_trainingLoss(H, e, os.path.sep.join([config.OUTPUT_PATH, "warmupLoss.png"]))
functions.plot_trainingAccuracy(H, e, os.path.sep.join([config.OUTPUT_PATH, "warmupAccuracy.png"]))

baseModel.save(os.path.sep.join([config.OUTPUT_PATH, "all_model_resnet_first_stage.h5"]))
baseModel.save_weights(os.path.sep.join([config.OUTPUT_PATH, "model_weights_resnet_first_stage.h5"]))

del baseModel
