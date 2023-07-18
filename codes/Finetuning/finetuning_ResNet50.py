# USAGE
# python3 finetuning_ResNet50.py

from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, CategoricalAccuracy, Precision, Recall, AUC
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import RMSprop
#Linux
from tensorflow.keras.optimizers import Adam
#Mac OS
#from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
import numpy as np
from imutils import paths
import os
import time
import functions
import config
matplotlib.use("Agg")

trainPath = os.path.sep.join([config.BASE_PATH, config.TRAIN])
valPath = os.path.sep.join([config.BASE_PATH, config.VAL])
testPath = os.path.sep.join([config.BASE_PATH, config.VAL])

totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(valPath)))

trainAug = ImageDataGenerator(
    rescale = 1./255,
	rotation_range = config.ROTATION, 
	horizontal_flip = config.HORIZONTAL_FLIP, 
	brightness_range = config.BRIGHTNESS_RANGE, 
	preprocessing_function = functions.cutout,
   	fill_mode = config.FILL_MODE, 
	cval=config.CVAL
)

valAug = ImageDataGenerator(rescale=1./255)
testAug = ImageDataGenerator(rescale=1./255)

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
    testPath,
    class_mode="categorical",
    target_size=(config.SIZE_IMAGE, config.SIZE_IMAGE),
    color_mode="rgb",
    shuffle=False,
    batch_size=config.BATCH_SIZE)


baseModel = ResNet50V2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(config.SIZE_IMAGE, config.SIZE_IMAGE, 3)))

print("Base model summary...")
baseModel.summary()

baseModel.trainable = False
inputs = Input(shape=(config.SIZE_IMAGE, config.SIZE_IMAGE, 3))
x = baseModel(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(len(config.CLASSES), activation="softmax")(x) 
model = Model(inputs, outputs)

#model.summary()

#for layer in model.layers:
#    print(layer, layer.trainable)


print("Compiling model...")
if config.OPT == 1:
    opt = RMSprop(learning_rate=config.INITIAL_LR_WARMUP)
elif config.OPT == 2:
    opt = Adam(learning_rate=config.INITIAL_LR_WARMUP)
elif config.OPT == 3:
    opt = Adagrad(learning_rate=config.INITIAL_LR_WARMUP)

METRICS = [
    TruePositives(name='tp'),
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'),
    CategoricalAccuracy(name='accuracy'),
    Precision(name='precision'),
    Recall(name='recall'),
    AUC(name='auc'),
]

model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=METRICS)

filepath = "output/weights_warmup-improvement-{epoch:04d}-{val_accuracy:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

print("Training head...")

csv_logger = CSVLogger('output/training_warmup.log')

start = time.time()

H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // config.BATCH_SIZE,
    epochs=config.EPOCHS_WARM,
    validation_data=valGen,
    validation_steps=totalVal // config.BATCH_SIZE,
    callbacks=[csv_logger, checkpoint])

e = len(H.history['loss'])

functions.plot_trainingLoss(H, e, config.WARMUP_LOSS_PLOT_PATH)
functions.plot_trainingAccuracy(H, e, config.WARMUP_ACCURACY_PLOT_PATH)

print("Serializing warmup network...")
model.save(config.MODEL_WARMUP_PATH)

trainGen.reset()
valGen.reset()

baseModel.trainable = True

#for layer in baseModel.layers:
#    print("{}: {}".format(layer, layer.trainable))

#model.summary()

print("Re-compiling model...")
if config.OPT == 1:
    opt = RMSprop(learning_rate=config.INITIAL_LR_UNFROZEN)
elif config.OPT == 2:
    opt = Adam(learning_rate=config.INITIAL_LR_UNFROZEN)
elif config.OPT == 3:
    opt = Adagrad(learning_rate=config.INITIAL_LR_UNFROZEN)

model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=METRICS)

filepath = "output/weights_unfrozen-improvement-{epoch:04d}-{val_accuracy:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
earlystopper = EarlyStopping(
    monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)


csv_logger = CSVLogger('output/training_unfrozen.log')

performance_cbk = functions.PerformanceVisualizationCallback(
    model=model,
    validation_data=testGen,
    image_dir='output/performance_vizualizations')

callbacks = [csv_logger,checkpoint, performance_cbk]

H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // config.BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal // config.BATCH_SIZE,
    epochs=config.EPOCHS,
    callbacks=callbacks)

end = time.time()
duration = end - start



e = len(H.history['loss'])

print('\n model_ResNet50V2 Fine Tuning took %0.2f seconds (%0.1f minutes) to train for %d epochs' %
      (duration, duration/60, e+config.EPOCHS_WARM))

functions.plot_trainingLoss(H, e, config.UNFROZEN_LOSS_PLOT_PATH)
functions.plot_trainingAccuracy(H, e, config.UNFROZEN_ACCURACY_PLOT_PATH)

print("Serializing unfrozen network...")
model.save(config.MODEL_UNFROZEN_PATH)
del model
