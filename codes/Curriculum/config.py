import os

BASE_PATH = "/Users/jorge/Documents/GitHub/basil/dataset/patch_images/background_hand_removed/split1"
OUTPUT_PATH = "output"

BATCH_SIZE = 32

TRAIN = "training"
VAL = "validation"
TEST = "validation"

CLASSES = ["W4T1", "W4T2", "W4T3", "W4T4"]

SIZE_IMAGE = 224
CHANNELS = 3
EPOCHS_WARM = 10 
EPOCHS = 30
INITIAL_LR_WARMUP = 1e-3
INITIAL_LR_UNFROZEN = 1e-5

ROTATION = 20
HORIZONTAL_FLIP= True
BRIGHTNESS_RANGE= [0.8, 1.2]
FILL_MODE = "nearest"
CVAL = 0

PRETRAINED_MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "pretrained_model.h5"])

LOADMODEl_INI = os.path.sep.join([OUTPUT_PATH, "all_model_resnet_first_stage.h5"])
LOADWEIGHT_INI = os.path.sep.join([OUTPUT_PATH, "model_weights_resnet_first_stage.h5"])
