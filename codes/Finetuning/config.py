import os

OUTPUT_PATH = "output"
BASE_PATH = "/Users/jorge/Documents/GitHub/basil/dataset/patch_images/background_hand_removed/split1"

# 1 = RMSProp
# 2 = Adam
# 3 = AdaGrad
# 4 = SGD
OPT = 2

BATCH_SIZE = 32
  
TRAIN = "training"
VAL = "validation"
TEST = "validation"

CLASSES = ["W4T1","W4T2", "W4T3", "W4T4"]

SIZE_IMAGE = 224

EPOCHS = 50
EPOCHS_WARM = 10

INITIAL_LR = 1e-2
INITIAL_LR_WARMUP = 1e-3
INITIAL_LR_UNFROZEN = 1e-5

ROTATION = 20
HORIZONTAL_FLIP= True
BRIGHTNESS_RANGE= [0.8, 1.2]
FILL_MODE = "nearest"
CVAL = 0


LE_PATH = os.path.sep.join([OUTPUT_PATH, "le.cpickle"])
BASE_CSV_PATH = OUTPUT_PATH

MODEL_WARMUP_PATH = os.path.sep.join([OUTPUT_PATH, "basil_warmup.model"])
MODEL_UNFROZEN_PATH = os.path.sep.join([OUTPUT_PATH, "basil_unfrozen.model"])

CR_PATH = os.path.sep.join([OUTPUT_PATH, "cr.log"])

UNFROZEN_LOSS_PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "unfrozenLoss.png"])
UNFROZEN_ACCURACY_PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "unfrozenAccuracy.png"])

WARMUP_LOSS_PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "warmupLoss.png"])
WARMUP_ACCURACY_PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "warmupAccuracy.png"])