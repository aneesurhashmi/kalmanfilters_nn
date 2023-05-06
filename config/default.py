from yacs.config import CfgNode as CN

_C = CN()

# model default values
_C.MODEL = CN()
_C.MODEL.TYPE = "RNN"
_C.MODEL.INPUT_SIZE = 19
_C.MODEL.OUTPUT_SIZE = 3
_C.MODEL.HIDDEN_SIZE = 128
_C.MODEL.NUM_LAYERS = 2
_C.MODEL.SEQUENCE_LENGTH = 28
_C.MODEL.DROPOUT = 0.2

# Solver default values
_C.SOLVER = CN()
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.BATCH_SIZE = 100
_C.SOLVER.LR = 0.001
_C.SOLVER.LOG_STEP = 500
_C.SOLVER.GPUS_PER_TRIAL = 1

# Data default values
_C.DATA = CN()
_C.DATA.TRAIN_DATA_DIR = "./data/2D/generated_data"
_C.DATA.EVAL_DATA_DIR = "./data/2D/evaluation_data"
_C.DATA.SETTING = '2D'
_C.DATA.ENVIRONMENT = 'fbcampus'

# Output default values
_C.OUTPUT = CN()
_C.OUTPUT.OUTPUT_DIR = "./output"
_C.OUTPUT.PLOT = False