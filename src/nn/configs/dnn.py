# DATA CONFIGS
DATA_DIR = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/data"
EXPAND_STEPS = 1
NOISE = 1.0
TRAIN_RATIO = 0.9

# LOGGING
LOGFILE = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/logs/nn.log"

# TRAINING CONFIGS
EPOCHS = 1000
BATCH_SIZE = 512
CHECKPOINT_DIR = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/checkpoints/nn"
LOG_INTERVAL = 1
NUM_WORKERS = 8

# MODEL CONFIGS
HIDDEN_SIZES = [400, 100]

# OPTIMIZER CONFIGS
OPTIMIZER = "adam"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.
