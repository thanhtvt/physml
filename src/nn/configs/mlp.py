# DATA CONFIGS
DATA_DIR = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/data"
EXPAND_STEPS = 1
NOISE = 1.0
TRAIN_RATIO = 0.9

# QM7 STATS
COULOMB_MEAN = 1.6812959
COULOMB_STD = 6.700323
EIGEN_MEAN = 13.707274
EIGEN_STD = 30.161158
ENERGY_MEAN = -1538.0377
ENERGY_STD = 223.91891

# LOGGING
LOGFILE = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/logs/nn.log"

# WANDB CONFIG
WANDB_PROJECT = "physml"
WANDB_DIR = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/logs"

# TRAINING CONFIGS
EPOCHS = 1000
BATCH_SIZE = 256
CHECKPOINT_DIR = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/checkpoints/nn"
LOG_INTERVAL = 20
NUM_WORKERS = 8
PATIENCE = 15

# MODEL CONFIGS
HIDDEN_SIZES = [400, 100]    # [400, 100]
ACTIVATION = "relu"
DROPOUT_RATE = 0.2

# OPTIMIZER CONFIGS
OPTIMIZER = "adamw"
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.01
MOMENTUM = 0.9
