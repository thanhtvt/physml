# DATA CONFIGS
DATA_DIR = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/data"
SAVE_DATA_DIR = DATA_DIR + "/processed"

# LOGGING
LOGFILE = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/logs/nn.log"

# WANDB CONFIG
WANDB_PROJECT = "physml"
WANDB_DIR = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/logs"
WANDB_ID = "z01akjcl"

# TRAINING CONFIGS
EPOCHS = 1000
BATCH_SIZE = 128
CHECKPOINT_DIR = "/home/jonnyjack/workspace/FPTAI/research-test/son-hy/physml/checkpoints/nn"
LOG_INTERVAL = 20
NUM_WORKERS = 8
PATIENCE = 15

# MODEL CONFIGS
NODE_INPUT_DIM = 1
EDGE_INPUT_DIM = 1
GNN_NODE_OUTPUT_DIM = 8
GNN_EDGE_HIDDEN_DIM = 16
NUM_STEP_MESSAGE_PASSING = 3
READOUT_NODE_HIDDEN_DIM = 8
READOUT_NODE_OUTPUT_DIM = 8
POOLING_MODE = "mean"
OUTPUT_DIM = 1

# OPTIMIZER CONFIGS
OPTIMIZER = "adamw"
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
MOMENTUM = 0.9
