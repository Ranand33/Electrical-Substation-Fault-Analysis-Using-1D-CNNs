DATA_DIR = "preprocessed_pcm_data"
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
PATIENCE = 10
GRAD_CLIP = 1.0
USE_AMP = True

MODEL_CONFIGS = {
    "simple": {"lr": 1e-3, "base_ch": 32, "dropout": 0.3},
    "resnet": {"lr": 5e-4, "base_ch": 32, "dropout": 0.3},
    "tcn": {"lr": 1e-3, "base_ch": 32, "dropout": 0.2},
}

FILTER_HIGH_VIOLATION_PCMS = True
VIOLATION_RATE_THRESHOLD = 0.95

THRESHOLD_METRIC = "recall_at_precision"
MIN_PRECISION = 0.40
