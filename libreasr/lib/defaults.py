DEFAULT_CONFIG_PATH = "./config/testing.yaml"
DEFAULT_SR = 16000
DEFAULT_BATCH_SIZE = 6

LM_ALPHA = 0.1
LM_THETA = 1.0
LM_TEMP = 1.0
LM_DEBUG = False

MODEL_TEMP = 1.0

# streaming
DEFAULT_STREAM_OPTS = {
    "reset_thresh": 7000,
    "buffer_n_frames": 3,
    "downsample": 8,
    "n_buffer": 2,
}
