# define configurations for training run
RUN = 'extract_feat'
# comment can be useful to add additional information to run_config.txt file
RUN_COMMENT = """Enter comment here."""
SEED = 41
IMAGE_INPUT_SIZE = 512
# IMAGE_INPUT_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 16
# CHECKPOINT = None
MULTI_GPU = False
CHECKPOINT = './checkpoints/full_model_checkpoint_val_loss_19.793_overall_steps_155252.pt'
CHECKPOINT_completer = './checkpoints/val_loss_0.054_epoch_88.pth'
