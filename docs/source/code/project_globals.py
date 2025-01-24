import pathlib

DATADIR = pathlib.Path('../data/raw')
TRAIN_DIR = DATADIR / 'train'
VAL_DIR = DATADIR / 'val'
TEST_DIR = DATADIR / 'test'
CHECKPOINT_DIR = DATADIR / 'checkpoints'