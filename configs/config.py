TRAIN_DATA_PATH = 'datasets/train/'
TEST_DATA_PATH = 'datasets/test/'

TRAIN_JSON_PATH = 'json/dataset_train.json'
TEST_JSON_PATH = 'json/dataset_test.json'
INFER_JSON_PATH = 'json/dataset_test.json'

LOG_PATH = 'logs/'
CHECKPOINT_PATH = 'checkpoints/unet_epoch_1000.pth'

NUM_CLASSES = 4
DEPTH = 3 # 3维组合输入

MAPPING = {
	(126, 126, 255): 0, #AM
	(255, 255, 0): 1, #SE
	(0, 255, 0): 2, #carbon
	(255, 0, 0): 3 #void
}
COLORS = [(126, 126, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
