from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from utils.dataset import dataset
from utils.common import PSNR
from model import ESPCN 
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=100000,          help='-')
parser.add_argument("--batch-size",     type=int, default=128,             help='-')
parser.add_argument("--scale",          type=int, default=2,               help='-')
parser.add_argument("--save-every",     type=int, default=100,             help='-')
parser.add_argument("--save-best-only", type=int, default=0,               help='-')
parser.add_argument("--ckpt-dir",       type=str, default="checkpoint/x2", help='-')


FLAG, unparsed = parser.parse_known_args()
steps = FLAG.steps
batch_size = FLAG.batch_size
scale = FLAG.scale
ckpt_dir = FLAG.ckpt_dir
save_every = FLAG.save_every
model_path = os.path.join(ckpt_dir, f"ESPCN-x{scale}.h5")
save_best_only = (FLAG.save_best_only == 1)


# -----------------------------------------------------------
#  Init datasets
# -----------------------------------------------------------

dataset_dir = "dataset"
lr_crop_size = 17
hr_crop_size = 17 * 2
if scale == 3:
    hr_crop_size = 17 * 3
elif scale == 4:
    hr_crop_size = 17 * 4

train_set = dataset(dataset_dir, "train")
train_set.generate(lr_crop_size, hr_crop_size)
train_set.load_data()

valid_set = dataset(dataset_dir, "validation")
valid_set.generate(lr_crop_size, hr_crop_size)
valid_set.load_data()

test_set = dataset(dataset_dir, "test")
test_set.generate(lr_crop_size, hr_crop_size)
test_set.load_data()


# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

espcn = ESPCN(scale)
espcn.setup(optimizer=Adam(learning_rate=2e-4),
            loss=MeanSquaredError(),
            model_path=model_path,
            metric=PSNR)

espcn.load_checkpoint(ckpt_dir)
espcn.train(train_set, valid_set, 
            steps=steps, batch_size=batch_size,
            save_best_only=save_best_only, 
            save_every=save_every)


# -----------------------------------------------------------
#  Test
# -----------------------------------------------------------
# todo: use the best checkpoint to evaluate the test set
espcn.load_weights(model_path)
loss, metric = espcn.evaluate(test_set)
print(f"loss: {loss} - psnr: {metric}")
