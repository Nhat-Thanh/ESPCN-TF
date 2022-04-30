import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.common import *
import numpy as np
from model import ESPCN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=float, default=2,                           help='-')
parser.add_argument("--image-path",   type=str,   default="dataset/test2.png",         help='-')
parser.add_argument("--ckpt-path",    type=str,   default="checkpoint/x2/ESPCN-x2.h5", help='-')

FLAGS, unparsed = parser.parse_known_args()
scale = FLAGS.scale
image_path = FLAGS.image_path
ckpt_path = FLAGS.ckpt_path

if scale not in [2, 3, 4]:
    ValueError("scale must be 2, 3, or 4")

# -----------------------------------------------------------
#  load model
# -----------------------------------------------------------

model = ESPCN(scale)
model.load_weights(ckpt_path)


# -----------------------------------------------------------
#  read image and save bicubic image
# -----------------------------------------------------------

lr_image = read_image(image_path)
bicubic_image = upscale(lr_image, scale)
write_image("bicubic.png", bicubic_image)


# -----------------------------------------------------------
# preprocess lr image 
# -----------------------------------------------------------
lr_image = gaussian_blur(lr_image, sigma=0.3)
lr_image = rgb2ycbcr(lr_image)
lr_image = norm01(lr_image[tf.newaxis, ...])


# -----------------------------------------------------------
#  predict and save image
# -----------------------------------------------------------

sr_image = model.predict(lr_image)[0]

sr_image = denorm01(sr_image)
sr_image = np.uint8(sr_image)
sr_image = ycbcr2rgb(sr_image)

write_image("sr.png", sr_image)

