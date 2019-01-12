import torch
import sys
from torch.autograd import Variable
import numpy as np
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from skimage import io
from skimage.transform import resize

model = create_model(opt)

input_height = 384
input_width = 512


def _generate_depth(img_path, output_path):
    total_loss = 0
    toal_count = 0
    #print("============================= TEST ============================")
    model.switch_to_eval()

    img = np.float32(io.imread(img_path)) / 255.0
    img = resize(img, (input_height, input_width), order=1)
    input_img = torch.from_numpy(np.transpose(img,
                                              (2, 0, 1))).contiguous().float()
    input_img = input_img.unsqueeze(0)

    input_images = Variable(input_img)
    pred_log_depth = model.netG.forward(input_images)
    pred_log_depth = torch.squeeze(pred_log_depth)

    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1 / pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    pred_inv_depth = pred_inv_depth / np.amax(pred_inv_depth)

    io.imsave(output_path, pred_inv_depth)
    print("saving image")
    # print(pred_inv_depth.shape)
    #sys.exit()


import os
import logging
from os import system
from shutil import copyfile
from PIL import Image, ImageMath, ImageFilter, ImageOps, ImageEnhance
#from subprocess import DEVNULL, STDOUT, check_call
import numpy as np

temp_input_path = "demo.jpg"
temp_depth_path = "demo.png"


def generate_depth_map(input_path):
    _generate_depth(input_path, temp_depth_path)
    print("generated")
    im = Image.open(input_path)
    depth = Image.open(temp_depth_path)
    rgb_depth = ImageMath.eval('im/256', {'im': depth})
    rgb_depth = rgb_depth.resize(im.size, Image.ANTIALIAS)
    rgb_depth.convert("RGB")
    return rgb_depth


def _lift_shadows(v):
    return v + (255 - v)**2 * 0.001 - 5


def _drop_highlights(v):
    return v - (v)**2 * 0.001


def normalize_depth_contrast(img):
    img = img.convert("RGB")
    img = img.filter(ImageFilter.SMOOTH_MORE)
    arr = np.array(img).astype(float)
    for _ in range(0, 2):
        arr = _lift_shadows(arr)
        arr = _drop_highlights(arr)
    arr = arr.astype("uint8")
    img = Image.fromarray(arr)
    img = img.convert("RGB")
    enh = ImageEnhance.Contrast(img)
    img = enh.enhance(3)
    return img


def _remove_temp_files():
    _try_remove(temp_input_path)

    _try_remove(temp_depth_path)


def _try_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
