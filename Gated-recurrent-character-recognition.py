import random
import time
from collections import namedtuple
from typing import Tuple
import cv2
import numpy as np
from jiwer import cer, wer
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import numpy as np
import cv2
import os
import pandas as pd
import string
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import activations
from autocorrect import Speller
spell = Speller()
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMNotebookCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from tensorflow.keras.layers import Layer, Conv2D, Multiply, Activation
from tensorflow.keras.constraints import MaxNorm
from keras.layers.convolutional_recurrent import ConvLSTM2D
import numpy as np
import tensorflow as tf
class DeformableConv2D(tf.keras.layers.Conv2D):
    def __init__(self, batch_size, filters, kernel_size, name, kernel_initializer, **kwargs):
        self.batch_size = batch_size       
        super().__init__(filters = filters, kernel_size = kernel_size, name = name, kernel_initializer = kernel_initializer, **kwargs)
        self.kernel = None
        self.bias = None
        self.offset_kernel = None
        self.offset_bias = None
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(name = 'kernel_{}' .format(self.name), shape = self.kernel_size + (input_dim, self.filters), initializer = self.kernel_initializer, trainable=True)
        self.bias = self.add_weight(name = 'bias_{}' .format(self.name), shape = (self.filters,), initializer=self.kernel_initializer, trainable=True)
        self.offset_kernel = self.add_weight(name = 'offset_kernel_{}'.format(self.name), shape = self.kernel_size + (input_dim, 2), initializer = tf.zeros_initializer(), trainable = True)
        self.offset_bias = self.add_weight(name = 'offset_bias_{}' .format(self.name), shape = (2,), initializer = tf.zeros_initializer(), trainable = True)
        self.built = True
    def call(self, input):
        return self.deformable_conv(input, self.name, self.batch_size, self.filters, self.kernel_size)
    def deformable_conv(self, input, name, batch_size, filters, kernel_size):
        input_size = input.get_shape().as_list()[1]
        grid_x, grid_y = tf.meshgrid(tf.range(input_size), tf.range(input_size))
        INPUT_GRID = []
        for grid in [grid_x, grid_y]:
            grid = tf.reshape(grid, [1, *grid.get_shape(), 1])
            patched_grid = tf.compat.v1.extract_image_patches(grid, ksizes = (1,) + kernel_size + (1,), strides = [1, 1, 1, 1], rates = [1, 1, 1, 1], padding = 'SAME')
            batch_patched_grid = tf.tile(patched_grid, [batch_size, 1, 1, 1])
            batch_patched_grid = tf.cast(batch_patched_grid, tf.float32)
            INPUT_GRID.append(batch_patched_grid)
        offset = tf.nn.conv2d(input, filters = self.offset_kernel, strides = [1, 1, 1, 1], padding = 'SAME')
        offset += self.offset_bias
        offset = tf.reshape(offset, [batch_size, input_size, input_size, -1, 2])
        off_x, off_y = offset[...,0], offset[...,1]
        OFFSET = []
        for offset in [off_x, off_y]:
            patched_offset = tf.compat.v1.extract_image_patches(offset, ksizes = (1,) + kernel_size + (1,), strides = [1, 1, 1, 1], rates = [1, 1, 1, 1], padding = 'SAME')
            OFFSET.append(patched_offset)

        x = tf.clip_by_value(INPUT_GRID[0] + OFFSET[0], 0, input_size - 1)
        y = tf.clip_by_value(INPUT_GRID[1] + OFFSET[1], 0, input_size - 1)
        x0, y0 = tf.cast(x, 'int32'), tf.cast(y, 'int32')
        x1, y1 = x0 + 1, y0 + 1
        x0, x1 = [tf.clip_by_value(i, 0, input_size - 1) for i in [x0, x1]]
        y0, y1 = [tf.clip_by_value(i, 0, input_size - 1) for i in [y0, y1]]
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        P = []
        for index in indices:
            tmp_y, tmp_x = index
            batch, h, w, n = tmp_y.get_shape().as_list()
            batch_idx = tf.reshape(tf.range(batch), (batch, 1, 1, 1))
            b = tf.tile(batch_idx, (1, h, w, n))
            pixel_idx = tf.stack([b, tmp_y, tmp_x], axis = -1)
            p = tf.gather_nd(input, pixel_idx)
            P.append(p)
        x0, x1, y0, y1 = [tf.compat.v1.to_float(i) for i in [x0, x1, y0, y1]]
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        pixels = tf.add_n([w0 * P[0], w1 * P[1], w2 * P[2], w3 * P[3]]) 
        pixels = tf.reshape(pixels, [batch_size, input_size * 3, input_size * 3, -1])
        output_logits = tf.nn.conv2d(pixels, filters = self.kernel, strides = [1, 3, 3, 1], padding = 'VALID')
        output_logits += self.bias
        return output_logits
    def _inference_grid_offset(self, input_images):
        if len(input_images.shape) < 4:
            raise "No"
        b, h, w, c = input_images.shape
        input_tensor = tf.placeholder(tf.float32, [None, h, w, c])
        grid_offset = tf.nn.conv2d(input_tensor, filter = self.offset_kernel, strides = [1, 1, 1, 1], padding = 'SAME')
        grid_offset += self.offset_bias
        sess = tf.keras.backend.get_session()
        offset = sess.run(grid_offset, feed_dict = {input_tensor : input_images})
        return offset
class GatedConv2D(Conv2D):
    def __init__(self, **kwargs):
        super(GatedConv2D, self).__init__(**kwargs)
    def call(self, inputs):
        output = super(GatedConv2D, self).call(inputs)
        linear = Activation("linear")(inputs)
        sigmoid = Activation("sigmoid")(output)
        return Multiply()([linear, sigmoid])
    def get_config(self):
        config = super(GatedConv2D, self).get_config()
        return config
class FullGatedConv2D(Conv2D):
    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters
    def call(self, inputs): 
        output = super(FullGatedConv2D, self).call(inputs)
        linear = Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.nb_filters:])
        return Multiply()([linear, sigmoid])
    def compute_output_shape(self, input_shape):
        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters * 2,)
    def get_config(self):
        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config
class OctConv2D(Layer):
    def __init__(self,
                 filters,
                 alpha,
                 kernel_size=(3,3),
                 strides=(1,1),
                 padding="same",
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        assert alpha >= 0 and alpha <= 1
        assert filters > 0 and isinstance(filters, int)
        super().__init__(**kwargs)
        self.alpha = alpha
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.low_channels = int(self.filters * self.alpha)
        self.high_channels = self.filters - self.low_channels
    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 4 and len(input_shape[1]) == 4
        assert input_shape[0][1] // 2 >= self.kernel_size[0]
        assert input_shape[0][2] // 2 >= self.kernel_size[1]
        assert input_shape[0][1] // input_shape[1][1] == 2
        assert input_shape[0][2] // input_shape[1][2] == 2
        assert K.image_data_format() == "channels_last"
        high_in = int(input_shape[0][3])
        low_in = int(input_shape[1][3])
        self.high_to_high_kernel = self.add_weight(name="high_to_high_kernel",
                                                   shape=(*self.kernel_size, high_in, self.high_channels),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint)
        self.high_to_low_kernel = self.add_weight(name="high_to_low_kernel",
                                                  shape=(*self.kernel_size, high_in, self.low_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        self.low_to_high_kernel = self.add_weight(name="low_to_high_kernel",
                                                  shape=(*self.kernel_size, low_in, self.high_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        self.low_to_low_kernel = self.add_weight(name="low_to_low_kernel",
                                                 shape=(*self.kernel_size, low_in, self.low_channels),
                                                 initializer=self.kernel_initializer,
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)
        super().build(input_shape)
def _create_octconv_last_block(inputs, ch, alpha):
    high, low = inputs

    high, low = OctConv2D(filters=ch, alpha=alpha)([high, low])
    high = BatchNormalization()(high)
    high = Activation("relu")(high)

    low = BatchNormalization()(low)
    low = Activation("relu")(low)

    high_to_high = Conv2D(ch, 3, padding="same")(high)
    low_to_high = Conv2D(ch, 3, padding="same")(low)
    low_to_high = Lambda(lambda x: K.repeat_elements(K.repeat_elements(x, 2, axis=1), 2, axis=2))(low_to_high)

    x = Add()([high_to_high, low_to_high])
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x
    def call(self, inputs):
        assert len(inputs) == 2
        high_input, low_input = inputs
        high_to_high = K.conv2d(high_input, self.high_to_high_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last")
        high_to_low = K.pool2d(high_input, (2, 2), strides=(2, 2), pool_mode="avg")
        high_to_low = K.conv2d(high_to_low, self.high_to_low_kernel, strides=self.strides, padding=self.padding, data_format="channels_last")
        low_to_high = K.conv2d(low_input, self.low_to_high_kernel,
                               strides=self.strides, padding=self.padding,
                               data_format="channels_last")
        low_to_high = K.repeat_elements(low_to_high, 2, axis=1)
        low_to_high = K.repeat_elements(low_to_high, 2, axis=2)
        low_to_low = K.conv2d(low_input, self.low_to_low_kernel, strides=self.strides, padding=self.padding, data_format="channels_last")
        high_add = high_to_high + low_to_high
        low_add = low_to_low + high_to_low
        return [high_add, low_add]
    def compute_output_shape(self, input_shapes):
        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (*high_in_shape[:3], self.high_channels)
        low_out_shape = (*low_in_shape[:3], self.low_channels)
        return [high_out_shape, low_out_shape]
    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "filters": self.filters,
            "alpha": self.alpha,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
        }
        return out_config
        
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
max_len = 0 
characters = "! \"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
base_path = "data"
base_image_path = os.path.join(base_path, "words")
batch_size = 1
padding_token = 99
image_width = 64
image_height = 64
AUTOTUNE = tf.data.AUTOTUNE
words_list = [] 
words = open(f"{base_path}/words.txt", "r").readlines()
for line in words:
    if line[0] == "#":
        continue
    if line.split(" ")[1] != "err":
        words_list.append(line)
np.random.shuffle(words_list)
split_idx = int(0.98 * len(words_list))
train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:]
val_split_idx = int(0.5 * len(test_samples))
validation_samples = test_samples[:val_split_idx]
test_samples = test_samples[val_split_idx:]
assert len(words_list) == len(train_samples) + len(validation_samples) + len(test_samples)
print("===================================================================================================")
print("===================================================================================================")
print(f"total samples: {len(words_list)}")
print(f"Total training samples: {len(train_samples)}")
print(f"Total validation samples: {len(validation_samples)}")
print(f"Total test samples: {len(test_samples)}")
def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels
def listToString(s): 
    str1 = "" 
    for ele in s: 
        str1 += ele    
    return str1
def encode_to_labels(txt):
    dig_lst = []
    for index, chara in enumerate(txt):
        dig_lst.append(characters.index(chara))
    return dig_lst

DeslantRes = namedtuple('DeslantRes', 'img, shear_val, candidates')
Candidate = namedtuple('Candidate', 'shear_val, score')
def _get_shear_vals(lower_bound: float,
                    upper_bound: float,
                    step: float) -> Tuple[float]:
    return tuple(np.arange(lower_bound, upper_bound + step, step))
def _shear_img(img: np.ndarray,
               s: float, bg_color: int,
               interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    h, w = img.shape
    offset = h * s
    w = w + int(abs(offset))
    tx = max(-offset, 0)
    shear_transform = np.asarray([[1, s, tx], [0, 1, 0]], dtype=float)
    img_sheared = cv2.warpAffine(img, shear_transform, (w, h), flags=interpolation, borderValue=bg_color)
    return img_sheared
def _compute_score(img_binary: np.ndarray, s: float) -> float:
    img_sheared = _shear_img(img_binary, s, 0)
    h = img_sheared.shape[0]
    img_sheared_mask = img_sheared > 0
    first_fg_px = np.argmax(img_sheared_mask, axis=0)
    last_fg_px = h - np.argmax(img_sheared_mask[::-1], axis=0)
    num_fg_px = np.sum(img_sheared_mask, axis=0)
    dist_fg_px = last_fg_px - first_fg_px
    col_mask = np.bitwise_and(num_fg_px > 0, dist_fg_px == num_fg_px)
    masked_dist_fg_px = dist_fg_px[col_mask]
    score = sum(masked_dist_fg_px ** 2)
    return score
def deslant_img(img: np.ndarray,
                optim_algo: 'str' = 'grid',
                lower_bound: float = -2,
                upper_bound: float = 2,
                num_steps: int = 20,
                bg_color=255) -> DeslantRes:
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert optim_algo in ['grid', 'powell']
    assert lower_bound < upper_bound
    img_binary = cv2.threshold(255 - img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] // 255
    best_shear_val = None
    candidates = None
    if optim_algo == 'grid':
        step = (upper_bound - lower_bound) / num_steps
        shear_vals = _get_shear_vals(lower_bound, upper_bound, step)
        candidates = [Candidate(s, _compute_score(img_binary, s)) for s in shear_vals]
        best_shear_val = sorted(candidates, key=lambda c: c.score, reverse=True)[0].shear_val
    elif optim_algo == 'powell':
        bounds = [[lower_bound], [upper_bound]]
        s0 = [(lower_bound + upper_bound) / 2]
        # minimize the negative score
        def obj_fun(s):
            return -_compute_score(img_binary, s)
        res = pybobyqa.solve(obj_fun, x0=s0, bounds=bounds, seek_global_minimum=True)
        best_shear_val = res.x[0]
    res_img = _shear_img(img, best_shear_val, bg_color, cv2.INTER_LINEAR)
    return DeslantRes(res_img, best_shear_val, candidates)

number_of_samples = len(words_list)#100 #len(words_list)
def distortion_free_resize(image, img_size):
    j=0
    k=0
    (wt, ht) = img_size
    (h, w) = image.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    image = cv2.resize(image, new_size)
    target = np.zeros([ht, wt]) * 255
    #target[0:new_size[1], 0:new_size[0]] = image
    #j=wt-new_size[0]# pad right
    ########pad between
    #================
    if j%2==0:
      j=j//2
    else:
      j=(j+1)//2
    k=ht-new_size[1]
    if k%2==0:
      k=k//2
    else:
      k=(k+1)//2
    #================'''
    j=0# pad left
    #
    target[0+k:new_size[1]+k, 0+j:new_size[0]+j] = image
    target = cv2.flip(target, 0)
    #for no padding comment line below
    image = target
    image = cv2.transpose(target)
    return image

def preprocess_image_aspect_ratio(image_path, img_size=(image_width, image_height)):
    #image = tf.io.read_file(image_path)
    image = cv2.imread(image_path, 0)
    #image = tf.image.decode_png(image, 1)
    image = image.astype("float32")/255
    image = distortion_free_resize(image, img_size)
    #image = tf.cast(image, tf.float32) / 255.0
    image = np.expand_dims(image, axis=-1) 
    return image
def preprocess_image_deslanted(image_path, img_size=(image_width, image_height)):
    image = cv2.imread(image_path, 0)
    image = deslant_img(image)
    image = image.img
    image = image.astype("float32")/255
    image = distortion_free_resize(image, img_size)
    image = np.expand_dims(image, axis=-1)
    return image
number_of_samples = len(words_list)
def get_primal_data_train(samples):
    begin = time.time()
    paths = []
    train_input_length = []
    original_text_cleaned = []
    labels_indices = []
    label_input_length = []
    images = []
    for (i, file_line) in enumerate(samples): 
        line_split = file_line.strip()
        line_split = line_split.split(" ")
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(base_image_path, partI, partI + "-" + partII, image_name + ".png")
        if i > number_of_samples:
          break
        if i%10000==0:
            finish = time.time()
            time=finish-begin
            m, s = divmod(time, 60)
            h, m = divmod(m, 60)
            print(f"Till sample {i} processed, now remains {number_of_samples-i} to end. Till here took {h} Hours and {m} Minutes and {s} Seconds.") 
        if os.path.getsize(img_path):
            paths.append(img_path)
            train_input_length.append(image_height-1)
            original_text_cleaned.append((file_line.split("\n")[0]).split(" ")[-1].strip())
            labels_indices.append(encode_to_labels((file_line.split("\n")[0]).split(" ")[-1].strip()))
            label_input_length.append(len((file_line.split("\n")[0]).split(" ")[-1].strip()))
            images.append(preprocess_image_deslanted(img_path))
    return paths, train_input_length, label_input_length, original_text_cleaned, labels_indices, images
def get_primal_data_valid(samples):
    paths = []
    train_input_length = []
    original_text_cleaned = []
    labels_indices = []
    label_input_length = []
    images = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(base_image_path, partI, partI + "-" + partII, image_name + ".png")
        if i > number_of_samples:
          break
        if i%10000==0:
            print(f"Till sample {i} processed. Now remains {number_of_samples-i} to end")
        if os.path.getsize(img_path):
            paths.append(img_path)
            train_input_length.append(image_height-1)
            original_text_cleaned.append((file_line.split("\n")[0]).split(" ")[-1].strip())
            labels_indices.append(encode_to_labels((file_line.split("\n")[0]).split(" ")[-1].strip()))
            label_input_length.append(len((file_line.split("\n")[0]).split(" ")[-1].strip()))
            images.append(preprocess_image_deslanted(img_path))
    return paths, train_input_length, label_input_length, original_text_cleaned, labels_indices, images
def get_primal_data_test(samples):
    paths = []
    train_input_length = []
    original_text_cleaned = []
    labels_indices = []
    label_input_length = []
    images = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(base_image_path, partI, partI + "-" + partII, image_name + ".png")
        if i > number_of_samples:
          break
        if i%10000==0:
            print(f"Till sample {i} processed. Now remains {number_of_samples-i} to end")
        if os.path.getsize(img_path):
            paths.append(img_path)
            train_input_length.append(image_height-1)
            original_text_cleaned.append((file_line.split("\n")[0]).split(" ")[-1].strip())
            labels_indices.append(encode_to_labels((file_line.split("\n")[0]).split(" ")[-1].strip()))
            label_input_length.append(len((file_line.split("\n")[0]).split(" ")[-1].strip()))
            images.append(preprocess_image_deslanted(img_path))
    return paths, train_input_length, label_input_length, original_text_cleaned, labels_indices, images
begin = time.time()
print("---")
print(f"Preprocessing train data...")
train_img_paths, train_input_img_length, train_label_length, train_original_label, train_labels_indices, train_images = get_primal_data_train(train_samples)
print("---")
print(f"Preprocessing valid data...")
valid_img_paths, valid_input_img_length, valid_label_length, valid_original_label, valid_labels_indices, valid_images = get_primal_data_valid(validation_samples)
print("---")
print(f"Preprocessing test data...")
test_img_paths, test_input_img_length, test_label_length, test_original_label, test_labels_indices, test_images = get_primal_data_test(test_samples)
print(f"End of preprocessing Input data")
for i in train_original_label:
  max_len = max(max_len, len(i))
print("---")
print("Maximum length characters of the labels: ", max_len)
print("---")
train_padded_label = pad_sequences(train_labels_indices, maxlen=max_len, padding='post', value=len(characters))
valid_padded_label = pad_sequences(valid_labels_indices, maxlen=max_len, padding='post', value=len(characters))
def n(e):
  return np.asarray(e)
print(f"Train_imgs_shape: {n(train_images).shape}, Train_length_shape: {n(train_input_img_length).shape}, Train_padded_labels_shape: {n(train_padded_label).shape}")
print(f"Validation_imgs_shape: {n(valid_images).shape}, Validation_length_shape: {n(valid_input_img_length).shape}, Validation_padded_labels_shape: {n(valid_padded_label).shape}")
print("---")
finish = time.time()
time_duration = finish-begin
m, s = divmod(time_duration, 60)
h, m = divmod(m, 60)
print(f"Total time duration for input preprocessing took: {h} Hours and {m} Minutes and {s} Seconds")
print("---")
def ctc_lambda_func(args):
      y_pred, labels, input_length, label_length = args
      return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
def ctc_lambda_func(args):
      y_pred, labels, input_length, label_length = args
      return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

### The second proposed architecture: Fully Convolutional Optical Character Recognition IGCRA ### 

def build_model():
    input_img = keras.Input(shape=(image_width, image_height, 1), name="image") # (None, 128*32*1)
    the_labels = Input(name='the_labels', shape=[max_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
	# 1st Block
    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu', kernel_initializer="he_uniform")(input_img)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)
	
	# 2nd Block
    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer="he_uniform")(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)
	
	# 3nd Block
    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer="he_uniform")(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
	cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

	# 4th Block
    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer="he_uniform")(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
	
	# 5th Block
    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer="he_uniform")(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)

    # 6th Block
	cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer="he_uniform")(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
	cnn = FullGatedConv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
	cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
	
	# 7th Block
	cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer="he_uniform")(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
	cnn = FullGatedConv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
	
	# 8th Block
	cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer="he_uniform")(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
	cnn = FullGatedConv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
	
	# 9th Block
	cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer="he_uniform")(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
	cnn = FullGatedConv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
	
	cnn1 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding="valid")(cnn)
	cnn1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn1)
	
	cnn2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
	cnn2 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding="valid")(cnn2)
	
	cnn = Add()([cnn1, cnn2])

    shape = cnn.get_shape()
    x = Reshape((shape[1], shape[2] * shape[3]))(cnn)
    
	# Recurrent Block
	
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
	x = Dropout(rate=0.4)(blstm)
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
	x = Dropout(rate=0.2)(blstm)

    x = keras.layers.Dense(len(characters) + 2, activation="softmax", name="dense2")(x) # (None, 32, digits+chars+(space+e))
	
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, the_labels, input_length, label_length])
	
    model = keras.models.Model(inputs=[input_img, the_labels, input_length, label_length], outputs=loss_out, name="handwriting_recognizer")
	
    optimizer_name = keras.optimizers.Adam()
	
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = optimizer_name, metrics=['accuracy'])
	
    return model

model = build_model()
model.summary(line_length=160)

# model to be used at test time # with 10 epoch CER:16.51, WER:42.69/  with autocorrect: CER:16.76 WER: 32.53
# using more data on train db: with 10 epoch, same model: CER:16:33, WER:37.85/ with autocorrect: CER:15.77 WER:29.43 
def decode_batch_predictions(prediction):
  results = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],greedy=True)[0][0])
  output_text = []
  for i, x in enumerate(results):
    for p in x:
      if (int(p) != -1 and int(p)<79):
        output_text.append(characters[int(p)])
  return output_text
def calculate_edit_distance(labels, predictions):
    saprse_labels = tf.cast(tf.sparse.from_dense(np.expand_dims(labels, 0)), dtype=tf.int64)
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.backend.ctc_decode(predictions, input_length=input_len, greedy=True)[0][0][:, :max_len]
    sparse_predictions = tf.cast(tf.sparse.from_dense(predictions_decoded), dtype=tf.int64)
    edit_distances = tf.edit_distance(sparse_predictions, saprse_labels, normalize=False)
    return tf.reduce_mean(edit_distances)      

class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model
    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []
        CER=[]
        WER=[]
        for i in range(len(valid_images)):
            labels = valid_padded_label[i]
            predictions = self.prediction_model.predict(np.expand_dims(valid_images[i], 0))
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())
        for i in (random.sample(range(len(valid_original_label)), batch_size//2)):
          c = cer(valid_original_label[i], (''.join(decode_batch_predictions(prediction_model.predict(np.expand_dims(valid_images[i], 0))))))
          w = wer(valid_original_label[i], (''.join(decode_batch_predictions(prediction_model.predict(np.expand_dims(valid_images[i], 0))))))
          CER.append(c)
          WER.append(w)
        print(f"CER and WER on epoch {epoch + 1}:")
        print(f"Mean CER for {batch_size//2} random samples: {np.mean(np.asarray(CER)*100)}")
        print(f"Mean WER for {batch_size//2} random samples: {np.mean(np.asarray(WER)*100)}")
        print(f"ground_truth: {valid_original_label[i]} --> predicted: {(''.join(decode_batch_predictions(prediction_model.predict(np.expand_dims(valid_images[i], 0)))))}")
        print(f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}")
        print(f"************************************************************************************")
model = build_model()
prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
edit_distance_callback = EditDistanceCallback(prediction_model)
batch_size = 1
epochs = 10000
e = str(epochs)
filepath="{}o-{}e-{}t-{}v.hdf5".format('adam', str(epochs), str(n(train_images).shape[0]), str(n(valid_images).shape[0]))
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, cooldown=0, min_lr=1e-8)
early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=0.005, patience=40, monitor="val_loss", restore_best_weights=True)
callbacks_list = [reduce_lr, early_stopping]
train_images = n(train_images)
train_padded_label = n(train_padded_label)
train_input_img_length = n(train_input_img_length)
train_label_length = n(train_label_length)

valid_images = n(valid_images)
valid_padded_label = n(valid_padded_label)
valid_input_img_length = n(valid_input_img_length)
valid_label_length = n(valid_label_length)

x = [train_images, train_padded_label, train_input_img_length, train_label_length]
y = np.zeros(len(train_images))
validation_data = ([valid_images, valid_padded_label, valid_input_img_length, valid_label_length], [np.zeros(len(valid_images))])
start = time.time()
print(f"Training process...")
history = model.fit(x=x,y=y, batch_size=batch_size, epochs=epochs, validation_data=validation_data, verbose=1, callbacks=callbacks_list)
end = time.time()
time = end-start
m, s = divmod(time, 60)
h, m = divmod(m, 60)
print(f"Total time duration for training: {h} Hours and {m} Minutes and {s} Seconds")
print("---")
print("Computing Character Error Rate and Word Error Rate...")
CER=[]
WER=[]
for i in range(len(valid_original_label)):
  c = cer(valid_original_label[i], (''.join(decode_batch_predictions(prediction_model.predict(np.expand_dims(valid_images[i], 0))))))
  w = wer(valid_original_label[i], (''.join(decode_batch_predictions(prediction_model.predict(np.expand_dims(valid_images[i], 0))))))
  CER.append(c)
  WER.append(w)
  if i%150==0:
    print(f"CER till here: {i}/{len(valid_original_label)} sample is <{np.mean(np.asarray(CER)*100)}> and corresponding WER is <{np.mean(np.asarray(WER)*100)}>")
print(f"CER: {np.mean(np.asarray(CER)*100)}")
print(f"WER: {np.mean(np.asarray(WER)*100)}")
print(f"Training process finished. {h} Hours and {m} Minutes and {s} Seconds")

print(f"********************************************")
print(f"auto correct")
CER=[]
WER=[]
for i in range(len(valid_original_label)):
  c = cer(valid_original_label[i], spell(listToString(''.join(decode_batch_predictions(prediction_model.predict(np.expand_dims(valid_images[i], 0)))))))
  w = wer(valid_original_label[i], spell(listToString(''.join(decode_batch_predictions(prediction_model.predict(np.expand_dims(valid_images[i], 0)))))))
  CER.append(c)
  WER.append(w)
  if i%50==0:
    print(f"CER till here: {i}/{len(valid_original_label)} sample is <{np.mean(np.asarray(CER)*100)}> and corresponding WER is <{np.mean(np.asarray(WER)*100)}>")
print(f"CER with postprocessing: {np.mean(np.asarray(CER)*100)}")
print(f"WER with postprocessing: {np.mean(np.asarray(WER)*100)}")


def plotgraph_loss(epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
def plotgraph_acc(epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
print(f"Plot for model loss")
plotgraph_loss(epochs, loss, val_loss)
print(f"Plot for model accuracy")
plotgraph_acc(epochs, acc, val_acc)
