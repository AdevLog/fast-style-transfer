# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import imageio
import cv2
from PIL import Image

import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
# import tensorflow_hub as hub

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  # image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # image_path = image_url
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  # img = tf.io.decode_image(
  #      tf.io.read_file(image_path),
  #      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  # img = tf.io.read_file(image_path)
  # img = tf.image.decode_jpeg(img, channels=3)
  # tf.image.convert_image_dtype(img, dtype=tf.float32)
  image_raw_data = os.path.basename(image_url)[-128:]
  img_data = tf.image.decode_jpeg(image_raw_data)
  img = tf.image.convert_image_dtype(img_data,dtype=tf.float32)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

# content_image = cv2.imread('Guishan_Island.jpg')
# style_image = cv2.imread('examples/style/wave.jpg')
# stylized_image = cv2.imread('wave_Guishan_Island.jpg')

content_image = 'Guishan_Island.jpeg'
style_image = '/examples/style/wave.jpeg'
stylized_image = 'wave_Guishan_Island.jpeg'

content_image = load_image(content_image, (384, 384))
style_image = load_image(style_image, (256, 256))
stylized_image = load_image(stylized_image, (384, 384))

images = [content_image, style_image, stylized_image]
titles=['Original content image', 'Style image', 'Stylized image']
n = len(images)
image_sizes = [image.shape[1] for image in images]
w = (image_sizes[0] * 6) // 320
plt.figure(figsize=(w * n, w))
gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
for i in range(n):
  plt.subplot(gs[i])
  plt.imshow(images[i][0], aspect='equal')
  plt.axis('off')
  plt.title(titles[i] if len(titles) > i else '')
plt.show()
