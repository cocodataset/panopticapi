#!/usr/bin/env python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

json_file = './sample_data/panoptic_example.json'
segmentations_folder = './sample_data/segmentations/'
img_folder = './sample_data/input_images/'

with open(json_file, 'r') as f:
    coco_d = json.load(f)

ann = np.random.choice(coco_d['annotations'])

# find inout img that correspond to the annotation
img = None
for image_info in coco_d['images']:
    if image_info['id'] == ann['image_id']:
        try:
            img = np.array(Image.open(os.path.join(img_folder, image_info['file_name'])))
        except:
            print("Undable to find correspoding input image.")
        break

segmentation = np.array(Image.open(os.path.join(segmentations_folder, ann['file_name'])), dtype=np.uint8)

# depict segments boundaries
segmentation_id = segmentation.astype(np.uint32)
segmentation_id = segmentation_id[:, :, 0] + 256 * segmentation_id[:, :, 1] + 256 * 256 * segmentation_id[:, :, 2]
boundaries = find_boundaries(segmentation_id, mode='thick')
segmentation[boundaries] = [0, 0, 0]

if img is None:
    plt.figure()
    plt.imshow(segmentation)
    plt.axis('off')
else:
    plt.figure(figsize=(9, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(segmentation)
    plt.axis('off')
    plt.tight_layout()
plt.show()
