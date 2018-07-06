#!/usr/bin/env python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import argparse
import numpy as np
import json
import time
import multiprocessing

import PIL.Image as Image

from utils import get_traceback

try:
    # set up path for pycocotools
    # sys.path.append('./cocoapi-master/PythonAPI/')
    from pycocotools import mask as COCOmask
except:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")

@get_traceback
def extract_instance_single_core(proc_id, annotations_set, categories, segmentations_folder, instance_json_file):
    annotations_instance = []
    for working_idx, annotation in enumerate(annotations_set):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, working_idx, len(annotations_set)))

        file_name = '{}.png'.format(annotation['file_name'].rsplit('.')[0])
        try:
            pan_format = np.array(Image.open(os.path.join(segmentations_folder, file_name)), dtype=np.uint32)
        except FileNotFoundError:
            raise KeyError('no prediction png file for id: {}'.format(annotation['id']))

        pan = pan_format[:, :, 0] + 256 * pan_format[:, :, 1] + 256 * 256 * pan_format[:, :, 2]

        for segm_info in annotation['segments_info']:
            if categories[segm_info['category_id']]['isthing'] != 1:
                continue
            mask = (pan == segm_info['id']).astype(np.uint8)
            mask = np.expand_dims(mask, axis=2)
            segm_info.pop('id')
            segm_info['image_id'] = annotation['image_id']
            segm_info['segmentation'] = COCOmask.encode(np.asfortranarray(mask))[0]
            annotations_instance.append(segm_info)

    print('Core: {}, all {} images processed'.format(proc_id, len(annotations_set)))
    return annotations_instance


def extract_instance(json_file, segmentations_folder, instance_json_file):
    start_time = time.time()

    print("Reading annotation information from {}".format(json_file))
    with open(json_file, 'r') as f:
        d_coco = json.load(f)
    annotations_panoptic = d_coco['annotations']
    categories = {el['id']: el for el in d_coco['categories']}

    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(annotations_panoptic, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotations_set in enumerate(annotations_split):
        p = workers.apply_async(extract_instance_single_core,
                                (proc_id, annotations_set, categories, segmentations_folder, instance_json_file))
        processes.append(p)
    annotations_instance = []
    for p in processes:
        annotations_instance.extend(p.get())
    for idx, ann in enumerate(annotations_instance):
        ann['id'] = idx

    d_coco['annotations'] = annotations_instance
    categories_instance = []
    for category in d_coco['categories']:
        if category['isthing'] != 1:
            continue
        category.pop('isthing')
        categories_instance.append(category)
    d_coco['categories'] = categories_instance
    with open(instance_json_file, 'w') as f:
        json.dump(d_coco, f)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str,
                        help="JSON file with panoptic data")
    parser.add_argument('--segmentations_folder', type=str, default=None,
                        help="Folder with panoptic COCO format segmentations. Default: 'segmentations' folder in th same location as json_file.")
    parser.add_argument('--instance_json_file', type=str,
                        help="JSON file for things in COCO insatnce segmentation format")
    args = parser.parse_args()
    extract_instance(args.json_file, args.segmentations_folder, args.instance_json_file)
