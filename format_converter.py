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
import itertools

import PIL.Image as Image

from utils import get_traceback

OFFSET = 1000

class ColorGenerator():
    def __init__(self, categories):
        self.taken_colors = set([0, 0, 0])
        self.categories = categories
        for category in self.categories.values():
            if category['isthing'] == 0:
                self.taken_colors.add(tuple(category['color']))

    def get_color(self, cat_id):
        def random_color(base, max_dist=30):
            new_color = base + np.random.randint(low=-max_dist, high=max_dist+1, size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        category = self.categories[cat_id]
        if category['isthing'] == 0:
            return category['color']
        base_color_array = category['color']
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color_array)
                if color not in self.taken_colors:
                     self.taken_colors.add(color)
                     return color


def rgb2id(color):
    return color[0] + 256 * color[1] + 256 * 256 * color[2]


@get_traceback
def convert_single_core(proc_id, image_set, categories, source_folder, segmentations_folder, VOID=0):
    annotations = []
    for working_idx, image_info in enumerate(image_set):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images converted'.format(proc_id, working_idx, len(image_set)))

        file_name = '{}.png'.format(image_info['file_name'].rsplit('.')[0])
        try:
            original_format = np.array(Image.open(os.path.join(source_folder, file_name)), dtype=np.uint32)
        except FileNotFoundError:
            raise KeyError('no prediction png file for id: {}'.format(image_info['id']))

        pan = OFFSET * original_format[:, :, 0] + original_format[:, :, 1]
        pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)

        color_gererator = ColorGenerator(categories)

        l = np.unique(pan)
        segm_info = []
        for el in l:
            sem = el // OFFSET
            if sem == VOID:
                continue
            if sem not in categories:
                raise KeyError('Unknown semantic label {}'.format(sem))
            mask = pan == el
            color = color_gererator.get_color(sem)
            pan_format[mask] = color
            segm_info.append({"id": rgb2id(color),
                              "category_id": sem})

        annotations.append({'image_id': image_info['id'],
                            'file_name': file_name,
                            "segments_info": segm_info})

        Image.fromarray(pan_format).save(os.path.join(segmentations_folder, file_name))
    print('Core: {}, all {} images processed'.format(proc_id, len(image_set)))
    return annotations


def converter(source_folder, images_json_file, categories_json_file,
              segmentations_folder, predictions_json_file,
              VOID=0):
    start_time = time.time()

    print("Reading image set information from {}".format(images_json_file))
    with open(images_json_file, 'r') as f:
        d_coco = json.load(f)
    images = d_coco['images']
    with open(categories_json_file, 'r') as f:
        categories_coco = json.load(f)
    categories = {el['id']: el for el in categories_coco}

    print("CONVERTING...")
    print("2 channels PNG panoptic format:")
    print("\tSource folder: {}".format(source_folder))
    print("TO")
    print("COCO format:")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(predictions_json_file))
    cpu_num = multiprocessing.cpu_count()
    images_split = np.array_split(images, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(images_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, image_set in enumerate(images_split):
        p = workers.apply_async(convert_single_core,
                                (proc_id, image_set, categories, source_folder, segmentations_folder, VOID))
        processes.append(p)
    annotations = []
    for p in processes:
        annotations.extend(p.get())

    print("Writing final JSON in {}".format(predictions_json_file))
    d_coco['annotations'] = annotations
    with open(predictions_json_file, 'w') as f:
        json.dump(d_coco, f)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', type=str,
                        help="folder that contains predictions in 2 channels PNG format")
    parser.add_argument('--images_json_file', type=str,
                        help="JSON file with correponding image set information")
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories informtation",
                        default='./panoptic_coco_categories.json')
    parser.add_argument('--segmentations_folder', type=str,
                        help="Folder with resulting COCO format segmentations")
    parser.add_argument('--predictions_json_file', type=str,
                        help="JSON file with resulting COCO format prediction")
    parser.add_argument('-v', '--void', type=int, default=0,
                        help="id that corresponds to VOID region in two channels PNG format")
    args = parser.parse_args()
    if not os.path.exists(args.segmentations_folder):
        os.mkdir(args.segmentations_folder)
    converter(args.source_folder, args.images_json_file, args.categories_json_file,
              args.segmentations_folder, args.predictions_json_file,
              args.void)
