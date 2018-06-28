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


def extract_semantic_single_core(proc_id, annotations_set, segmentations_folder, semantic_seg_folder):
    for working_idx, annotation in enumerate(annotations_set):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, working_idx, len(annotations_set)))

        file_name = '{}.png'.format(annotation['file_name'].rsplit('.')[0])
        try:
            pan_format = np.array(Image.open(os.path.join(segmentations_folder, file_name)), dtype=np.uint32)
        except FileNotFoundError:
            print('no prediction png file for id: {}'.format(annotation['id']))
            sys.exit(-1)

        pan = pan_format[:, :, 0] + 256 * pan_format[:, :, 1] + 256 * 256 * pan_format[:, :, 2]
        semantic = np.zeros(pan.shape, dtype=np.uint8)

        for segm_info in annotation['segments_info']:
            semantic[pan == segm_info['id']] = segm_info['category_id']

        Image.fromarray(semantic).save(os.path.join(semantic_seg_folder, file_name))
    print('Core: {}, all {} images processed'.format(proc_id, len(annotations_set)))


def extract_semantic(json_file, segmentations_folder, semantic_seg_folder):
    start_time = time.time()

    print("Reading annotation information from {}".format(json_file))
    with open(json_file, 'r') as f:
        d_coco = json.load(f)
    annotations = d_coco['annotations']

    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(annotations, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotations_set in enumerate(annotations_split):
        p = workers.apply_async(extract_semantic_single_core,
                                (proc_id, annotations_set, segmentations_folder, semantic_seg_folder))
        processes.append(p)
    for p in processes:
        p.get()

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str,
                        help="JSON file with panoptic data")
    parser.add_argument('--segmentations_folder', type=str, default=None,
                        help="Folder with panoptic COCO format segmentations. Default: 'segmentations' folder in th same location as json_file.")
    parser.add_argument('--semantic_seg_folder', type=str,
                        help="folder for semnatic segmentation")
    args = parser.parse_args()
    extract_semantic(args.json_file, args.segmentations_folder, args.semantic_seg_folder)
