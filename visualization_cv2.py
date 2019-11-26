 #!/usr/bin/env python2
'''
Inspired from visualization.py:

Visualization demo for panoptic COCO sample_data using opencv.

The code shows an example of color generation for panoptic data (with
"generate_new_colors" set to True). For each segment distinct color is used in
a way that it close to the color of corresponding semantic class.

Example usage:
    python visualization_cv2.py \
    --json_file ./sample_data/panoptic_examples.json \
    --img_folder ./sample_data/input_images/ \
    --win_scale=0.5 \
    --mode transparency

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json
import argparse
import glob

import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from panopticapi.utils import IdGenerator, rgb2id

# whether from the PNG are used or new colors are generated
generate_new_colors = True


def get_parser():
    parser = argparse.ArgumentParser(description="Visualize predictions from a "
                                     "folder")
    parser.add_argument("--json_file",
                        type=str,
                        help="Path to the json file corresponding \
                        to the panoptic seg. predictions.",
                        default="./sample_data/panoptic_examples.json")
    parser.add_argument("--segmentations_folder",
                        type=str,
                        default=None,
                        help="Folder with panoptic COCO format segmentations. \
                        Default: X if input_json_file is X.json")
    parser.add_argument("--img_folder",
                        type=str,
                        help="Path to the corresponding images. ",
                        default="./sample_data/panoptic_examples/")
    parser.add_argument("--panoptic_coco_categories",
                        type=str,
                        default = "./panoptic_coco_categories.json",
                        help="Path to the json containing the categories ")
    parser.add_argument("--win_scale",
                        type=float,
                        default=1,
                        help="Scale factor for displayed window compared \
                        to the original image \
                        (do not set to keep the original image size)")
    parser.add_argument("--mode",
                        choices=["transparency", "side2side"],
                        default="side2side",
                        help="Display mode: can be side to side or segmentation \
                        on top of the original image (transparency)")
    parser.add_argument("--out_folder",
                        default=None,
                        help="If set, images are saved in out_folder instead of \
                        being displayed")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.out_folder is None:
        cv2.namedWindow("0")

    json_file = args.json_file
    if args.out_folder is not None:
        os.makedirs(args.out_folder, exist_ok=True)

    if args.segmentations_folder is None:
        segmentations_folder = json_file.rsplit('.', 1)[0]
    else:
        segmentations_folder = args.segmentations_folder
    img_folder = args.img_folder
    panoptic_coco_categories = args.panoptic_coco_categories

    with open(json_file, 'r') as f:
        coco_d = json.load(f)

    with open(panoptic_coco_categories, 'r') as f:
        categories_list = json.load(f)

    categories = {category['id']: category for category in categories_list}

    try:
        annotations = sorted(coco_d['annotations'],
                             key=lambda k: int(k['image_id']))
    except:
        annotations = coco_d['annotations']
    nb_annotations = len(annotations)
    for ii, ann in enumerate(annotations):
        # find input img that correspond to the annotation
        img = None
        for image_info in coco_d['images']:
            if image_info['id'] == ann['image_id']:
                try:
                    if os.path.isfile(os.path.join(img_folder, image_info['file_name'])):
                        p = os.path.join(img_folder, image_info['file_name'])
                    else:
                        print("Image {} not found. trying with other extensions".format(image_info['file_name']))
                        p = glob.glob( os.path.join(img_folder, "{}.*".format(image_info['id'])))[0]

                    img = np.array(Image.open(p))[:, :, ::-1]  # BGR for opencv visualisation
                except:
                    print("Unable to find corresponding input image.")
                break

        segmentation = np.array(
            Image.open(os.path.join(segmentations_folder, ann['file_name'])),
            dtype=np.uint8
        )
        segmentation_id = rgb2id(segmentation)
        # find segments boundaries
        boundaries = find_boundaries(segmentation_id, mode='thick')

        if generate_new_colors:
            segmentation[:, :, :] = 0
            color_generator = IdGenerator(categories)
            for segment_info in ann['segments_info']:
                color = color_generator.get_color(segment_info['category_id'])
                mask = segmentation_id == segment_info['id']
                segmentation[mask] = color

        # depict boundaries
        segmentation[boundaries] = [0, 0, 0]
        segmentation = segmentation[:, :, ::-1]  # BGR for opencv display
        if img is None:
            vis = segmentation
        else:
            if args.mode == "side2side":
                h1, w1 = img.shape[:2]
                h2, w2 = segmentation.shape[:2]
                vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
                vis[:h1, :w1] = img
                vis[:h2, w1:w1+w2] = segmentation
            elif args.mode == "transparency":
                vis = np.uint8(img / 2 + segmentation / 2)

            else:
                raise NotImplementedError(
                    "Display mode {} not recognised".format(args.mode)
                )

        vis = cv2.resize(vis, dsize=(0, 0), fx=args.win_scale, fy=args.win_scale)
        if args.out_folder is None:
            cv2.imshow("0", vis)
            cv2.waitKey(-1)

        else:
            if type(ann["image_id"]) == str:
                out_img_name = '{}.png'.format(ann["image_id"])
            else:
                out_img_name = '{0:06d}.png'.format(ann["image_id"])
            cv2.imwrite(os.path.join(args.out_folder, out_img_name), vis)
