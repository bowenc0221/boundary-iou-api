from collections import defaultdict
import multiprocessing
import math
import time
from pathlib import Path
import json

import cv2
import numpy as np

import pycocotools.mask as mask_utils

# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


# COCO/LVIS related util functions, to get the boundary for every annotations.
def augment_annotations_with_boundary_single_core(proc_id, annotations, ann_to_mask, dilation_ratio=0.02):
    new_annotations = []

    for ann in annotations:
        mask = ann_to_mask(ann)
        # Find mask boundary.
        boundary = mask_to_boundary(mask, dilation_ratio)
        # Add boundary to annotation in RLE format.
        ann['boundary'] = mask_utils.encode(
            np.array(boundary[:, :, None], order="F", dtype="uint8"))[0]
        new_annotations.append(ann)

    return new_annotations


def augment_annotations_with_boundary_multi_core(annotations, ann_to_mask, dilation_ratio=0.02):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(annotations, cpu_num)
    print("Number of cores: {}, annotations per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []

    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(augment_annotations_with_boundary_single_core,
                                (proc_id, annotation_set, ann_to_mask, dilation_ratio))
        processes.append(p)

    new_annotations = []
    for p in processes:
        new_annotations.extend(p.get())

    workers.close()
    workers.join()

    return new_annotations


def add_boundary_multi_core(coco, cpu_num=16, dilation_ratio=0.02):
    if coco.get_boundary == True:
        print('Found existing boundaries, skipping...')
        return

    print('Adding `boundary` to annotation.')
    tic = time.time()
    cpu_num = min(cpu_num, multiprocessing.cpu_count())

    annotations = coco.dataset["annotations"]
    annotations_split = np.array_split(annotations, cpu_num)
    print("Number of cores: {}, annotations per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []

    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(augment_annotations_with_boundary_single_core,
                                (proc_id, annotation_set, coco.annToMask, dilation_ratio))
        processes.append(p)

    new_annotations = []
    for p in processes:
        new_annotations.extend(p.get())

    workers.close()
    workers.join()

    coco.dataset["annotations"] = new_annotations
    coco.createIndex()
    coco.get_boundary = True
    print('`boundary` added! (t={:0.2f}s)'.format(time.time()- tic))


def coco_add_boundaries_and_save_as_annotation_file(annotation_file: str, stem_addon='_b'):
    # NOTE: we only need coco at the moment, feel free to generalize this function
    from ..coco_instance_api.coco import COCO

    annotation_file = Path(annotation_file).expanduser().resolve()
    coco = COCO(annotation_file)
    add_boundary_multi_core(coco)

    with open(annotation_file, 'r') as f:
        coco_json = json.load(f)

    # boundary counts are byte strings, we need to decode them so they are json serializable
    anns_b = list(coco.anns.values())
    for a in anns_b:
        a['boundary']['counts'] = a['boundary']['counts'].decode("utf-8")
    coco_json['annotations'] = anns_b

    new_annotation_file = annotation_file.parent / (annotation_file.stem + stem_addon + annotation_file.suffix)
    with open(new_annotation_file, 'w', encoding='utf-8') as f:
        json.dump(coco_json, f)

    print(f"\nAdded boundaries to ({annotation_file.parent}) '{annotation_file.name}' and saved as: '{new_annotation_file.name}'")
