"""
Evaluation for LVIS val:
python ./tools/lvis_instance_evaluation.py \
    --gt-json-file LVIS_GT_JSON \
    --dt-json-file LVIS_DT_JSON \
    --iou-type boundary

Evaluation for COCO val2017:
python ./tools/lvis_instance_evaluation.py \
    --max-dets 100 \
    --gt-json-file COCO_GT_JSON \
    --dt-json-file COCO_DT_JSON \
    --iou-type boundary

Evaluation for COCOfiedLVIS val2017:
python ./tools/lvis_instance_evaluation.py \
    --max-dets 100 \
    --gt-json-file COCOFIED_LVIS_GT_JSON \
    --dt-json-file COCO_DT_JSON \
    --iou-type boundary
"""
import argparse
import os

import numpy as np

from boundary_iou.lvis_instance_api.eval import LVISEval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json-file", default="")
    parser.add_argument("--dt-json-file", default="")
    parser.add_argument("--iou-type", default="segm")
    parser.add_argument("--dilation-ratio", default="0.020", type=float)
    parser.add_argument("--lvis", action='store_true')
    parser.add_argument("--max-dets", default='300', type=int)
    args = parser.parse_args()
    print(args)

    annFile = args.gt_json_file
    resFile = args.dt_json_file
    dilation_ratio = args.dilation_ratio
    lvisMaskEval = LVISEval(lvis_gt=annFile, lvis_dt=resFile, iou_type=args.iou_type, max_dets=args.max_dets, dilation_ratio=dilation_ratio)
    lvisMaskEval.evaluate()
    lvisMaskEval.accumulate()
    lvisMaskEval.summarize()
    lvisMaskEval.print_results()


if __name__ == '__main__':
    main()
