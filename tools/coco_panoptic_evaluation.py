"""
Evaluation for COCO panoptic val:
python ./tools/coco_panoptic_evaluation.py \
    --gt_json_file PANOPTIC_GT_JSON \
    --gt_folder PANOPTIC_GT_DIR \
    --pred_json_file PANOPTIC_PRED_JSON \
    --pred_folder PANOPTIC_PRED_DIR \
    --iou-type boundary
"""
import argparse
import contextlib
import io
import tempfile
from tabulate import tabulate

from boundary_iou.coco_panoptic_api.evaluation import pq_compute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground turth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument("--iou-type", default="segm")
    parser.add_argument("--dilation-ratio", default="0.020", type=float)
    args = parser.parse_args()
    assert args.iou_type in ["segm", "boundary"]
    print(args)

    gt_json = args.gt_json_file
    gt_folder = args.gt_folder
    pred_json = args.pred_json_file
    pred_dir = args.pred_folder

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            gt_json,
            pred_json,
            gt_folder=gt_folder,
            pred_folder=pred_dir,
            iou_type=args.iou_type,
            dilation_ratio=args.dilation_ratio,
        )
                
    _print_panoptic_results(pq_res)


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".1f", stralign="center", numalign="center"
    )
    print("Panoptic Evaluation Results:\n" + table)


if __name__ == '__main__':
    main()
