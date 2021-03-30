"""
Evaluation for Cityscapes val:
python ./tools/cityscapes_instance_evaluation.py \
    --gt_dir GT_DIR \
    --result_dir RESULT_DIR \
    --iou-type boundary
"""
import argparse
import os
import glob

from fvcore.common.file_io import PathManager

import boundary_iou.cityscapes_instance_api.evalInstanceLevelSemanticLabeling as cityscapes_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str,
                        help="Gt dir")
    parser.add_argument('--result_dir', type=str,
                        help="Result dir")
    parser.add_argument("--iou-type", default="segm")
    parser.add_argument("--dilation-ratio", default="0.005", type=float)
    parser.add_argument('--clear-results', action="store_true",
                        help="Remove gtInstances.json file from previous evaluation.")
    args = parser.parse_args()
    assert args.iou_type in ["segm", "boundary"]
    print(args)

    if args.gt_dir is None:
        raise ValueError('Must provide cityscapes path for evaluation.')

    print("Evaluating results under {} ...".format(args.result_dir))

    # set some global states in cityscapes evaluation API, before evaluating
    cityscapes_eval.args.predictionPath = os.path.abspath(args.result_dir)
    cityscapes_eval.args.predictionWalk = None
    cityscapes_eval.args.JSONOutput = False
    cityscapes_eval.args.colorized = False
    cityscapes_eval.args.gtInstancesFile = os.path.join(args.result_dir, "gtInstances.json")
    cityscapes_eval.args.iou_type = args.iou_type
    cityscapes_eval.args.dilation_ratio = args.dilation_ratio

    if args.clear_results:
        if os.path.exists(cityscapes_eval.args.gtInstancesFile):
            os.remove(cityscapes_eval.args.gtInstancesFile)
            print("Remove previous gtInstanceFile from {}".format(cityscapes_eval.args.gtInstancesFile))

    # These lines are adopted from
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
    gt_dir = PathManager.get_local_path(args.gt_dir)
    groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_instanceIds.png"))
    assert len(
        groundTruthImgList
    ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
        cityscapes_eval.args.groundTruthSearch
    )
    predictionImgList = []
    for gt in groundTruthImgList:
        predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
    results = cityscapes_eval.evaluateImgLists(
        predictionImgList, groundTruthImgList, cityscapes_eval.args
    )["averages"]

    print(results)


if __name__ == '__main__':
    main()
