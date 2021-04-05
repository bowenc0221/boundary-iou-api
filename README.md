# Boundary IoU API (Beta version)

Bowen Cheng, Ross Girshick, Piotr Dollár, Alexander C. Berg, Alexander Kirillov

[[`arXiv`](https://arxiv.org/abs/2103.16562)] [[`Project`](https://bowenc0221.github.io/boundary-iou/)] [[`BibTeX`](#CitingBoundaryIoU)]

This API is an experimental version of Boundary IoU for 5 datasets:
- [COCO instance segmentation](https://cocodataset.org/#detection-eval)
- [LVIS instance segmentation](https://www.lvisdataset.org/)
- [Cityscapes instance segmentation](https://www.cityscapes-dataset.com/benchmarks/)
- [COCO panoptic segmentation](https://cocodataset.org/#panoptic-eval)
- [Cityscapes panoptic segmentation](https://www.cityscapes-dataset.com/benchmarks/)

To install Boundary IoU API, run:
```bash
pip install git+https://github.com/bowenc0221/boundary-iou-api.git
```
or
```bash
git clone git@github.com:bowenc0221/boundary-iou-api.git
cd boundary_iou_api
pip install -e .
```

OpenCV is required to use Boundary IoU API.

## Summary of usage
We provide two ways to use this api, you can either replace imports with our api or do offline evaluation.

### Replacing imports
Our Boundary IoU API supports both evaluation with Mask IoU and Boundary IoU with the same interface as original ones.
Thus, you only need to change the import, without worried about breaking your existing code.

1. COCO instance segmentation  
    replace
    ```python
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ```
    with
    ```python
    from boundary_iou.coco_instance_api.coco import COCO
    from boundary_iou.coco_instance_api.cocoeval import COCOeval
    ```
    and set
    ```python
    COCOeval(..., iouType="boundary")
    ```


2. LVIS instance segmentation  
    replace
    ```python
    from lvis import LVISEval
    ```
    with
    ```python
    from boundary_iou.lvis_instance_api.eval import LVISEval
    ```
    and set
    ```python
    LVISEval(..., iou_type="boundary")
    ```


3. Cityscapes instance segmentation  
    replace
    ```python
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval
    ```
    with
    ```python
    import boundary_iou.cityscapes_instance_api.evalInstanceLevelSemanticLabeling as cityscapes_eval
    ```
    and set
    ```python
    cityscapes_eval.args.iou_type = "boundary"
    ```


4. COCO panoptic segmentation  
    replace
    ```python
    from panopticapi.evaluation import pq_compute
    ```
    with
    ```python
    from boundary_iou.coco_panoptic_api.evaluation import pq_compute
    ```
    and set
    ```python
    pq_compute(..., iou_type="boundary")
    ```


5. Cityscapes panoptic segmentation  
    replace
    ```python
    from cityscapesscripts.evaluation.evalPanopticSemanticLabeling as evaluatePanoptic
    ```
    with
    ```python
    from boundary_iou.cityscapes_panoptic_api.evalPanopticSemanticLabeling import evaluatePanoptic
    ```
    and set
    ```python
    evaluatePanoptic(..., iou_type="boundary")
    ```

### Offline evaluation
We also provide evaluation code that can evaluates your prediction files for each dataset.

1. COCO instance segmentation  
    ```bash
    python ./tools/coco_instance_evaluation.py \
        --gt-json-file COCO_GT_JSON \
        --dt-json-file COCO_DT_JSON \
        --iou-type boundary
    ```


2. LVIS instance segmentation  
    ```bash
    python ./tools/lvis_instance_evaluation.py \
        --gt-json-file LVIS_GT_JSON \
        --dt-json-file LVIS_DT_JSON \
        --iou-type boundary
    ```


3. Cityscapes instance segmentation  
    ```bash
    python ./tools/cityscapes_instance_evaluation.py \
        --gt_dir GT_DIR \
        --result_dir RESULT_DIR \
        --iou-type boundary
    ```


4. COCO panoptic segmentation  
    ```bash
    python ./tools/coco_panoptic_evaluation.py \
        --gt_json_file PANOPTIC_GT_JSON \
        --gt_folder PANOPTIC_GT_DIR \
        --pred_json_file PANOPTIC_PRED_JSON \
        --pred_folder PANOPTIC_PRED_DIR \
        --iou-type boundary
    ```


5. Cityscapes panoptic segmentation  
    ```bash
    python ./tools/cityscapes_panoptic_evaluation.py \
        --gt_json_file PANOPTIC_GT_JSON \
        --gt_folder PANOPTIC_GT_DIR \
        --pred_json_file PANOPTIC_PRED_JSON \
        --pred_folder PANOPTIC_PRED_DIR \
        --iou-type boundary
    ```

## <a name="CitingBoundaryIoU"></a>Citing Boundary IoU
If you find Boundary IoU helpful in your research or wish to refer to the referenced results, please use the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021boundary,
  title={Boundary {IoU}: Improving Object-Centric Image Segmentation Evaluation},
  author={Bowen Cheng and Ross Girshick and Piotr Doll{\'a}r and Alexander C. Berg and Alexander Kirillov},
  booktitle={CVPR},
  year={2021}
}
```

## Contact
If you have any questions regarding this API, please contact us at bcheng9 AT illinois.edu
