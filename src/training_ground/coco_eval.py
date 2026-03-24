import io
from contextlib import redirect_stdout

import faster_coco_eval

faster_coco_eval.init_as_pycocotools()
import numpy as np
from pycocotools.cocoeval import COCOeval  # noqa: E402


def run_coco_eval(coco_gt, results: list[dict], iou_type: str) -> dict[str, object]:
    if not results:
        return {
            "stats": None,
            "per_class_ap50": {},
            "per_class_ap": {},
        }

    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    capture = io.StringIO()
    with redirect_stdout(capture):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval["precision"]
    iou_thresholds = evaluator.params.iouThrs
    ap50_index = int(np.argmin(np.abs(iou_thresholds - 0.5)))

    per_class_ap = {}
    per_class_ap50 = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        class_precision = precision[:, :, category_index, 0, -1]
        class_precision = class_precision[class_precision > -1]
        category_id = int(category_id)
        per_class_ap[category_id] = (
            float(class_precision.mean()) if class_precision.size else 0.0
        )

        class_precision_ap50 = precision[ap50_index, :, category_index, 0, -1]
        class_precision_ap50 = class_precision_ap50[class_precision_ap50 > -1]
        per_class_ap50[category_id] = (
            float(class_precision_ap50.mean()) if class_precision_ap50.size else 0.0
        )

    return {
        "stats": [float(value) for value in evaluator.stats.tolist()],
        "per_class_ap50": per_class_ap50,
        "per_class_ap": per_class_ap,
    }
