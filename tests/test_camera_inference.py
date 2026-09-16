import unittest

import numpy as np
import torch
from PIL import Image

from training_ground.camera_inference import (
    IMAGE_SIZE,
    MASK_SIZE,
    NUM_SELECT,
    postprocess,
    prepare_image,
)


def _model_outputs(num_queries: int):
    boxes = torch.zeros((1, num_queries, 4), dtype=torch.float32)
    logits = torch.full((1, num_queries, 5), -100.0, dtype=torch.float32)
    masks = torch.ones((1, num_queries, MASK_SIZE, MASK_SIZE), dtype=torch.float32)
    return boxes, logits, masks


class CameraSelectionTests(unittest.TestCase):
    def test_disabled_classes_do_not_consume_global_top_k_slots(self):
        boxes, logits, masks = _model_outputs(7)
        # All disabled candidates outrank every enabled candidate. The camera
        # removes them before selecting the global top 20.
        logits[0, :, 0] = 100.0
        logits[0, :, 4] = 99.0
        logits[0, :, 1:4] = torch.arange(21, dtype=torch.float32).reshape(7, 3)

        detections = postprocess(boxes, logits, masks, width=4, height=2, threshold=0.0)

        self.assertEqual(len(detections), NUM_SELECT)
        self.assertNotIn(0, detections.class_id)
        self.assertNotIn(4, detections.class_id)
        # The sole enabled candidate outside top-20 is query 0, class 1.
        self.assertEqual(detections.class_id[-1], 2)

    def test_per_class_confidence_boundaries_are_inclusive(self):
        boxes, logits, masks = _model_outputs(2)
        thresholds = torch.tensor([0.9, 0.55, 0.6], dtype=torch.float32)
        logits[0, 0, 1:4] = torch.logit(thresholds)
        logits[0, 1, 1:4] = torch.logit(thresholds) - 1e-4

        detections = postprocess(boxes, logits, masks, width=4, height=2)

        np.testing.assert_array_equal(detections.class_id, [1, 3, 2])
        np.testing.assert_allclose(
            detections.confidence,
            [0.9, 0.6, 0.55],
            rtol=0,
            atol=1e-7,
        )


class CameraMaskTests(unittest.TestCase):
    def test_mask_crops_letterbox_padding_before_resizing(self):
        boxes, logits, masks = _model_outputs(1)
        logits[0, 0, 2] = 10.0
        masks.fill_(-1.0)
        # A 4x2 output maps to the centered 108-row content area in the
        # 216x216 mask. Values outside it are letterbox padding.
        masks[0, 0, 54:162, :] = 1.0

        detections = postprocess(boxes, logits, masks, width=4, height=2)

        np.testing.assert_array_equal(detections.mask[0], np.ones((2, 4), dtype=bool))

    def test_mask_threshold_is_strictly_greater_than_zero_after_interpolation(self):
        boxes, logits, masks = _model_outputs(1)
        logits[0, 0, 2] = 10.0
        masks.zero_()

        detections = postprocess(boxes, logits, masks, width=3, height=2)

        self.assertFalse(detections.mask.any())


class CameraLetterboxTests(unittest.TestCase):
    def test_small_images_upscale_without_nan(self):
        tensor = prepare_image(Image.new("RGB", (32, 16), "white"))
        self.assertTrue(torch.isfinite(tensor).all())
        expected = (torch.ones(3) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
            [0.229, 0.224, 0.225]
        )
        torch.testing.assert_close(tensor[0, :, 216, 216], expected)

    def test_prepare_image_centers_aspect_preserving_resize_on_black(self):
        # Production camera frames are larger than the network input. An
        # 864x432 frame scales to 432x216 with 108 rows of padding per side.
        image = Image.fromarray(np.full((432, 864, 3), 255, dtype=np.uint8), mode="RGB")

        tensor = prepare_image(image)

        self.assertEqual(tuple(tensor.shape), (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        means = torch.tensor([0.485, 0.456, 0.406])
        stds = torch.tensor([0.229, 0.224, 0.225])
        expected_black = -means / stds
        expected_white = (1 - means) / stds
        torch.testing.assert_close(tensor[0, :, 0, 0], expected_black)
        torch.testing.assert_close(tensor[0, :, 107, 200], expected_black)
        torch.testing.assert_close(tensor[0, :, 108, 200], expected_white)
        torch.testing.assert_close(tensor[0, :, 323, 200], expected_white)
        torch.testing.assert_close(tensor[0, :, 324, 200], expected_black)


if __name__ == "__main__":
    unittest.main()
