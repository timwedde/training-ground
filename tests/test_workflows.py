import asyncio
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import torch
import yaml
from PIL import Image
from typer.testing import CliRunner

from training_ground.coco import decode_segmentation, encode_binary_mask
from training_ground.evaluation import run_evaluation
from training_ground.main import app
from training_ground.overlays import render_prediction_overlay
from training_ground.predictors import DetectionBatch, YOLO26PredictorAdapter
from training_ground.training import prepare_yolo26_dataset
from training_ground.training_backends import BOX_TASK, SEGMENTATION_TASK
from training_ground.upload import upload_artifact_bundle


def write_split(
    root,
    split,
    categories,
    annotations,
    filename="nested/sample_name.jpg",
    size=(32, 32),
):
    directory = root / split
    image_path = directory / filename
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(image_path)
    document = {
        "images": [
            {"id": 1, "width": size[0], "height": size[1], "file_name": filename}
        ],
        "categories": [
            {"id": category_id, "name": name, "supercategory": "object"}
            for category_id, name in categories
        ],
        "annotations": annotations,
    }
    path = directory / "_annotations.coco.json"
    path.write_text(json.dumps(document))
    return image_path


def annotation(category_id):
    return {
        "id": 1,
        "image_id": 1,
        "category_id": category_id,
        "bbox": [2, 3, 12, 13],
        "area": 156,
        "iscrowd": 0,
    }


class WorkflowTests(unittest.TestCase):
    def test_rectangular_autobatch_probes_dataset_shape_and_refines_after_oom(self):
        from training_ground.rectangular_training import RectangularSegmentationTrainer

        model = torch.nn.Linear(1, 1)
        model.yaml = {"channels": 3}
        trainer = RectangularSegmentationTrainer.__new__(RectangularSegmentationTrainer)
        trainer.model = model
        trainer.device = torch.device("cuda:0")
        trainer.batch_size = 0.85
        trainer.data = {"train": "train"}
        trainer.args = SimpleNamespace(amp=True, cache=True)
        trainer.amp = True
        dataset = MagicMock()
        dataset.__len__.return_value = 80
        dataset.datasets = [
            SimpleNamespace(
                batch_shapes=[(448, 864), (480, 864)], labels=[{"cls": [0]}]
            )
        ]
        trainer.build_dataset = Mock(return_value=dataset)

        def profile(image, *_args, **_kwargs):
            batch, channels, height, width = image
            self.assertEqual((channels, height, width), (3, 480, 864))
            if batch >= 64:
                raise torch.OutOfMemoryError("simulated OOM")
            return [(None, None, 1 + batch / 2)]

        with (
            torch.backends.cudnn.flags(benchmark=False),
            patch("torch.cuda.mem_get_info", return_value=(32 * 2**30, 32 * 2**30)),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.empty_cache"),
            patch("training_ground.rectangular_training.autocast"),
            patch("torch.empty", side_effect=lambda *shape: shape),
            patch(
                "training_ground.rectangular_training.profile_ops", side_effect=profile
            ),
        ):
            self.assertEqual(trainer.auto_batch(), 52)

    def test_settings_sweep_uses_validation_and_reports_class_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            categories = [(9, "tree"), (20, "pole")]
            val_path = write_split(root, "valid", categories, [annotation(9)])
            test_path = write_split(root, "test", categories, [annotation(9)])
            boxes = np.array([[2, 3, 14, 16], [18, 18, 28, 28], [20, 3, 28, 9]])
            validation = DetectionBatch(
                boxes, np.array([0.4, 0.2, 0.25]), np.array([3, 3, 1]), None
            )
            test = DetectionBatch(boxes[:1], np.array([0.95]), np.array([3]), None)
            model = SimpleNamespace(
                class_id_to_name={3: "tree", 1: "pole"},
                model_settings=lambda: {"num_queries": 50, "image_size": [384, 384]},
                predict=Mock(side_effect=[test, validation]),
            )
            output = run_evaluation(
                root, root / "model.pth", "test", None, 0.5, model=model, task=BOX_TASK
            )
            self.assertEqual(
                [call.args[0] for call in model.predict.call_args_list],
                [str(test_path), str(val_path)],
            )
            report = json.loads((output / "recommended_settings.json").read_text())
            self.assertEqual(
                report["model"]["classes"],
                [{"class_id": 1, "name": "pole"}, {"class_id": 3, "name": "tree"}],
            )
            sweep = report["sweep"]
            self.assertEqual(sweep["split"], "valid")
            self.assertEqual(sweep["recommended"]["confidence"], 0.39)
            self.assertEqual(sweep["recommended"]["f1"], 1.0)
            self.assertIsNone(sweep["per_class"][0]["best"])
            self.assertEqual(sweep["per_class"][1]["best"]["confidence"], 0.39)

    def test_rfdetr_evaluation_uses_model_slots_and_keeps_reports_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split(root, "test", [(2, "group"), (9, "pole")], [annotation(9)])
            checkpoint = root / "checkpoint.pth"
            boxes = np.array([[2, 3, 14, 16]])
            detection = DetectionBatch(boxes, np.array([0.99]), np.array([5]), None)
            empty = DetectionBatch(boxes[:0], np.array([]), np.array([]), None)
            model = SimpleNamespace(
                class_id_to_name={5: "pole"},
                model_settings=dict,
                predict=Mock(side_effect=[detection, empty]),
            )

            first = run_evaluation(
                root, checkpoint, "test", None, 0.5, model=model, task=BOX_TASK
            )
            second = run_evaluation(
                root, checkpoint, "test", None, 0.5, model=model, task=BOX_TASK
            )

            with (first / "per_image_metrics.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                (row["true_positives"], row["false_positives"], row["false_negatives"]),
                ("1", "0", "0"),
            )
            self.assertNotEqual(first, second)
            self.assertEqual(list((first / "false_negatives").rglob("*.jpg")), [])
            self.assertTrue(
                (
                    second / "false_negatives/pole/nested/sample_name.jpg_fn.jpg"
                ).is_file()
            )
            self.assertTrue(
                (first / "overlays/nested/sample_name.jpg_overlay.jpg").is_file()
            )
            summary = json.loads((second / "summary.json").read_text())
            self.assertEqual(summary["coco_metrics"]["bbox"]["stats"][0], 0)

    def test_coco_ap_is_independent_of_operating_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = np.zeros((3, 32, 32), dtype=bool)
            annotations = []
            boxes = []
            for index, x in enumerate((2, 12, 22)):
                masks[index, 2:8, x : x + 6] = True
                boxes.append([x, 2, x + 6, 8])
                annotations.append(
                    annotation(9)
                    | {
                        "id": index + 1,
                        "bbox": [x, 2, 6, 6],
                        "area": 36,
                        "segmentation": encode_binary_mask(masks[index]),
                    }
                )
            image_path = write_split(root, "test", [(9, "tree")], annotations)
            detections = DetectionBatch(
                np.array(boxes),
                np.array([0.6, 0.2, 0.0005], dtype=np.float32),
                np.array([3, 3, 3]),
                masks,
            )
            model = SimpleNamespace(
                class_id_to_name={3: "tree"}, model_settings=dict, predict=Mock()
            )
            for camera_profile, threshold, expected_tp, inference_floor in (
                (False, None, 1, 0.001),
                (False, 0.0, 3, 0.0),
                (True, None, 1, 0.001),
                (True, 0.2, 2, 0.001),
            ):
                with self.subTest(camera_profile=camera_profile, threshold=threshold):
                    model.camera_profile = camera_profile
                    model.predict.reset_mock()
                    model.predict.return_value = detections
                    report = run_evaluation(
                        root, root / "model.pth", "test", threshold, 0.5, model=model
                    )
                    model.predict.assert_called_once_with(
                        str(image_path), threshold=inference_floor
                    )
                    summary = json.loads((report / "summary.json").read_text())
                    self.assertEqual(summary["coco_confidence_floor"], 0.001)
                    for task in ("bbox", "segm"):
                        self.assertAlmostEqual(
                            summary["coco_metrics"][task]["stats"][8], 2 / 3
                        )
                    self.assertAlmostEqual(
                        summary["overall"]["recall"], expected_tp / 3
                    )
                    with (report / "prediction_metrics.csv").open() as handle:
                        self.assertEqual(len(list(csv.DictReader(handle))), expected_tp)

    def test_yolo_dataset_reuses_training_class_order_and_keeps_nested_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            write_split(dataset, "train", [(4, "pole"), (9, "tree")], [annotation(9)])
            for split in ("valid", "test"):
                write_split(dataset, split, [(20, "tree")], [annotation(20)])
            existing = dataset / "yolo26"
            existing.mkdir()
            (existing / "keep.txt").write_text("existing files")

            with patch.dict(os.environ, {"YOLO_CONFIG_DIR": str(root / "settings")}):
                yaml_path = prepare_yolo26_dataset(dataset, root / "prepared", BOX_TASK)
            self.assertEqual(
                yaml.safe_load(yaml_path.read_text())["names"], ["pole", "tree"]
            )
            for split in ("train", "valid", "test"):
                label = yaml_path.parent / f"labels/{split}/nested/sample_name.txt"
                self.assertEqual(label.read_text().split()[0], "1")
                self.assertTrue(
                    (
                        yaml_path.parent / f"images/{split}/nested/sample_name.jpg"
                    ).is_file()
                )
            self.assertEqual((existing / "keep.txt").read_text(), "existing files")

    def test_yolo_segmentation_converts_uncompressed_rle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            counts = [33, 4, 28, 4, 28, 4, 28, 4, 891]
            item = annotation(9) | {
                "segmentation": {"size": [32, 32], "counts": counts}
            }
            for split in ("train", "valid", "test"):
                write_split(dataset, split, [(9, "tree")], [item])
            with patch.dict(os.environ, {"YOLO_CONFIG_DIR": str(root / "settings")}):
                yaml_path = prepare_yolo26_dataset(
                    dataset, root / "prepared", SEGMENTATION_TASK
                )
            label = (
                (yaml_path.parent / "labels/train/nested/sample_name.txt")
                .read_text()
                .split()
            )
            self.assertEqual(
                [float(value) for value in label],
                [0, 0.03125, 0.03125, 0.03125, 0.125, 0.125, 0.125, 0.125, 0.03125],
            )

    def test_yolo_adapter_uses_checkpoint_input_size_and_original_resolution_masks(
        self,
    ):
        adapter = YOLO26PredictorAdapter.__new__(YOLO26PredictorAdapter)
        adapter.tiled = False
        mask = torch.zeros((1, 32, 64), dtype=torch.uint8)
        mask[0, 0, :10] = 1
        result = SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=torch.tensor([[0, 0, 10, 1]]),
                conf=torch.tensor([0.9]),
                cls=torch.tensor([0]),
            ),
            masks=SimpleNamespace(data=mask),
        )
        adapter._model = SimpleNamespace(
            task="segment",
            overrides={"imgsz": 448},
            predict=Mock(return_value=[result]),
        )
        predictions = adapter.predict("frame.jpg", None)
        self.assertEqual(adapter._model.predict.call_args.kwargs["imgsz"], 448)
        self.assertTrue(adapter._model.predict.call_args.kwargs["retina_masks"])
        np.testing.assert_array_equal(predictions.mask, mask.numpy().astype(bool))

    def test_tiled_yolo_evaluates_original_boxes_and_reuses_validation_predictions(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            box = annotation(9) | {"bbox": [570, 470, 30, 30], "area": 900}
            for split in ("train", "valid", "test"):
                write_split(root, split, [(9, "bottle")], [box], size=(1000, 500))
            data = prepare_yolo26_dataset(root, root / "prepared", BOX_TASK, tiled=True)
            with Image.open(
                data.parent / "images/train/nested/sample_name.jpg.png"
            ) as image:
                self.assertEqual(image.size, (1000, 640))
                self.assertEqual(image.getpixel((0, 500)), (114, 114, 114))
            path = data.parent / "labels/train/nested/sample_name.jpg.txt"
            np.testing.assert_allclose(
                [float(value) for value in path.read_text().split()],
                [0, 585 / 1000, 485 / 640, 30 / 1000, 30 / 640],
                atol=1e-6,
            )

            def predict(**kwargs):
                self.assertEqual(len(kwargs["source"]), 2)
                self.assertEqual(kwargs["source"][1].shape, (640, 640, 3))
                self.assertTrue((kwargs["source"][1][500:] == 114).all())
                rows = [
                    [[570, 470, 600, 510, 0.4, 0], [50, 505, 60, 530, 0.9, 0]],
                    [[210, 470, 240, 510, 0.4, 0]],
                ]
                return [
                    SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor(tile)))
                    for tile in rows
                ]

            adapter = YOLO26PredictorAdapter.__new__(YOLO26PredictorAdapter)
            adapter.tiled = True
            adapter.checkpoint_name = "best.pt"
            adapter.class_id_to_name = {0: "bottle"}
            adapter._model = SimpleNamespace(
                predict=Mock(side_effect=predict),
                model=None,
                predictor=None,
                task="detect",
                names=adapter.class_id_to_name,
            )
            output = run_evaluation(
                root,
                root / "best.pt",
                "test",
                None,
                0.5,
                model=adapter,
                backend="yolo26",
                task=BOX_TASK,
                tiled=True,
            )
            self.assertEqual(adapter._model.predict.call_count, 2)
            report = json.loads((output / "recommended_settings.json").read_text())
            self.assertEqual(report["sweep"]["recommended"]["confidence"], 0.39)
            self.assertEqual(report["sweep"]["recommended"]["nms_iou"], 0.5)
            with (output / "per_image_metrics.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["tile_count"], "2")
            self.assertEqual(row["true_positives"], "1")
            self.assertEqual(row["false_positives"], "0")
            summary = json.loads((output / "summary.json").read_text())
            self.assertAlmostEqual(summary["coco_metrics"]["bbox"]["stats"][0], 1.0)

            checkpoint = root / "best.pt"
            checkpoint.touch()
            predictions = root / "predictions"
            with (
                patch("ultralytics.YOLO", return_value=adapter._model),
                patch(
                    "training_ground.directory_prediction.render_prediction_overlay",
                    wraps=render_prediction_overlay,
                ) as render,
            ):
                result = CliRunner().invoke(
                    app,
                    [
                        "predict-dir",
                        str(root / "test"),
                        str(predictions),
                        "--checkpoint-path",
                        str(checkpoint),
                        "--backend",
                        "yolo26",
                        "--task",
                        "boxes",
                        "--tiled",
                        "--threshold",
                        "0.3",
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            render.assert_called_once()
            items = render.call_args.kwargs["pred_items"]
            self.assertEqual([item["bbox"] for item in items], [[570, 470, 600, 500]])
            self.assertTrue((predictions / "bottle/nested/sample_name.jpg").is_file())

    def test_coco_decodes_uncompressed_rle(self):
        result = decode_segmentation({"size": [2, 2], "counts": [1, 2, 1]}, 2, 2)
        np.testing.assert_array_equal(result, [[False, True], [True, False]])

    def test_predict_dir_returns_error_when_roboflow_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (32, 32)).save(source / "frame.jpg")
            checkpoint = root / "model.pth"
            checkpoint.touch()
            empty = DetectionBatch(np.zeros((0, 4)), np.array([]), np.array([]), None)
            model = SimpleNamespace(
                class_id_to_name={0: "pole"}, predict=Mock(return_value=empty)
            )
            project = SimpleNamespace(upload=Mock(return_value=[]))
            workspace = SimpleNamespace(project=Mock(return_value=project))
            roboflow = SimpleNamespace(
                login=Mock(),
                Roboflow=lambda: SimpleNamespace(workspace=lambda: workspace),
            )
            with (
                patch(
                    "training_ground.directory_prediction.create_model",
                    return_value=model,
                ),
                patch.dict("sys.modules", {"roboflow": roboflow}),
            ):
                result = CliRunner().invoke(
                    app,
                    [
                        "predict-dir",
                        str(source),
                        str(root / "output"),
                        "--checkpoint-path",
                        str(checkpoint),
                        "--upload",
                        "--project",
                        "test",
                    ],
                )
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("Uploaded 0 of 1", result.output)
            self.assertTrue((root / "output/no_detections/frame.jpg").is_file())


class UploadTests(unittest.TestCase):
    def test_incomplete_signer_response_leaves_existing_archive_and_starts_no_uploads(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            report.mkdir()
            (report / "summary.json").write_text("{}")
            archive = root / "evaluation_artifacts.zip"
            archive.write_bytes(b"existing archive")
            with (
                patch(
                    "training_ground.upload.get_signed_urls",
                    AsyncMock(return_value=(1, "test/", {})),
                ),
                patch(
                    "training_ground.upload.upload_file_chunked", AsyncMock()
                ) as upload,
                self.assertRaisesRegex(ValueError, "Signed URLs do not match"),
            ):
                asyncio.run(upload_artifact_bundle("test", {archive.name: report}))
            upload.assert_not_awaited()
            self.assertEqual(archive.read_bytes(), b"existing archive")

    def test_failed_upload_waits_for_other_writes_and_removes_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            (report / "summary.json").write_text("{}")
            paths = []

            async def sign(artifacts, dataset_name, metadata):
                return (
                    1,
                    "test/",
                    {artifact: "https://example.invalid" for artifact in artifacts},
                )

            async def send(url, path, artifact):
                paths.append(path)
                if artifact == "evaluation_artifacts.zip":
                    raise RuntimeError("private signed URL")
                await asyncio.sleep(0)
                self.assertTrue(path.exists())

            with (
                patch("training_ground.upload.get_signed_urls", sign),
                patch("training_ground.upload.upload_file_chunked", send),
                self.assertRaisesRegex(
                    RuntimeError,
                    r"^Failed to upload: evaluation_artifacts.zip \(RuntimeError\)\.$",
                ),
            ):
                asyncio.run(
                    upload_artifact_bundle("test", {"evaluation_artifacts.zip": report})
                )

            self.assertEqual(len(paths), 2)
            self.assertTrue(all(not path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
