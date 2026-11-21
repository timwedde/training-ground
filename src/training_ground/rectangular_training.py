from copy import deepcopy
from itertools import chain

import torch
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.patches import override_configs
from ultralytics.utils.torch_utils import autocast, profile_ops


class RectangularSegmentationTrainer(SegmentationTrainer):
    def auto_batch(self):
        if self.device.type != "cuda" or torch.backends.cudnn.benchmark:
            return super().auto_batch()

        with override_configs(self.args, {"cache": False}):
            dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        batch_limit = min(len(dataset), 1024)
        subsets = getattr(dataset, "datasets", [dataset])
        shapes = list(chain.from_iterable(subset.batch_shapes for subset in subsets))
        # Bound both axes so the probe still covers batches regrouped at the chosen size.
        height, width = (int(max(axis)) for axis in zip(*shapes, strict=True))
        max_objects = max(
            len(label["cls"]) for subset in subsets for label in subset.labels
        )
        del dataset, subsets

        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        reusable_bytes = torch.cuda.memory_reserved(self.device)
        available_gib = (free_bytes + reusable_bytes) / 2**30
        fraction = self.batch_size if 0 < self.batch_size < 1 else 0.6
        budget_gib = available_gib * fraction
        prefix = colorstr("AutoBatch: ")
        LOGGER.info(
            f"{prefix}Profiling rectangular batches (height={height}, width={width}) "
            f"with a {budget_gib:.2f} GiB budget."
        )
        model = deepcopy(self.model).train()
        channels = model.yaml.get("channels", 3)
        amp = torch.bfloat16 if self.args.amp == "bf16" else self.amp

        def measure(batch_size, model):
            try:
                with autocast(enabled=amp, device=self.device.type):
                    (result,) = profile_ops(
                        torch.empty(batch_size, channels, height, width),
                        model,
                        n=1,
                        device=self.device,
                        max_num_obj=max_objects,
                    )
                return result[2] if result is not None else None
            except torch.OutOfMemoryError:
                return None
            finally:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        best = 0
        peak_gib = 0.0
        upper = batch_limit + 1
        candidate = 1
        try:
            while candidate < upper:
                usage = measure(candidate, model)
                if usage is not None and 0 < usage <= budget_gib:
                    best, peak_gib = candidate, usage
                else:
                    upper = candidate
                if upper == batch_limit + 1 and best < batch_limit:
                    candidate = min(best * 2, batch_limit)
                else:
                    candidate = (best + upper + 1) // 2
        finally:
            del model
            torch.cuda.empty_cache()

        if best == 0:
            raise RuntimeError(
                "Rectangular AutoBatch could not fit one image within its VRAM budget. "
                "Reduce the input size, free GPU memory, or set --batch-size explicitly."
            )
        LOGGER.info(
            f"{prefix}Using batch-size {best} for {self.device} "
            f"({peak_gib:.2f}/{available_gib:.2f} GiB profiled)."
        )
        return best
