import asyncio
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import httpx
import typer

from .training_backends import (
    RFDETR_BACKEND,
    RFDETR_MODEL_FAMILY,
    RFDETR_SEG_MODELS,
    YOLO26_BACKEND,
    YOLO26_MODEL_FAMILY,
    YOLO26_SEG_MODELS,
    TrainingArtifacts,
)

# Hardcoded configuration
OBJECT_LEDGER_URL = "https://ledger.staging.agrowizard.com"
API_KEY = "98SMkwyMFU6Z6Obd3K5/o5H/cJ19Nj7uj+ddHI+i1hY="

# Upload settings
CHUNK_SIZE = 32 * 1024 * 1024  # 32MB chunks for large files
MAX_CONCURRENT_UPLOADS = 8
UPLOAD_TIMEOUT = 900  # 15 minutes per file
METADATA_FILENAMES = ("metadata.json", "upload_metadata.json")


def slugify_dataset_name(dataset_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", dataset_name.strip().lower()).strip("-")
    if not slug:
        raise ValueError(
            "Dataset name must contain at least one alphanumeric character"
        )
    return slug


def _relative_to_runs_dir(path: Path, runs_dir: Path) -> str:
    try:
        return str(path.relative_to(runs_dir))
    except ValueError:
        return str(path)


def build_training_metadata(
    dataset_name: str, artifacts: TrainingArtifacts
) -> dict[str, Any]:
    dataset_slug = slugify_dataset_name(dataset_name)
    metadata = {
        "dataset_name": dataset_name,
        "dataset_slug": dataset_slug,
        "backend": artifacts.backend,
        "task": "segmentation",
        "model_name": artifacts.model_name,
        "model_size": artifacts.model_size,
        "model_family": (
            YOLO26_MODEL_FAMILY
            if artifacts.backend == YOLO26_BACKEND
            else RFDETR_MODEL_FAMILY
        ),
        "runs_dir": ".",
        "primary_checkpoint_path": _relative_to_runs_dir(
            artifacts.primary_checkpoint_path, artifacts.runs_dir
        ),
        "secondary_checkpoint_path": (
            _relative_to_runs_dir(
                artifacts.secondary_checkpoint_path, artifacts.runs_dir
            )
            if artifacts.secondary_checkpoint_path is not None
            else None
        ),
        "metrics_path": _relative_to_runs_dir(
            artifacts.metrics_path, artifacts.runs_dir
        ),
        "eval_dir": _relative_to_runs_dir(artifacts.eval_dir, artifacts.runs_dir),
        "onnx_path": _relative_to_runs_dir(artifacts.onnx_path, artifacts.runs_dir),
    }
    if artifacts.mask_downsample_ratio is not None:
        metadata["mask_downsample_ratio"] = artifacts.mask_downsample_ratio
    return metadata


def write_upload_metadata(
    runs_dir: Path,
    dataset_name: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    payload = dict(metadata or {})
    payload.setdefault("dataset_name", dataset_name)
    payload.setdefault("dataset_slug", slugify_dataset_name(dataset_name))

    canonical_path = runs_dir / METADATA_FILENAMES[0]
    text = json.dumps(payload, indent=2) + "\n"
    canonical_path.write_text(text, encoding="utf-8")

    legacy_path = runs_dir / METADATA_FILENAMES[1]
    legacy_path.write_text(text, encoding="utf-8")

    return canonical_path


def read_upload_metadata(runs_dir: Path) -> dict[str, Any] | None:
    for name in METADATA_FILENAMES:
        path = runs_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def resolve_onnx_artifact_path(
    runs_dir: Path,
    preferred_relative_path: str | None = None,
    *,
    default_filename: str,
) -> Path:
    if preferred_relative_path:
        preferred_path = runs_dir / preferred_relative_path
        if preferred_path.exists():
            return preferred_path

    default_path = runs_dir / default_filename
    if default_path.exists():
        return default_path

    common_filenames = [
        name
        for name in (
            preferred_relative_path,
            "inference_model.onnx",
            "model.onnx",
        )
        if name
    ]
    for filename in common_filenames:
        candidate = runs_dir / filename
        if candidate.exists():
            return candidate

    onnx_files = sorted(path for path in runs_dir.glob("*.onnx") if path.is_file())
    if len(onnx_files) == 1:
        return onnx_files[0]
    if len(onnx_files) > 1:
        prioritized_names = {"inference_model.onnx", "model.onnx"}
        prioritized_files = [
            path for path in onnx_files if path.name in prioritized_names
        ]
        if len(prioritized_files) == 1:
            return prioritized_files[0]
        return max(onnx_files, key=lambda path: path.stat().st_mtime)

    return default_path


def resolve_dataset_name(runs_dir: Path, dataset_name: str | None = None) -> str:
    if dataset_name:
        return dataset_name

    metadata = read_upload_metadata(runs_dir)
    if metadata is not None:
        stored_name = metadata.get("dataset_name")
        if isinstance(stored_name, str) and stored_name.strip():
            return stored_name

    if runs_dir.name != "runs":
        return runs_dir.name

    raise ValueError(
        f"Could not determine dataset name. Pass it explicitly or add {METADATA_FILENAMES[0]} to {runs_dir}."
    )


def artifact_files_for_training(artifacts: TrainingArtifacts) -> dict[str, Path]:
    if artifacts.backend == RFDETR_BACKEND:
        assert artifacts.secondary_checkpoint_path is not None
        return {
            "checkpoint_best_ema.pth": artifacts.primary_checkpoint_path,
            "checkpoint_best_regular.pth": artifacts.secondary_checkpoint_path,
            "metrics.csv": artifacts.metrics_path,
            "evaluation_artifacts.zip": artifacts.eval_dir,
            "model.onnx": artifacts.onnx_path,
        }

    return {
        "best.pt": artifacts.primary_checkpoint_path,
        "last.pt": artifacts.secondary_checkpoint_path
        or artifacts.runs_dir / "weights" / "last.pt",
        "metrics.csv": artifacts.metrics_path,
        "evaluation_artifacts.zip": artifacts.eval_dir,
        "model.onnx": artifacts.onnx_path,
    }


def resolve_training_artifacts(
    runs_dir: Path, backend: str
) -> tuple[TrainingArtifacts, dict[str, Any] | None]:
    from .training_backends import normalize_backend

    normalized_backend = normalize_backend(backend)
    metadata = read_upload_metadata(runs_dir)

    if normalized_backend == RFDETR_BACKEND:
        preferred_onnx_path = (metadata or {}).get("onnx_path")
        if not isinstance(preferred_onnx_path, str):
            preferred_onnx_path = None
        artifacts = TrainingArtifacts(
            backend=RFDETR_BACKEND,
            model_name=str(
                (metadata or {}).get("model_name", RFDETR_SEG_MODELS["nano"][0])
            ),
            model_size=(metadata or {}).get("model_size"),
            runs_dir=runs_dir,
            primary_checkpoint_path=runs_dir / "checkpoint_best_ema.pth",
            secondary_checkpoint_path=runs_dir / "checkpoint_best_regular.pth",
            metrics_path=runs_dir / "metrics.csv",
            eval_dir=Path(
                runs_dir
                / str(
                    (metadata or {}).get(
                        "eval_dir", "checkpoint_best_ema_test_evaluation"
                    )
                )
            ),
            onnx_path=resolve_onnx_artifact_path(
                runs_dir,
                preferred_onnx_path,
                default_filename="inference_model.onnx",
            ),
        )
        return artifacts, metadata

    model_size = None
    if metadata is not None:
        raw_size = metadata.get("model_size")
        if isinstance(raw_size, str) and raw_size in YOLO26_SEG_MODELS:
            model_size = raw_size
    if model_size is None and runs_dir.name.startswith("yolo26-"):
        candidate_size = runs_dir.name.removeprefix("yolo26-")
        if candidate_size in YOLO26_SEG_MODELS:
            model_size = candidate_size

    artifacts = TrainingArtifacts(
        backend=YOLO26_BACKEND,
        model_name=YOLO26_SEG_MODELS.get(model_size or "", "yolo26n-seg.pt"),
        model_size=model_size,
        runs_dir=runs_dir,
        primary_checkpoint_path=runs_dir / "weights" / "best.pt",
        secondary_checkpoint_path=runs_dir / "weights" / "last.pt",
        metrics_path=runs_dir / "results.csv",
        eval_dir=Path(
            runs_dir / str((metadata or {}).get("eval_dir", "best_test_evaluation"))
        ),
        onnx_path=resolve_onnx_artifact_path(
            runs_dir,
            (metadata or {}).get("onnx_path")
            if isinstance((metadata or {}).get("onnx_path"), str)
            else None,
            default_filename="model.onnx",
        ),
    )
    return artifacts, metadata


async def get_signed_urls(
    artifacts: list[str],
    dataset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[int, str, dict[str, str]]:
    url = f"{OBJECT_LEDGER_URL}/v1/storage/signed-urls"
    headers = {"x-api-key": API_KEY}
    data = {
        "artifacts": artifacts,
        "dataset_name": dataset_name,
        "metadata": metadata,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()

    result = response.json()
    return result["run_id"], result["run_prefix"], result["urls"]


async def upload_file_chunked(
    signed_url: str, local_path: Path, artifact_name: str, progress: bool = True
) -> None:
    file_size = local_path.stat().st_size

    if progress:
        typer.echo(f"Uploading {artifact_name} ({file_size / 1024 / 1024:.1f} MB)...")

    if file_size > CHUNK_SIZE:
        await _upload_large_file(
            signed_url, local_path, file_size, artifact_name, progress
        )
    else:
        await _upload_small_file(signed_url, local_path, artifact_name, progress)

    if progress:
        typer.echo(f"{artifact_name} uploaded successfully")


async def _upload_small_file(
    signed_url: str, local_path: Path, artifact_name: str, progress: bool
) -> None:
    del artifact_name, progress
    async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT) as client:
        with local_path.open("rb") as handle:
            response = await client.put(signed_url, content=handle.read())
            response.raise_for_status()


async def _upload_large_file(
    signed_url: str,
    local_path: Path,
    file_size: int,
    artifact_name: str,
    progress: bool,
) -> None:
    async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT * 2) as client:

        async def file_stream():
            bytes_uploaded = 0
            with local_path.open("rb") as handle:
                while True:
                    chunk = handle.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    bytes_uploaded += len(chunk)
                    if progress and bytes_uploaded % (50 * 1024 * 1024) == 0:
                        percent = (bytes_uploaded / file_size) * 100
                        typer.echo(
                            f"    {artifact_name}: {percent:.0f}% ({bytes_uploaded / 1024 / 1024:.1f} MB)"
                        )
                    yield chunk

        response = await client.put(signed_url, content=file_stream())
        response.raise_for_status()


async def upload_artifacts(
    urls: dict[str, str],
    local_paths: dict[str, Path],
    max_concurrent: int = MAX_CONCURRENT_UPLOADS,
) -> None:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def upload_with_semaphore(artifact: str, signed_url: str) -> None:
        async with semaphore:
            await upload_file_chunked(signed_url, local_paths[artifact], artifact)

    await asyncio.gather(
        *[
            upload_with_semaphore(artifact, signed_url)
            for artifact, signed_url in urls.items()
        ],
        return_exceptions=False,
    )


def zip_directory(source_dir: Path, zip_path: Path, progress: bool = True) -> None:
    if progress:
        typer.echo(f"Zipping {source_dir.name}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(source_dir))

    if progress:
        zip_size = zip_path.stat().st_size
        typer.echo(f"Created {zip_path.name} ({zip_size / 1024 / 1024:.1f} MB)")


async def upload_artifact_bundle(
    runs_dir: Path,
    dataset_name: str,
    artifact_files: dict[str, Path],
    metadata: dict[str, Any] | None = None,
) -> int:
    typer.echo("Preparing artifacts for upload...")
    dataset_slug = slugify_dataset_name(dataset_name)
    typer.echo(f"Dataset scope: {dataset_slug}")

    temp_files: list[Path] = []
    local_paths: dict[str, Path] = {}
    for artifact_name, source_path in artifact_files.items():
        if source_path.is_dir():
            zip_path = runs_dir / artifact_name
            zip_directory(source_path, zip_path)
            local_paths[artifact_name] = zip_path
            temp_files.append(zip_path)
        else:
            local_paths[artifact_name] = source_path

    metadata_path = write_upload_metadata(runs_dir, dataset_name, metadata)
    local_paths.setdefault("metadata.json", metadata_path)
    request_metadata = None
    if metadata is not None:
        request_metadata = {
            key: metadata.get(key)
            for key in ("backend", "model_family", "model_name", "model_size", "task")
            if metadata.get(key) is not None
        }

    typer.echo("Requesting upload URLs from object-ledger...")
    try:
        run_id, run_prefix, urls = await get_signed_urls(
            list(local_paths),
            dataset_name,
            request_metadata,
        )
        typer.echo(f"Assigned scoped run ID: {dataset_slug}-{run_id}")
    except httpx.HTTPError as exc:
        typer.echo(f"Failed to get signed URLs: {exc}", err=True)
        raise

    typer.echo(f"Uploading {len(local_paths)} artifacts to GCS...")
    await upload_artifacts(urls, local_paths)

    for temp_file in temp_files:
        temp_file.unlink(missing_ok=True)

    typer.echo(f"Successfully uploaded run {dataset_slug}-{run_id} to GCS!")
    typer.echo(f"Location: stm-ai-bucket/{run_prefix}{run_id}/")

    return run_id


async def upload_training_run(
    runs_dir: Path,
    dataset_name: str,
    checkpoint_ema_path: Path,
    checkpoint_regular_path: Path,
    metrics_path: Path,
    eval_dir: Path,
    onnx_path: Path,
) -> int:
    artifacts = TrainingArtifacts(
        backend=RFDETR_BACKEND,
        model_name=RFDETR_SEG_MODELS["nano"][0],
        model_size=None,
        runs_dir=runs_dir,
        primary_checkpoint_path=checkpoint_ema_path,
        secondary_checkpoint_path=checkpoint_regular_path,
        metrics_path=metrics_path,
        eval_dir=eval_dir,
        onnx_path=onnx_path,
    )
    metadata = build_training_metadata(dataset_name, artifacts)
    return await upload_artifact_bundle(
        runs_dir=runs_dir,
        dataset_name=dataset_name,
        artifact_files=artifact_files_for_training(artifacts),
        metadata=metadata,
    )
