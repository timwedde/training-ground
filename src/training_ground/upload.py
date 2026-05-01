import asyncio
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
import typer

# Hardcoded configuration
OBJECT_LEDGER_URL = "https://ledger.staging.agrowizard.com"
API_KEY = "98SMkwyMFU6Z6Obd3K5/o5H/cJ19Nj7uj+ddHI+i1hY="

# Upload settings
CHUNK_SIZE = 32 * 1024 * 1024  # 32MB chunks for large files
MAX_CONCURRENT_UPLOADS = 8
UPLOAD_TIMEOUT = 900  # 15 minutes per file
UPLOAD_METADATA_FILENAME = "upload_metadata.json"


def slugify_dataset_name(dataset_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", dataset_name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("Dataset name must contain at least one alphanumeric character")
    return slug


def write_upload_metadata(runs_dir: Path, dataset_name: str) -> Path:
    metadata_path = runs_dir / UPLOAD_METADATA_FILENAME
    metadata = {
        "dataset_name": dataset_name,
        "dataset_slug": slugify_dataset_name(dataset_name),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def resolve_dataset_name(runs_dir: Path, dataset_name: str | None = None) -> str:
    if dataset_name:
        return dataset_name

    metadata_path = runs_dir / UPLOAD_METADATA_FILENAME
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stored_name = metadata.get("dataset_name")
        if isinstance(stored_name, str) and stored_name.strip():
            return stored_name

    if runs_dir.name != "runs":
        return runs_dir.name

    raise ValueError(
        f"Could not determine dataset name. Pass it explicitly or add {UPLOAD_METADATA_FILENAME} to {runs_dir}."
    )


async def get_signed_urls(
    artifacts: List[str], dataset_name: str | None = None
) -> Tuple[int, str, Dict[str, str]]:
    url = f"{OBJECT_LEDGER_URL}/v1/storage/signed-urls"
    headers = {"x-api-key": API_KEY}
    data = {"artifacts": artifacts, "dataset_name": dataset_name}

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

    # For large files, use chunked upload
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
    async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT) as client:
        with open(local_path, "rb") as f:
            content = f.read()
            response = await client.put(signed_url, content=content)
            response.raise_for_status()


async def _upload_large_file(
    signed_url: str,
    local_path: Path,
    file_size: int,
    artifact_name: str,
    progress: bool,
) -> None:
    async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT * 2) as client:
        # Stream the file to avoid loading it all into memory
        async def file_stream():
            bytes_uploaded = 0
            with open(local_path, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    bytes_uploaded += len(chunk)
                    if (
                        progress and bytes_uploaded % (50 * 1024 * 1024) == 0
                    ):  # Every 50MB
                        percent = (bytes_uploaded / file_size) * 100
                        typer.echo(
                            f"    {percent:.0f}% ({bytes_uploaded / 1024 / 1024:.1f} MB)"
                        )
                    yield chunk

        response = await client.put(signed_url, content=file_stream())
        response.raise_for_status()


async def upload_artifacts(
    urls: Dict[str, str],
    local_paths: Dict[str, Path],
    max_concurrent: int = MAX_CONCURRENT_UPLOADS,
) -> None:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def upload_with_semaphore(artifact: str, signed_url: str) -> None:
        async with semaphore:
            local_path = local_paths[artifact]
            await upload_file_chunked(signed_url, local_path, artifact)

    # Create upload tasks for all artifacts
    tasks = [
        upload_with_semaphore(artifact, signed_url)
        for artifact, signed_url in urls.items()
    ]

    # Run all uploads concurrently (up to max_concurrent at a time)
    await asyncio.gather(*tasks, return_exceptions=False)


def zip_directory(source_dir: Path, zip_path: Path, progress: bool = True) -> None:
    if progress:
        typer.echo(f"Zipping {source_dir.name}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)

    if progress:
        zip_size = zip_path.stat().st_size
        typer.echo(f"Created {zip_path.name} ({zip_size / 1024 / 1024:.1f} MB)")


async def upload_training_run(
    runs_dir: Path,
    dataset_name: str,
    checkpoint_ema_path: Path,
    checkpoint_regular_path: Path,
    metrics_path: Path,
    eval_dir: Path,
    onnx_path: Path,
) -> int:
    typer.echo("Preparing artifacts for upload...")
    dataset_slug = slugify_dataset_name(dataset_name)
    typer.echo(f"Dataset scope: {dataset_slug}")

    # Create zip file for evaluation artifacts
    zip_path = runs_dir / "evaluation_artifacts.zip"
    zip_directory(eval_dir, zip_path)

    # Define artifacts to upload
    artifact_files = {
        "checkpoint_best_ema.pth": checkpoint_ema_path,
        "checkpoint_best_regular.pth": checkpoint_regular_path,
        "metrics.csv": metrics_path,
        "evaluation_artifacts.zip": zip_path,
        "model.onnx": onnx_path,
    }
    artifacts = list(artifact_files)
    local_paths = dict(artifact_files)

    # Get signed URLs from object-ledger
    typer.echo("Requesting upload URLs from object-ledger...")
    try:
        run_id, run_prefix, urls = await get_signed_urls(artifacts, dataset_name)
        typer.echo(f"Assigned scoped run ID: {dataset_slug}-{run_id}")
    except httpx.HTTPError as e:
        typer.echo(f"Failed to get signed URLs: {e}", err=True)
        raise

    # Upload all artifacts in parallel
    typer.echo(f"Uploading {len(artifacts)} artifacts to GCS...")
    await upload_artifacts(urls, local_paths)

    # Cleanup zip file
    zip_path.unlink()

    typer.echo(f"Successfully uploaded run {dataset_slug}-{run_id} to GCS!")
    typer.echo(f"Location: stm-ai-bucket/{run_prefix}{run_id}/")

    return run_id
