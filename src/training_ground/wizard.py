from concurrent.futures import ThreadPoolExecutor

import questionary
import roboflow
import torch.multiprocessing as mp
from questionary import Choice


def fetch_project_info(data):
    workspace, project_id = data
    project = workspace.project(project_id)
    return project, project.versions()


def run_wizard():
    mp.set_sharing_strategy("file_system")
    roboflow.login()
    rf = roboflow.Roboflow()

    workspace = rf.workspace()
    projects = []
    with ThreadPoolExecutor() as executor:
        for project, versions in executor.map(
            fetch_project_info,
            [(workspace, project.split("/")[-1]) for project in workspace.projects()],
        ):
            projects.append((project, versions))

    projects.sort(key=lambda p: p[0].updated, reverse=True)

    project, versions = questionary.select(
        "Select your project",
        choices=[Choice(title=p.id, value=(p, v)) for p, v in projects],
    ).ask()

    version = questionary.select(
        "Select the dataset version",
        choices=[Choice(title=v.id.split("/")[-1], value=v) for v in versions],
    ).ask()

    batch_size, grad_accum_steps = questionary.select(
        "Select the GPU VRAM",
        choices=[
            Choice(title="RTX  4080 (16GB)", value=(16, 1)),
            Choice(title="RTX  4090 (24GB)", value=(32, 1)),
            Choice(title="RTX  5090 (32GB)", value=(48, 1)),
            Choice(title="RTX A6000 (48GB)", value=(68, 1)),
        ],
    ).ask()

    dataset_path = f"./datasets/{version.id}"
    version.download(model_format="coco", location=dataset_path)

    # Import here to avoid slow startup when just using other CLI commands
    from rfdetr.detr import RFDETRSegNano

    model = RFDETRSegNano()
    model.train(
        dataset_dir=dataset_path,
        epochs=100,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        resolution=372,
        early_stopping=True,
        early_stopping_patience=3,
        progress_bar=True,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=False,
        num_queries=50,
        num_select=20,
        output_dir="runs",
    )
