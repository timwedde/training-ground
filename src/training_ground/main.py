from concurrent.futures import ThreadPoolExecutor

import questionary
import roboflow
import typer
from questionary import Choice
from rfdetr import RFDETRSegNano
from roboflow.core.project import Project
from roboflow.core.version import Version

app = typer.Typer()


def fetch_project_info(data):
    workspace, project_id = data
    project = workspace.project(project_id)
    versions = project.versions()
    return project, versions


@app.command()
def wizard():
    import torch.multiprocessing as mp

    mp.set_sharing_strategy("file_system")

    roboflow.login()

    rf = roboflow.Roboflow()

    # List all projects for your workspace
    workspace = rf.workspace()
    projects: list[tuple[Project, list[Version]]] = []
    with ThreadPoolExecutor() as executor:
        for project, versions in executor.map(
            fetch_project_info,
            [(workspace, project.split("/")[-1]) for project in workspace.projects()],
        ):
            projects.append((project, versions))

    projects.sort(key=lambda p: p[0].updated, reverse=True)

    project, versions = questionary.select(
        "Select your project",
        choices=[
            Choice(title=project.id, value=(project, versions))
            for project, versions in projects
        ],
    ).ask()

    version: Version = questionary.select(
        "Select the dataset version",
        choices=[
            Choice(title=version.id.split("/")[-1], value=version)
            for version in versions
        ],
    ).ask()

    batch_size, grad_accum_steps = questionary.select(
        "Select the GPU VRAM",
        choices=[
            Choice(title="16GB", value=(4, 4)),
            Choice(title="24GB", value=(8, 2)),
            Choice(title="32GB", value=(16, 1)),
            Choice(title="40GB", value=(24, 1)),
        ],
    ).ask()

    dataset_path = f"./datasets/{version.id}"
    version.download(model_format="coco", location=dataset_path)

    model = RFDETRSegNano()

    model.train(
        dataset_dir=dataset_path,
        epochs=100,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        resolution=372,
        early_stopping=True,
        early_stopping_patience=5,
        eval_interval=2,
        progress_bar=True,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=False,
        output_dir="runs",
    )


if __name__ == "__main__":
    app()
