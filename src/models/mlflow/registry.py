import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional, Any
import torch

from ..base import BaseModel


def register_model(
    model_path: str,
    name: str,
    version: Optional[str] = None,
    stage: str = "None",
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Register a model in the MLflow Model Registry.

    Args:
        model_path: Path to the model, either a local path or a run-relative artifact path
        name: Name to register the model under
        version: Version string (auto-incremented if not provided)
        stage: Stage for the model ('None', 'Staging', 'Production', 'Archived')
        description: Description for the model
        tags: Dictionary of tags to attach to the model

    Returns:
        Model URI in the registry
    """
    client = MlflowClient()

    # Check if model_path is a run artifact (e.g., "runs:/run_id/model")
    if model_path.startswith("runs:/"):
        # Register model from run artifact
        run_id = model_path.split("/")[1]
        artifact_path = "/".join(model_path.split("/")[2:])

        model_details = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{artifact_path}", name=name
        )
        version_num = model_details.version
    else:
        # Register model from local path
        try:
            # Load the PyTorch model
            pytorch_model = torch.load(model_path)

            # Get model class information
            model_class = pytorch_model.get("model_class", "BaseModel")

            # Get example input for signature if available
            if "input_dim" in pytorch_model.get("hyperparameters", {}):
                input_dim = pytorch_model["hyperparameters"]["input_dim"]
                example_input = torch.randn(1, input_dim)

                # Create model instance for inference
                from ..models import create_model

                temp_model = create_model(model_class, pytorch_model["hyperparameters"])
                temp_model.load_state_dict(pytorch_model["model_state_dict"])

                signature = mlflow.models.infer_signature(
                    example_input.numpy(), temp_model(example_input).detach().numpy()
                )
            else:
                signature = None

            # Log the model
            mlflow.pytorch.log_model(
                pytorch_model,
                artifact_path="model",
                registered_model_name=name,
                signature=signature,
            )

            # Get latest version
            versions = client.search_model_versions(f"name='{name}'")
            version_num = max([int(v.version) for v in versions])

        except Exception as e:
            raise ValueError(f"Failed to register model: {str(e)}")

    # Set model version if provided
    if version is not None:
        client.set_model_version_tag(
            name=name, version=version_num, key="manual_version", value=version
        )

    # Set stage if different from default
    if stage != "None":
        client.transition_model_version_stage(name=name, version=version_num, stage=stage)

    # Set description if provided
    if description is not None:
        client.update_model_version(name=name, version=version_num, description=description)

    # Set tags if provided
    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(name=name, version=version_num, key=key, value=value)

    # Return model URI
    return f"models:/{name}/{version_num}"


def get_latest_model_version(name: str, stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get the latest version of a registered model.

    Args:
        name: Name of the registered model
        stage: Filter by stage ('None', 'Staging', 'Production', 'Archived')

    Returns:
        Dictionary with model version details, or None if not found
    """
    client = MlflowClient()

    # Get all versions of the model
    versions = client.search_model_versions(f"name='{name}'")

    if not versions:
        return None

    # Filter by stage if specified
    if stage is not None:
        versions = [v for v in versions if v.current_stage == stage]

    if not versions:
        return None

    # Get the latest version
    latest_version = max(versions, key=lambda v: int(v.version))

    # Get run if available
    run_info = None
    if latest_version.run_id:
        try:
            run = client.get_run(latest_version.run_id)
            run_info = {
                "run_id": run.info.run_id,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }
        except Exception:
            pass

    # Build result
    result = {
        "name": latest_version.name,
        "version": latest_version.version,
        "stage": latest_version.current_stage,
        "description": latest_version.description,
        "run_id": latest_version.run_id,
        "creation_timestamp": latest_version.creation_timestamp,
        "last_updated_timestamp": latest_version.last_updated_timestamp,
        "run_info": run_info,
    }

    return result


def list_model_versions(name: str, stages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    List all versions of a registered model.

    Args:
        name: Name of the registered model
        stages: Filter by stages, e.g. ['Production', 'Staging']

    Returns:
        List of dictionaries with model version details
    """
    client = MlflowClient()

    # Get all versions of the model
    versions = client.search_model_versions(f"name='{name}'")

    if not versions:
        return []

    # Filter by stages if specified
    if stages:
        versions = [v for v in versions if v.current_stage in stages]

    # Build results
    results = []
    for version in versions:
        # Get run if available
        run_info = None
        if version.run_id:
            try:
                run = client.get_run(version.run_id)
                run_info = {
                    "run_id": run.info.run_id,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                }
            except Exception:
                pass

        # Build result
        result = {
            "name": version.name,
            "version": version.version,
            "stage": version.current_stage,
            "description": version.description,
            "run_id": version.run_id,
            "creation_timestamp": version.creation_timestamp,
            "last_updated_timestamp": version.last_updated_timestamp,
            "run_info": run_info,
        }

        results.append(result)

    # Sort by version (descending)
    results.sort(key=lambda x: int(x["version"]), reverse=True)

    return results


def load_registered_model(
    name: str, version: Optional[str] = None, stage: Optional[str] = "Production"
) -> BaseModel:
    """
    Load a model from the MLflow Model Registry.

    Args:
        name: Name of the registered model
        version: Specific version to load (overrides stage)
        stage: Stage to load from ('Production', 'Staging', etc.)

    Returns:
        Loaded model
    """
    # Determine model URI
    if version is not None:
        model_uri = f"models:/{name}/{version}"
    else:
        model_uri = f"models:/{name}/{stage}"

    # Load model as PyTorch model
    model = mlflow.pytorch.load_model(model_uri)

    return model
