"""Neural River Operations retraining module for Dams Downstream.

This module handles the retraining process for water level prediction models
using MLflow for experiment tracking and model management. Supports both
standard and autoregressive (AR) training modes.
"""

# Standard library imports
import json
import os
import traceback
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Third-party imports
import mlflow
from mlflow.models import ModelSignature, infer_signature
import pandas as pd
import torch
import yaml
from neuralhydrology.utils.config import Config

# Local imports
from retrain_eval import evaluate_and_visualize
from retrain_callback import start_run
from basin_config import setup_basin_paths, find_yaml_file

parser = argparse.ArgumentParser(description="Neural River Operations retraining script")
parser.add_argument("--ar", action="store_true", default=False, help="Enable autoregressive mode")
parser.add_argument("--only-hwls", nargs="*", help="List of HWL IDs to retrain")
parser.add_argument("--basin", type=str, help="Basin identifier (e.g., 1, 2)")
args = parser.parse_args()

# Setup basin paths
is_ar_mode = bool(args.ar)
basin_paths = setup_basin_paths(args.basin, "retrain", is_ar_mode)

if basin_paths["river_basin"]:
    print(f"Using basin configuration: {basin_paths}")
    RIVER_BASIN = basin_paths["river_basin"]
    STEP = basin_paths["train_step"]
    REGION = basin_paths["region"]
    TARGET_VARIABLE = basin_paths["target_variable"]
    CMD_NUM = basin_paths["cmd_num"]
    MLFLOW_TRAIN = basin_paths["mlflow_train"]
    MLFLOW_FORECAST = basin_paths["mlflow_forecast"]

BASE_DIR = basin_paths["base_dir"]
DATA_DIR = basin_paths["data_dir"]
YML_NAME = f"{STEP}_{REGION}_{TARGET_VARIABLE}_initial.yml"
input_yaml_file = find_yaml_file(DATA_DIR)

# Evaluation metrics
EVALUATION_METRICS = ["NSE", "KGE", "Alpha-NSE", "Beta-NSE", "Pearson-r", "RMSE"]


@dataclass
class HWL:
    """Data class representing a water level monitoring station.

    Attributes:
        id: Unique identifier for the station
        target_code: Code for the target variable
        target_name: Name of the target variable
        input_hoflw_code: Optional list of high flow input codes
        input_hrf_code: Optional list of river flow input codes
        input_hwl_code: Optional list of water level input codes
    """

    id: str
    target_code: str
    target_name: str
    input_hoflw_code: Optional[List[str]] = None
    input_hrf_code: Optional[List[str]] = None
    input_hwl_code: Optional[List[str]] = None
    train_start_date: str = None
    train_end_date: str = None
    validation_start_date: str = None
    validation_end_date: str = None
    test_start_date: str = None
    test_end_date: str = None

    @property
    def input_columns(self) -> List[str]:
        """Get all input column names including date and input codes."""
        input_columns = ["date"]
        if self.input_hoflw_code:
            input_columns.extend(self.input_hoflw_code)
        if self.input_hrf_code:
            input_columns.extend(self.input_hrf_code)
        if self.input_hwl_code:
            input_columns.extend(self.input_hwl_code)
        return input_columns

    @property
    def target_columns(self) -> List[str]:
        """Get target column names including date and target code."""
        return ["date", self.target_code]


def get_path(*path_segments: str) -> str:
    """Construct a path by joining segments with the base output directory.

    Args:
        *path_segments: Path segments to join

    Returns:
        str: Full path constructed from base directory and segments
    """
    return os.path.join(DATA_DIR, *path_segments)


def create_config(hwl: HWL, model: str) -> Config:
    """Create and return a configuration object for model training.

    Args:
        hwl: HWL object containing station information
        model: Name of the model to use

    Returns:
        Config: Configuration object for model training
    """
    base_yml = get_path(YML_NAME)
    with open(base_yml) as f:
        config = yaml.safe_load(f)

    config.update(
        {
            "train_start_date": hwl.train_start_date,
            "train_end_date": hwl.train_end_date,
            "validation_start_date": hwl.validation_start_date,
            "validation_end_date": hwl.validation_end_date,
            "test_start_date": hwl.test_start_date,
            "test_end_date": hwl.test_end_date,
            "experiment_name": f"{hwl.target_code}_{model}",
            "run_dir": get_path(f"runs/{hwl.target_code}/model"),
            "model": model,
            "dynamic_inputs": hwl.input_columns[1:],
            "target_variables": hwl.target_columns[1:],
            "clip_targets_to_zero": hwl.target_columns[1:],
            "train_basin_file": get_path(f"{STEP}_{REGION}.txt"),
            "test_basin_file": get_path(f"{STEP}_{REGION}.txt"),
            "validation_basin_file": get_path(f"{STEP}_{REGION}.txt"),
            "data_dir": get_path("train_data"),
        }
    )

    output_yml = get_path("Train_Overwrite.yml")
    with open(output_yml, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    return Config(Path(output_yml))


def get_hwls(
    region: str,
    is_ar_mode: bool,
    only_hwls: set[str],
) -> List[HWL]:
    """Load HWL configurations from the YAML file and create HWL instances.

    Args:
        region: Region identifier

    Returns:
        List[HWL]: List of HWL instances

    Raises:
        ValueError: If target_code is missing or invalid value type is found
    """

    def ensure_list_or_str_or_none(value: Any) -> Optional[List[str]]:
        """Ensure the value is a list or None if the field is empty or missing."""
        if value is None:
            return None
        if isinstance(value, list):
            return value if value else None
        if isinstance(value, str):
            return [value]
        raise ValueError(f"Invalid value type: {value}. Expected list, str, or None.")

    yaml_path = get_yaml_path(region, is_ar_mode)
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    hwls = []
    for item in config.get(f"{REGION}_{TARGET_VARIABLE}", []):
        key = list(item.keys())[0]
        if only_hwls and key not in only_hwls:
            continue
        body = item[key]
        if "target_code" not in body:
            raise ValueError(f"Missing target_code for HWL with id: {key}")

        hwls.append(
            HWL(
                id=key,
                target_code=body["target_code"],
                target_name=body["target_name"],
                input_hoflw_code=ensure_list_or_str_or_none(body.get("input_hoflw_code")),
                input_hrf_code=ensure_list_or_str_or_none(body.get("input_hrf_code")),
                input_hwl_code=ensure_list_or_str_or_none(body.get("input_hwl_code")),
                train_start_date=body["train_start_date"],
                train_end_date=body["train_end_date"],
                validation_start_date=body["validation_start_date"],
                validation_end_date=body["validation_end_date"],
                test_start_date=body["test_start_date"],
                test_end_date=body["test_end_date"],                
            )
        )
    return hwls


def prepare_data_and_signature(region: str, hwl: HWL, is_ar_mode: bool = False) -> ModelSignature:
    """Prepare and log training data for MLflow tracking and returns input/output signature.

    Args:
        region: Region identifier
        hwl: HWL object containing station information
        is_ar_mode: Whether to use autoregressive mode
    """
    input_artifact_dir = "input"
    target_artifact_dir = "target"
    file_name = f"{STEP}_{region}_hourly.csv"
    input_csv_path = get_path(f"train_data/time_series/{file_name}")
    input_columns = hwl.input_columns
    target_columns = hwl.target_columns
    inputs = pd.read_csv(input_csv_path)

    if is_ar_mode:
        input_df = inputs[input_columns[:-1]].set_index("date")
        target_df = inputs[target_columns].set_index("date")
        target = target_df.columns[0]
        AR_col = target + "_AR"
        input_df[AR_col] = target_df[target].shift(+1)
    else:
        input_df = inputs[input_columns].set_index("date")
        target_df = inputs[target_columns].set_index("date")

    mlflow.log_param("input_shape", input_df.shape)
    mlflow.log_param("target_shape", target_df.shape)

    input_path = get_path(f"train_data/time_series/input/hourly/{hwl.id}")
    target_path = get_path(f"train_data/time_series/target/hourly/{hwl.id}")
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)

    input_csv_path = f"{input_path}/{file_name}"
    target_csv_path = f"{target_path}/{file_name}"
    input_df.to_csv(input_csv_path)
    target_df.to_csv(target_csv_path)

    input_stats = input_df.describe().to_json()
    target_stats = target_df.describe().to_json()
    mlflow.log_dict(json.loads(input_stats), f"{input_artifact_dir}/input_statistics.json")
    mlflow.log_dict(json.loads(target_stats), f"{target_artifact_dir}/target_statistics.json")

    input_null_counts = input_df.isnull().sum().to_dict()
    target_null_counts = target_df.isnull().sum().to_dict()
    for col, count in input_null_counts.items():
        mlflow.log_metric(f"input_null_{col}", count)
    for col, count in target_null_counts.items():
        mlflow.log_metric(f"target_null_{col}", count)

    # 데이터셋 로깅 방식을 변경하여 경고 메시지 제거
    # source 매개변수를 명시적으로 지정하지 않음
    input_dataset = mlflow.data.from_pandas(
        input_df,
        name="input_dataset",
        source=None,  # 파일 경로를 직접 전달하지 않음
    )
    target_dataset = mlflow.data.from_pandas(
        target_df,
        name="target_dataset",
        source=None,  # 파일 경로를 직접 전달하지 않음
    )
    
    # 데이터셋 로깅
    mlflow.log_input(input_dataset, context="train")
    mlflow.log_input(target_dataset, context="train")

    # 파일 자체를 아티팩트로 로그
    mlflow.log_artifact(input_csv_path, artifact_path=input_artifact_dir)
    mlflow.log_artifact(target_csv_path, artifact_path=target_artifact_dir)

    return infer_signature(input_df, target_df)


def setup_mlflow_experiment(experiment_name: str) -> str:
    """Set up MLflow experiment and return experiment ID.

    Args:
        experiment_name: Name of the experiment

    Returns:
        str: Experiment ID

    Raises:
        mlflow.exceptions.MlflowException: If experiment creation fails
    """
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            raise

    mlflow.set_experiment(experiment_name)
    return experiment_id


def get_or_create_parent_run_id(experiment_id: str, parent_run_name: str) -> str:
    """Get or create a parent run ID for MLflow tracking.

    Args:
        experiment_id: ID of the experiment
        parent_run_name: Name of the parent run

    Returns:
        str: Parent run ID
    """
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_name = '{parent_run_name}'",
        output_format="list",
    )

    if runs:
        print(f"Found existing parent run: {parent_run_name}")
        return runs[0].info.run_id

    print(f"Parent run not found. Creating a new parent run: {parent_run_name}")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=parent_run_name,
    ) as parent_run:
        return parent_run.info.run_id


def build_model_name(region: str, hwl: HWL, is_ar_mode: bool) -> str:
    model_name = f"{hwl.target_name}({hwl.target_code.split('_')[0]})_수위예측모델"
    if is_ar_mode:
        model_name = f"ar_{model_name}"
    return model_name


def run(experiment_id: str, hwl: HWL, region: str, model: str, is_ar_mode: bool = False) -> None:
    """Run the training process for a single HWL station.

    Args:
        experiment_id: ID of the MLflow experiment
        hwl: HWL object containing station information
        region: Region identifier
        model: Name of the model to use
        is_ar_mode: Whether to use autoregressive mode
    """
    # Generate parent run name based on mode
    # base_name = f"{hwl.target_name}({hwl.target_code.split('_')[0]})"
    parent_run_name = f"{hwl.target_name}({hwl.target_code.split('_')[0]})"
    parent_run_id = get_or_create_parent_run_id(experiment_id, parent_run_name)

    # Generate run name based on mode
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = (
        f"ar_{parent_run_name}_{timestamp}"
        if is_ar_mode
        else f"{parent_run_name}_{timestamp}"
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        nested=True,
        parent_run_id=parent_run_id,
        tags={"model": model, "region": region, "ar_mode": str(is_ar_mode)},
    ):
        mlflow.log_params(
            {
                "target_code": hwl.target_code,
                "model": model,
                "region": region,
                "ar_mode": is_ar_mode,
            }
        )

        model_name = build_model_name(region, hwl, is_ar_mode)

        signature = prepare_data_and_signature(region=region, hwl=hwl, is_ar_mode=is_ar_mode)
        cfg = create_config(hwl, model)
        trainer_cfg = start_run(cfg, model_name, signature)
        
        # Initialize basin configuration for evaluation
        from retrain_eval import initialize_basin_config
        initialize_basin_config(args.basin, is_ar_mode)
        
        evaluate_and_visualize(hwl.target_code, trainer_cfg, EVALUATION_METRICS)
        torch.cuda.empty_cache()


def get_yaml_path(region: str, is_ar_mode: bool) -> str:
    """Get the path to the YAML configuration file for a region.

    Args:
        region: Region identifier

    Returns:
        str: Path to the YAML file

    Raises:
        NotImplementedError: If the region is not supported
    """
    yml_name = ""
    yml_name += input_yaml_file.split("/")[-1]

    if region == REGION:
        return get_path(yml_name)
    raise NotImplementedError(f"Region {region} is not supported.")


def main(region: str, is_ar_mode: bool = False, only_hwls: list[str] = None) -> None:
    """Main entry point for the retraining process.

    Args:
        region: Region identifier
        is_ar_mode: Whether to use autoregressive mode
    """
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    if not username or not password:
        print("Error: User not authenticated. Please log in first.")
        exit(1)

    print(f"Running as user: {username}")
    print(f"Mode: {'Autoregressive' if is_ar_mode else 'Standard'}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.enable_system_metrics_logging()
    mlflow.autolog()

    regions = [region]
    model = "cudalstm"

    for region in regions:
        experiment_id = setup_mlflow_experiment(MLFLOW_TRAIN)
        if only_hwls:
            only_hwls = set(only_hwls)
        hwls = get_hwls(region, is_ar_mode, only_hwls)
        for hwl in hwls:
            try:
                run(experiment_id, hwl, region, model, is_ar_mode)
            except Exception:
                print("An error occurred:")
                traceback.print_exc()

if __name__ == "__main__":
    main(REGION, args.ar, args.only_hwls)