import os
from typing import Optional
import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel
import mlflow.pytorch
from neuralhydrology.utils.config import Config
from pathlib import Path

import yaml


def to_dict(path: str):
    with open(path) as fd:
        return yaml.safe_load(fd)


class ModelRegistry:
    def __init__(
        self,
        region: str,
        hwl_id: str,
        name: str,
        is_ar_mode: bool,
        version: Optional[int] = None,
        verbose: bool = False,
    ):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.tracking_uri)

        self.region = region
        self.hwl_id = hwl_id
        self.name = name
        self.is_ar_mode = is_ar_mode
        self.verbose = verbose
        if version is None:
            self.version = "latest"
        else:
            self.version = str(version)
        self.model_name = ModelRegistry.build_model_name(
            self.region,
            self.name,
            self.hwl_id,
            self.is_ar_mode,
        )
        self.metadata = ModelRegistry.build_metadata(self.model_name, self.version)
        self.run_id = self.metadata.run_id

        self.model = ModelRegistry.build_torch_model(
            self.model_name,
            self.version,
        )

        self.config = ModelRegistry.build_config(self.run_id)
        self.scaler = ModelRegistry.build_scaler(self.run_id)

    @staticmethod
    def build_model_name(region: str, hwl_id: str, name: str, is_ar_mode: bool) -> str:
        model_name = f"{hwl_id}({name.split('_')[0]})_수위예측모델"
        if is_ar_mode:
            model_name = f"ar_{model_name}"
        return model_name

    @staticmethod
    def build_metadata(model_name: str, version: str):
        def get_model(model_name: str, version: str) -> PyFuncModel:
            return mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{version}")

        return get_model(model_name, version).metadata

    def build_torch_model(model_name: str, version: str):
        return mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{version}")

    @staticmethod
    def build_config(run_id: str, artifact_path: str = "runs/config.yml"):
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)

        return Config(yml_path_or_dict=to_dict(path))

    @staticmethod
    def build_scaler(run_id: str, artifact_path: str = "runs/train_data/train_data_scaler.yml"):
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        return to_dict(path)

    def __repr__(self):
        info = {
            "tracking_uri": self.tracking_uri,
            "region": self.region,
            "hwl_id": self.hwl_id,
            "is_ar_mode": self.is_ar_mode,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "version": self.version,
            "metadata": self.metadata.to_dict(),
        }
        if self.verbose:
            print("############# Config #############")
            print(self.config.as_dict())
            print("############# Scaler #############")
            print(self.scaler)
            print("############# Model #############")
            print(self.model)
        print("############# ModelRegistry #############")
        return yaml.safe_dump(info, default_flow_style=False)


if __name__ == "__main__":
    region = "YD_Dam_Downstream"
    hwl_id = "3002655_hwl"
    is_ar_mode = False

    mr = ModelRegistry(region, hwl_id, is_ar_mode, verbose=True)
    print(mr)
