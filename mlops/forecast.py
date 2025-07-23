import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
import traceback
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import timedelta
import torch
import yaml
import mlflow
import matplotlib.pyplot as plt
from forecast_eval import forecast
from model import ModelRegistry
import warnings
from basin_config import setup_basin_paths, find_yaml_file

warnings.simplefilter(action="ignore")

parser = argparse.ArgumentParser(description="Neural River Operations forecasting script")
parser.add_argument("--ar", action="store_true", default=False, help="Enable autoregressive mode")
parser.add_argument("--only-hwls", nargs="*", help="List of HWL IDs to forecast")
parser.add_argument("--basin", type=str, help="Basin identifier (e.g., 1, 2)")
args = parser.parse_args()

# Setup basin paths
is_ar_mode = bool(args.ar)
basin_paths = setup_basin_paths(args.basin, "forecast", is_ar_mode)

if basin_paths["river_basin"]:
    print(f"Using basin configuration: {basin_paths}")
    RIVER_BASIN = basin_paths["river_basin"]
    STEP = basin_paths["forecast_step"]
    REGION = basin_paths["region"]
    TARGET_VARIABLE = basin_paths["target_variable"]
    CMD_NUM = basin_paths["cmd_num"]
    MLFLOW_TRAIN = basin_paths["mlflow_train"]
    MLFLOW_FORECAST = basin_paths["mlflow_forecast"]

BASE_DIR = basin_paths["base_dir"]
DATA_DIR = basin_paths["data_dir"]
YML_NAME = f"{STEP}_{REGION}_{TARGET_VARIABLE}_initial.yml"
MLFLOW_EXPERIMENT_NAME = MLFLOW_FORECAST
input_yaml_file = find_yaml_file(DATA_DIR)

# To-do-list: postgreSQL을 쿼리하는 것으로 변경필요
# 1. 기상예측자료 취득에 대한 Options 필요: Option 선택 후에 
# 2. 필요에 따라서 방류량과 기상예측 자료를 
forecast_ori = pd.read_csv(DATA_DIR + f"/forecast_data/time_series/forecast_start_{REGION}_hourly_ori.csv")
forecast_ori.to_csv(DATA_DIR + f"/forecast_data/time_series/forecast_start_{REGION}_hourly.csv", index=False)

class ForecastConfig:
    def __init__(self, is_ar_mode: bool = False):
        self.is_ar_mode = is_ar_mode

        self.step = STEP
        self.region = REGION
        self.target_variable = TARGET_VARIABLE
        #self.obs_overlap = 12

        if self.is_ar_mode:
            self.forecast_window = len(forecast_ori) # 72
            self.forecast_periods = None
            self.yaml_prefix = ""
        else:
            self.forecast_window = 1
            self.forecast_periods = len(forecast_ori) # 72
            self.yaml_prefix = ""

        self.mlflow_experiment_name = f"{self.step}{self.target_variable}_{self.region}"
        self.forecast_start_date = forecast_ori["date"][0] #"2023-06-13 00:00:00"  # TODO: needs to read from database
        self.input_path = self.get_path(f"forecast_data/time_series/input/hourly")
        self.target_path = self.get_path(f"forecast_data/time_series/target/hourly")

    def get_path(self, *path_segments):
        return os.path.join(DATA_DIR, *path_segments)

    def get_yaml_path(self, region: str) -> str:
        if region == self.region:
            return self.get_path(f"{self.yaml_prefix}{self.step}_{region}_hwl_targets_lists.yml")
        raise NotImplementedError(f"Region {region} is not supported.")


@dataclass
class HWL:
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
        return ["date", self.target_code]


# Original function has been replaced with the new signature below


def get_hwls(region: str, is_ar_mode: bool, only_hwls: Optional[List[str]] = None) -> List[HWL]:
    """Load HWL configurations from the YAML file and create HWL instances.
    
    Args:
        region: Region identifier
        is_ar_mode: Whether to use autoregressive mode
        only_hwls: Optional list of HWL IDs to filter by
        
    Returns:
        List of HWL instances
    """
    # Create a ForecastConfig instance
    config = ForecastConfig(is_ar_mode=is_ar_mode)
    
    def ensure_list_or_str_or_none(value) -> Optional[List[str]]:
        """Ensure the value is a list or None if the field is empty or missing."""
        if value is None:
            return None
        if isinstance(value, list):
            return value if value else None
        if isinstance(value, str):
            return [value]
        raise ValueError(f"Invalid value type: {value}. Expected list, str, or None.")

    yaml_path = config.get_yaml_path(region)
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)

    hwls = []
    for item in yaml_config.get(f"{region}_{TARGET_VARIABLE}", []):
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
                train_start_date=body.get("train_start_date"),
                train_end_date=body.get("train_end_date"),
                validation_start_date=body.get("validation_start_date"),
                validation_end_date=body.get("validation_end_date"),
                test_start_date=body.get("test_start_date"),
                test_end_date=body.get("test_end_date"),                
            )
        )
    return hwls


def prepare_data(config: ForecastConfig, region: str, hwl: HWL):
    input_artifact_dir = "input"
    #target_artifact_dir = "target"
    
    file_name = f"{config.step}{region}{hwl.target_code}_hourly.csv"

    forecast_init_file = f"forecast_start_{REGION}_hourly.csv"
    sequence_file = f"sequence_{REGION}_hourly.csv"

    # 입력/타겟 데이터 준비
    forecast_init_csv_path = config.get_path(f"forecast_data/time_series/{forecast_init_file}")
    sequence_csv_path = config.get_path(f"forecast_data/time_series/{sequence_file}")

    forecast_init_df = pd.read_csv(forecast_init_csv_path, index_col=0)
    sequence_df = pd.read_csv(sequence_csv_path, index_col=0)

    combined_df_init = pd.concat([sequence_df, forecast_init_df])

    input_columns = hwl.input_columns
    target_columns = hwl.target_columns

    if config.is_ar_mode:
        input_df = combined_df_init[input_columns[:-1]].set_index("date")
        target_df = combined_df_init[target_columns].set_index("date")
        target = target_df.columns[0]
        AR_col = target + "_AR"
        input_df[AR_col] = target_df[target].shift(+1)
        stats_suffix = ""
    else:
        input_df = combined_df_init[input_columns].set_index("date")
        target_df = combined_df_init[target_columns].set_index("date")
        input_df.index = pd.to_datetime(input_df.index)
        target_df.index = pd.to_datetime(target_df.index)
        stats_suffix = ""

    # MLflow에 통계 정보 저장
    mlflow.log_param("input_shape", input_df.shape)

    # 데이터 디렉토리 생성
    input_st_path = os.path.join(config.input_path, hwl.id)
    os.makedirs(input_st_path, exist_ok=True)

    # CSV 저장
    input_csv_path = f"{input_st_path}/{file_name}"
    input_df.to_csv(input_csv_path)

    # 데이터 통계 정보를 JSON으로 저장
    input_stats = input_df.describe().to_json()
    mlflow.log_dict(
        json.loads(input_stats), f"{input_artifact_dir}/input{stats_suffix}_statistics.json"
    )

    # 각 컬럼별 null 값 로깅
    input_null_counts = input_df.isnull().sum().to_dict()

    for col, count in input_null_counts.items():
        mlflow.log_metric(f"input{stats_suffix}_null_{col}", count)

    # MLflow dataset 로깅
    input_dataset = mlflow.data.from_pandas(
        input_df,
        name=f"input{stats_suffix}_dataset",
        source=input_csv_path,
    )

    mlflow.log_input(input_dataset, context="forecast")
    mlflow.log_artifact(input_csv_path, artifact_path=input_artifact_dir)

    return input_df, target_columns, forecast_init_df, sequence_df, combined_df_init


def setup_mlflow_experiment(experiment_name: str):
    """MLflow 실험 설정"""
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
    """
    주어진 실험 ID와 parent_run_name으로 parent_run_id를 가져오거나, 없으면 새로 생성
    """
    # Parent Run 검색
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_name = '{parent_run_name}'",
        output_format="list",
    )

    # Parent Run이 존재하면 ID 반환
    if runs:
        print(f"Found existing parent run: {parent_run_name}")
        return runs[0].info.run_id

    # Parent Run이 없으면 새로 생성
    print(f"Parent run not found. Creating a new parent run: {parent_run_name}")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=parent_run_name,
    ) as parent_run:
        return parent_run.info.run_id


def make_forecast_one_step(model, valid_date, scaled_data, nh_config):
    input_data = scaled_data.loc[:valid_date]
    input_data_seq = input_data.iloc[-nh_config.seq_length :, :]
    input_data = np.expand_dims(input_data_seq.values, axis=0)
    tensor_data = torch.tensor(input_data, dtype=torch.float32)
    # Move tensor to the same device as the model
    device = next(model.parameters()).device
    tensor_data = tensor_data.to(device)
    out = model({"x_d": tensor_data})

    scaled_yhat = out["y_hat"]
    return scaled_yhat.flatten()[-1].item()


def get_forecasts_in_forecast_window_ar(
    model, forecast_date, scaled_data, nh_config, scaler, forecast_window
):
    scaled_data_date = scaled_data.copy()
    yhats = []

    for i in range(forecast_window):
        data = {}

        # make forecast
        valid_date = scaled_data_date.loc[forecast_date:].index[i]
        yhat_i = make_forecast_one_step(model, valid_date, scaled_data_date, nh_config)

        # add forecast to the data_dict
        data["forecast_time"] = forecast_date
        data["valid_time"] = valid_date
        data["yhat"] = yhat_i
        data["lead_time"] = i
        yhats.append(data)

        # 🔥 AR 모드: 다음 시점의 AR 변수 업데이트
        if i+1 < forecast_window:
            next_valid_date = scaled_data_date.loc[forecast_date:].index[i + 1]
        else: 
            pass
        
        AR_variable = [v for v in nh_config.dynamic_inputs if "AR" in v][0]
        scaled_data_date.loc[next_valid_date, AR_variable] = yhat_i

    df_yhats = pd.DataFrame(yhats)
    target_var = nh_config.target_variables[0]
    y_scale = scaler["xarray_feature_scale"]["data_vars"][target_var]["data"]
    y_center = scaler["xarray_feature_center"]["data_vars"][target_var]["data"]
    df_yhats.loc[:, "yhat_unscaled"] = df_yhats["yhat"] * y_scale + y_center

    return df_yhats


def get_forecasts_in_forecast_window_normal(
    model, forecast_date, scaled_data, nh_config, scaler, forecast_window
):
    scaled_data_date = scaled_data.copy()
    yhats = []

    for i in range(forecast_window):
        data = {}

        # make forecast
        valid_date = scaled_data_date.loc[forecast_date:].index[i]
        yhat_i = make_forecast_one_step(model, valid_date, scaled_data_date, nh_config)

        # add forecast to the data_dict
        data["forecast_time"] = forecast_date
        data["valid_time"] = valid_date
        data["yhat"] = yhat_i
        data["lead_time"] = i + 1
        yhats.append(data)

        next_row = pd.DataFrame(
            [[yhat_i]], columns=[nh_config.target_variables[0]], index=[valid_date]
        )
        scaled_data_date = pd.concat([scaled_data_date, next_row])

    df_yhats = pd.DataFrame(yhats)
    target_var = nh_config.target_variables[0]
    y_scale = scaler["xarray_feature_scale"]["data_vars"][target_var]["data"]
    y_center = scaler["xarray_feature_center"]["data_vars"][target_var]["data"]
    df_yhats.loc[:, "yhat_unscaled"] = df_yhats["yhat"] * y_scale + y_center
    return df_yhats


def run(
    config: ForecastConfig,
    experiment_id: str,
    hwl: HWL,
    region: str,
    model_name: str,
):
    parent_run_name = f"{hwl.target_name}({hwl.target_code.split('_')[0]})"
    if config.is_ar_mode:
        run_name=f"ar_{hwl.target_name}({hwl.target_code.split('_')[0]})_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    else:
        run_name=f"{hwl.target_name}({hwl.target_code.split('_')[0]})_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"

    parent_run_id = get_or_create_parent_run_id(experiment_id, parent_run_name)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        nested=True,
        parent_run_id=parent_run_id,
        tags={
            "model": model_name,
            "region": region,
            "is_ar_mode": str(config.is_ar_mode),
        },
    ):
        # 0. 실험 기본 정보 로깅
        mlflow.log_params(
            {
                "target_code": hwl.target_code,
                "model": model_name,
                "region": region,
                "is_ar_mode": config.is_ar_mode,
            }
        )

        # 1. 데이터 준비
        input_df, taget_col, forecast_init_df, sequence_df, combined_df = prepare_data(config=config, region=region, hwl=hwl)

        mr = ModelRegistry(
            region=region,
            hwl_id=hwl.id,
            name=hwl.target_name,
            is_ar_mode=config.is_ar_mode,
            verbose=True
        )

        X_scl = input_df.copy()
        for col in X_scl:
            #print(f"mr.scaler:{mr.scaler}")
            #print(f'mr.scaler["xarray_feature_scale"]["data_vars"]:{mr.scaler["xarray_feature_scale"]["data_vars"]}')

            col_scale = mr.scaler["xarray_feature_scale"]["data_vars"][col]["data"]
            col_center = mr.scaler["xarray_feature_center"]["data_vars"][col]["data"]
            X_scl.loc[:, col] = (input_df.loc[:, col] - col_center) / col_scale

        if not config.is_ar_mode and hwl.input_hwl_code:
            predicted_hwl_values = {}
            for input_code in hwl.input_hwl_code:
                if input_code in predicted_hwl_values:
                    pred_series = predicted_hwl_values[input_code]
                    last_72_idx = X_scl.index[-72:]
                    pred_series = pred_series.reindex(last_72_idx)

                    # 예측값에 대해 스케일링 적용
                    col_scale = mr.scaler["xarray_feature_scale"]["data_vars"][input_code]["data"]
                    col_center = mr.scaler["xarray_feature_center"]["data_vars"][input_code][
                        "data"
                    ]
                    scaled_pred_series = (pred_series - col_center) / col_scale

                    # 스케일링된 예측값을 입력으로 사용
                    X_scl.loc[last_72_idx, input_code] = scaled_pred_series.values
                    print(
                        f"[DEBUG] Forecasted last 72hrs of '{input_code}' in input to predict '{hwl.target_code}'"
                    )

                    forecasted_info = {
                        "input_station": input_code,
                        "target_station": hwl.target_code,
                        "forecast_period": "72 hours",
                        "forecast_start": last_72_idx[0].strftime("%Y-%m-%d %H:%M:%S"),
                        "forecast_end": last_72_idx[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    mlflow.log_params(
                        {
                            f"forecasted_input.{input_code}.target": hwl.target_code,
                            f"forecasted_input.{input_code}.period": "72 hours",
                            f"forecasted_input.{input_code}.time_range": f"{last_72_idx[0].strftime('%Y-%m-%d %H:%M')} to {last_72_idx[-1].strftime('%Y-%m-%d %H:%M')}",
                        }
                    )
                    mlflow.log_dict(forecasted_info, f"forecasted_data/{input_code}_details.json")
                else:
                    print(f"[WARNING] Missing prediction for '{input_code}'")

        # 검증 시간 설정
        forecast_start_time = pd.to_datetime(forecast_ori["date"][0])
        forecast_end_time = forecast_start_time + timedelta(hours=71)
        forecast_end_time = forecast_end_time.strftime('%Y-%m-%d %H:%M')

        validation_times = X_scl.loc[
            forecast_ori["date"][0] : forecast_end_time
        ].index
        test_date = pd.DataFrame(validation_times)
        test_date_index = test_date.index
        test_date = test_date.set_index("date")
        test_date["index"] = test_date_index

        validation_dfs = []
        if config.is_ar_mode:
            for t in validation_times[
                test_date["index"][config.forecast_start_date] : test_date["index"][
                    config.forecast_start_date
                ]
                + 1
            ]:  # [24*10*30 + 500 : 24*10*30 + 600]
                yhats_t = get_forecasts_in_forecast_window_ar(
                    mr.model, t, X_scl, mr.config, mr.scaler, config.forecast_window
                )
                validation_dfs.append(yhats_t)
        else:
            for t in validation_times[
                test_date["index"][config.forecast_start_date] : test_date["index"][
                    config.forecast_start_date
                ]
                + config.forecast_periods
            ]:
                yhats_t = get_forecasts_in_forecast_window_normal(
                    mr.model, t, X_scl, mr.config, mr.scaler, config.forecast_window
                )
                validation_dfs.append(yhats_t)

        val_results = pd.concat(validation_dfs)
        val_results = pd.merge(val_results, X_scl.reset_index(), left_on="valid_time", right_on="date")
        val_results[taget_col[1]] = val_results["yhat_unscaled"].values
        forecast_init_df[taget_col[1]] = val_results["yhat_unscaled"].values

        combined_df.loc[combined_df.index[-72:], "yhat_unscaled"] = val_results["yhat_unscaled"].values
        dt = pd.to_datetime(config.forecast_start_date) - timedelta(hours=5)
        graph_start_time = dt.strftime('%Y-%m-%d %H:%M')
        combined_df = combined_df.set_index("date")
        combined_df["date"] = combined_df.index
        val_results = combined_df.loc[graph_start_time:]

        forecast_df = combined_df[["yhat_unscaled"]].loc[forecast_ori["date"][0]:]
        forecast_df = forecast_df.rename(columns={'yhat_unscaled': taget_col[1]})
        
        forecast_artifact_dir = "forecast"
        mlflow.log_param("forecast_shape", forecast_df.shape)
        forecast_st_path = os.path.join(config.target_path, hwl.id)
        os.makedirs(forecast_st_path, exist_ok=True)
        file_name = f"{config.step}{region}_hourly.csv"
        forecast_csv_path = f"{forecast_st_path}/{file_name}"
        forecast_df.to_csv(forecast_csv_path)
        target_stats = forecast_df.describe().to_json()
        mlflow.log_dict(
        json.loads(target_stats), f"{forecast_artifact_dir}/target_statistics.json"
        )
        target_null_counts = forecast_df.isnull().sum().to_dict()
        for col, count in target_null_counts.items():
            mlflow.log_metric(f"target_null_{col}", count)
        target_dataset = mlflow.data.from_pandas(
            forecast_df,
            name=f"target_dataset",
            source=forecast_csv_path,
        )
        mlflow.log_input(target_dataset, context="forecast")
        mlflow.log_artifact(forecast_csv_path, artifact_path=forecast_artifact_dir)

        # 일반 모드에서만 X 파라미터 전달
        if config.is_ar_mode:
            forecast_init_df.to_csv(DATA_DIR + f"/forecast_data/time_series/forecast_start_{REGION}_hourly.csv")
            forecast(val_results, hwl.target_code, hwl.target_name, input_df)
        else:
            forecast_init_df.to_csv(DATA_DIR + f"/forecast_data/time_series/forecast_start_{REGION}_hourly.csv")
            forecast(val_results, hwl.target_code, hwl.target_name, input_df)

        # 5. GPU 메모리 정리
        torch.cuda.empty_cache()


def main(region: str, is_ar_mode: bool = False, only_hwls: list[str] = None) -> None:
    """Main entry point for the forecasting process.

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
        experiment_id = setup_mlflow_experiment(MLFLOW_FORECAST)
        if only_hwls:
            only_hwls = set(only_hwls)
        hwls = get_hwls(region, is_ar_mode, only_hwls)
        for hwl in hwls:
            try:
                # Create a ForecastConfig instance
                config = ForecastConfig(is_ar_mode=is_ar_mode)
                run(config, experiment_id, hwl, region, model)
            except Exception:
                print("An error occurred:")
                traceback.print_exc()

if __name__ == "__main__":
    main(REGION, args.ar, args.only_hwls)
