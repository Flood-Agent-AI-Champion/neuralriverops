import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import pandas as pd
import argparse

# Local imports
from basin_config import setup_basin_paths, find_yaml_file

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

if TARGET_VARIABLE == "hwl":
    y_value = "WaterLevel"
    UNIT = "m"
elif TARGET_VARIABLE == "hflw":
    y_value = "Streamflow"
    UNIT = "cms"

class Visualizer:

    @staticmethod
    def create_plotly_plot(plot_name:str, qobs, qsim, target_code:str, target_name:str, input_df=None) -> None:
        input_vars = input_df.columns if input_df is not None else []
        n_input = len(input_vars)
        total_rows = 1 + n_input

        base_width = 1200
        base_height_per_plot = 300
        total_plots = 2 + len(input_vars)
        qobs["date"] = pd.to_datetime(qobs["date"], errors='coerce')
        qsim["date"] = pd.to_datetime(qsim["date"], errors='coerce')

        fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04)

        fig.add_trace(
            go.Scatter(
                x=qobs["date"].dt.strftime("%d-%m-%Y %H-%M"),
                y=qobs[target_code],
                name=f"{TARGET_VARIABLE} (observation) ({UNIT})",
                mode="lines+markers",
                marker=dict(symbol="diamond", size=5)
            ), 
            row=1 ,col=1,
            #secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=qsim["date"].dt.strftime("%d-%m-%Y %H-%M"),
                y=qsim["yhat_unscaled"],
                name=f"{TARGET_VARIABLE} (simulation) ({UNIT})",
                mode="lines+markers",
                marker=dict(symbol="diamond", size=5)
            ), 
            row=1, col=1,
            #secondary_y=False,
        )

        if input_df is not None:
            for i, var in enumerate(input_vars):
                fig.add_trace(
                    go.Scatter(
                        x=input_df.index,
                        y=input_df[var],
                        name=f"Input:{var}",
                        mode="lines+markers",
                        marker=dict(symbol="diamond", size=5)
                    ), 
                    row=i+2, col=1,
                    #secondary_y=True,
                )
        
        fig.update_layout(
            title_text=f"<b>{target_name}({target_code[0:7]}) Station</b>",
            xaxis_title="<b>Date and Time",
            yaxis_title=f"<b>{TARGET_VARIABLE} ({UNIT})</b>",
            width=base_width,
            height=base_height_per_plot*total_plots,
            legend=dict(yanchor="top", y=1, xanchor="left", x=1.05, font=dict(size=14)),
            margin=go.layout.Margin(l=50, r=50, b=100, t=70, pad=4),
            autosize=True,
        )

        with tempfile.TemporaryDirectory() as tem_dir:
            html_path = os.path.join(tem_dir, f"{plot_name}.html")
            fig.write_html(html_path)
            mlflow.log_artifact(html_path, artifact_path="plots")


def forecast(output, target_code, target_name, input_df=None):
    """모델 평가 및 시각화 메인 함수"""
    # DATA_DIR와 REGION이 정의되어 있는지 확인
    #if DATA_DIR is None or REGION is None:
    #    # 기본 basin_id를 사용하여 초기화
    #    initialize_basin_config()
        
    # 예측 시작일 자동 추출
    forecast_start_dt = pd.to_datetime(output["date"].iloc[0])

    #관측/예측 추출
    hourly_qobs = output[["date", target_code]].copy()
    hourly_qsim = output[["date", "yhat_unscaled"]].copy()
    hourly_qobs["date"] = pd.to_datetime(hourly_qobs["date"], errors='coerce')
    hourly_qsim["date"] = pd.to_datetime(hourly_qsim["date"], errors='coerce')

    # Sequence_date 
    seq_obs_file = os.path.join(DATA_DIR, f"forecast_data/time_series/sequence_{REGION}_hourly.csv")
    seq_obs_df = pd.read_csv(seq_obs_file)
    seq_obs_df["date"] = pd.to_datetime(seq_obs_df["date"])
    seq_qobs = seq_obs_df[seq_obs_df["date"] < forecast_start_dt][["date", target_code]]
    seq_qsim = hourly_qsim[hourly_qsim["date"] < forecast_start_dt]
    
    # input_df의 인덱스를 datetime으로 변환
    if input_df is not None:
        input_df.index = pd.to_datetime(input_df.index)
        seq_input = input_df[input_df.index < forecast_start_dt]
    else:
        seq_input = None

    # Forecast_date
    fore_qobs = hourly_qobs[hourly_qobs["date"] >= forecast_start_dt]
    fore_qsim = hourly_qsim[hourly_qsim["date"] >= forecast_start_dt]
    fore_input = input_df[input_df.index >= forecast_start_dt] if input_df is not None else None

    # 시각화 생성
    Visualizer.create_plotly_plot(
        f"{target_name}_{target_code}_sequence",
        seq_qobs,
        seq_qsim,
        target_code,
        target_name,
        seq_input
    )

    Visualizer.create_plotly_plot(
        f"{target_name}_{target_code}_forecast",
        fore_qobs,
        fore_qsim,
        target_code,
        target_name,
        fore_input
    )

    return {}