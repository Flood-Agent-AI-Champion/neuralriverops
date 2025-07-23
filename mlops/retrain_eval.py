"""Neural River Operations evaluation and visualization module.

This module handles the evaluation and visualization of water level prediction models,
including metric calculation and plot generation for both matplotlib and plotly.
"""

# Standard library imports
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Third-party imports
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config

# Local imports
from basin_config import setup_basin_paths, find_yaml_file

# Setup basin paths (can be overridden by calling functions)
def initialize_basin_config(basin_id: str = None, is_ar_mode: bool = False):
    """Initialize basin configuration for evaluation.
    
    Args:
        basin_id: Basin identifier (e.g., "1", "2")
        is_ar_mode: Whether running in autoregressive mode
    """
    global BASE_OUT_DIR, region_file, first_line, STEP, TARGET_VARIABLE, REGION, y_value, UNIT
    
    # Setup basin paths
    basin_paths = setup_basin_paths(basin_id, is_ar_mode)

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
    
    if TARGET_VARIABLE == "hwl":
        y_value = "WaterLevel"
        UNIT = "m"
    elif TARGET_VARIABLE == "hflw":
        y_value = "Streamflow"
        UNIT = "cms"

PLOT_DPI = 120
PLOT_FIGSIZE = (16, 7)
PLOT_MARKER_SIZE = 10
PLOT_FONT_SIZE = 15
PLOT_LEGEND_FONT_SIZE = 20


class Visualizer:
    """Class for creating and managing visualization plots.

    This class provides static methods for creating both matplotlib and plotly
    visualizations of water level predictions.
    """

    @staticmethod
    def create_matplotlib_plot(
        plot_name: str,
        qobs: pd.DataFrame,
        qsim: pd.DataFrame,
        input_data,
        target_code: str,
        period: str,
        nse: float,
    ) -> None:
        """Create and save a matplotlib plot of observed vs simulated water levels.

        Args:
            plot_name: Name for the plot file
            qobs: DataFrame containing observed water levels
            qsim: DataFrame containing simulated water levels
            target_code: Code identifying the target station
            period: Time period being plotted (train/validation/test)
            nse: Nash-Sutcliffe Efficiency score
        """

        input_vars = [col for col in input_data.columns if col != "date"]
        n_input = len(input_vars)
        total_rows = 1 + n_input

        fig, axes = plt.subplots(total_rows, 1, figsize=(10, 4 * total_rows), sharex=True)

        # --- x축 범위 계산 (관측/예측값이 하나라도 존재하는 구간) ---
        obs_dates = qobs['date'].values
        sim_dates = qsim['date'].values
        obs_vals = qobs.values[:, 0]
        sim_vals = qsim.values[:, 0]
        valid_mask = (~np.isnan(obs_vals)) | (~np.isnan(sim_vals))
        if valid_mask.any():
            min_idx = np.where(valid_mask)[0][0]
            max_idx = np.where(valid_mask)[0][-1]
            xlim = (obs_dates[min_idx], obs_dates[max_idx])
        else:
            xlim = (obs_dates[0], obs_dates[-1])

        # 1. 관측/예측값
        axes[0].plot(obs_dates, obs_vals, label="Observed", c="b")
        axes[0].scatter(sim_dates, sim_vals, label="Simulated", c="r", s=PLOT_MARKER_SIZE)
        axes[0].legend(fontsize=PLOT_FONT_SIZE)
        axes[0].set_ylabel(f"{y_value} ({UNIT})")
        axes[0].set_title(f"{REGION}, {target_code[0:7]} Station\n{period.capitalize()} period - Hourly NSE {nse:.3f}")
        axes[0].set_xlim(xlim)

        # 2. input 변수별 subplot (전체 x축 범위)
        for i, var in enumerate(input_vars):
            axes[i+1].plot(input_data["date"], input_data[var], label=var)
            axes[i+1].set_ylabel(var)
            axes[i+1].legend(fontsize=10)

        plt.xticks(rotation=45)
        plt.tight_layout()

        with tempfile.TemporaryDirectory() as tmp_dir:
            png_path = os.path.join(tmp_dir, f"{plot_name}.png")
            plt.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
            mlflow.log_artifact(png_path, artifact_path="plots")
        plt.close()

    @staticmethod
    def create_plotly_plot(
        plot_name: str,
        qobs: pd.DataFrame,
        qsim: pd.DataFrame,
        input_data,
        target_code: str,
        period: str,
        nse: float,
    ) -> None:
        """Create and save a plotly plot of observed vs simulated water levels.

        Args:
            plot_name: Name for the plot file
            qobs: DataFrame containing observed water levels
            qsim: DataFrame containing simulated water levels
            target_code: Code identifying the target station
            period: Time period being plotted (train/validation/test)
            nse: Nash-Sutcliffe Efficiency score
        """

        # 각 기간의 x축 범위
        period_dates = pd.to_datetime(qobs.coords["date"].values)
        xlim = (period_dates[0], period_dates[-1])
        obs_vals = qobs.values[:, 0]
        sim_vals = qsim.values[:, 0]

        input_vars = [col for col in input_data.columns if col != "date"]
        n_input = len(input_vars)
        total_rows = 1 + n_input

        base_width = 1500
        base_height = 400 * total_rows

        fig = make_subplots(
            rows=total_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05
        )

        # 관측값
        fig.add_trace(
            go.Scatter(
                x=period_dates,
                y=obs_vals,
                name=f"{y_value} (observation) ({UNIT})",
                mode="lines+markers",
                marker=dict(symbol="diamond", size=3),
                line=dict(color="royalblue"),
                showlegend=True
            ),
            row=1, col=1,
        )

        # 예측값
        fig.add_trace(
            go.Scatter(
                x=period_dates,
                y=sim_vals,
                name=f"{y_value} (simulation) ({UNIT})",
                mode="lines+markers",
                marker=dict(symbol="diamond", size=3),
                line=dict(color="tomato"),
                showlegend=False
            ),
            row=1, col=1,
        )
        
        # 입력 변수 subplot
        colors = ['#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896', '#98df8a', '#ffbb78', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#393b79', '#637939']
        for i, var in enumerate(input_vars):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=period_dates,
                    y=input_data[var].values,
                    name=f"Input:{var}",
                    mode="lines+markers",
                    marker=dict(symbol="diamond", size=3, color=color),
                    line=dict(color=color),
                    showlegend=True
                ),
                row=i+2, col=1,
            )
            
        fig.update_layout(
            title_text=f"<b>{y_value}, {target_code[0:7]} Station\n{period.capitalize()} period - Hourly NSE {nse:.3f}</b>",
            xaxis_title="<b>Date and Time</b>",
            yaxis_title=f"<b>{y_value}({UNIT})</b>",
            width=base_width,
            height=base_height,
            margin=go.layout.Margin(l=50, r=50, b=100, t=70, pad=4),
            legend=dict(font=dict(size=14)),
            font=dict(size=13),
            showlegend=True,
        )

        # 1번 subplot(관측/예측)만 x축 범위 제한
        fig.update_xaxes(
            range=[xlim[0], xlim[1]],
            tickformat="%Y-%m-%d"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = os.path.join(tmp_dir, f"{plot_name}.html")
            fig.write_html(html_path)
            mlflow.log_artifact(html_path, artifact_path="plots")


def evaluate_period(cfg: Config, period: str, metrics: List[str]) -> Dict[str, Any]:
    """Evaluate model performance for a specific time period.

    Args:
        cfg: Configuration object containing model settings
        period: Time period to evaluate (train/validation/test)
        metrics: List of metrics to calculate

    Returns:
        Dict[str, Any]: Dictionary containing evaluation results
    """
    tester = get_tester(cfg=cfg, run_dir=cfg.run_dir, period=period, init_model=True)
    return tester.evaluate(save_results=False, metrics=metrics)


def process_metrics(results: Dict[str, Any], period: str) -> Dict[str, float]:
    """Process and extract metrics from evaluation results.

    Args:
        results: Dictionary containing evaluation results
        period: Time period of the evaluation

    Returns:
        Dict[str, float]: Dictionary of processed metrics with period prefix
    """
    return {
        f"{period}_{k}": float(v)
        for k, v in results[f"{STEP}_{REGION}"]["1H"].items()
        if isinstance(v, (int, float, np.number))
    }


def evaluate_and_visualize(target_code: str, cfg: Config, metrics: List[str]) -> Dict[str, Any]:
    """Evaluate model performance and create visualizations for all periods.

    Args:
        target_code: Code identifying the target station
        cfg: Configuration object containing model settings
        metrics: List of metrics to calculate

    Returns:
        Dict[str, Any]: Dictionary containing evaluation results for all periods
    """
    periods = ("train", "validation", "test")
    results = {}
    all_metrics = {}

    for period in periods:
        base_plot_name = f"{target_code}_{period}"

        # Evaluate period
        period_results = evaluate_period(cfg, period, metrics)
        print(f"Evaluate period results: {period_results}")
        results[period] = period_results

        # Process and log metrics
        period_metrics = process_metrics(period_results, period)
        all_metrics.update(period_metrics)
        mlflow.log_metrics(period_metrics)

        # Extract observed and simulated values
        hourly_qobs = period_results[f"{STEP}_{REGION}"]["1H"]["xr"][target_code + "_obs"]
        hourly_qsim = period_results[f"{STEP}_{REGION}"]["1H"]["xr"][target_code + "_sim"]
        nse = period_results[f"{STEP}_{REGION}"]["1H"]["NSE"]

        # Get input data
        input_data = pd.read_csv(os.path.join(cfg.data_dir, "time_series/input/hourly/", cfg.target_variables[0], f"{STEP}_{REGION}_hourly.csv"))
        input_data = input_data.set_index("date")
        input_data = input_data.loc[str(pd.Timestamp(hourly_qobs["date"].values[0])):str(pd.Timestamp(hourly_qobs["date"].values[-1]))]
        input_data = input_data[cfg.dynamic_inputs]
        input_data["date"] = pd.to_datetime(input_data.index)

        # Create visualizations
        Visualizer.create_matplotlib_plot(
            f"{base_plot_name}_hourly",
            hourly_qobs,
            hourly_qsim,
            input_data,
            target_code,
            period,
            nse
        )

        Visualizer.create_plotly_plot(
            f"{base_plot_name}_hourly",
            hourly_qobs,
            hourly_qsim,
            input_data,
            target_code,
            period,
            nse
        )

    # Save and log metrics
    metrics_path = "metrics.json"
    pd.DataFrame([all_metrics]).to_json(metrics_path)
    mlflow.log_artifact(metrics_path)
    os.remove(metrics_path)

    return results
