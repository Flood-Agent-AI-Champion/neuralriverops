"""River basin configuration utilities.

This module provides common functions for loading and processing river basin
configuration from YAML files.
"""

# Standard library imports
import os
import json
from typing import Dict, Any, Optional, Tuple


# Third-party imports
import yaml

def find_value(data: Any, target_value: Any) -> list[Dict[str, Any]]:
    """Find a specific value in JSON data and return contextual information.
    
    Args:
        data: JSON data to search through
        target_value: Value to search for
        
    Returns:
        List of dictionaries containing path and context information for each match
    """
    results = []
    
    def search(data, path=None):
        """Recursive function to search for values"""
        path = path or []
        
        if isinstance(data, dict):
            # Search through dictionary
            for key, value in data.items():
                current_path = path + [key]
                
                # Found value
                if value == target_value:
                    results.append({
                        'path': current_path,
                        'context': data
                    })
                
                # Search nested structures
                if isinstance(value, (dict, list)):
                    search(value, current_path)
                    
        elif isinstance(data, list):
            # Search through list
            for i, item in enumerate(data):
                current_path = path + [i]
                
                # Found value
                if item == target_value:
                    results.append({
                        'path': current_path,
                        'context': {str(j): v for j, v in enumerate(data)}
                    })
                
                # Search nested structures
                if isinstance(item, (dict, list)):
                    search(item, current_path)
    
    # Start search
    search(data)
    return results

def load_basins_json_config() -> Dict[str, Any]:
    """Load basin configuration from basins_config.json file.
    
    Returns:
        Dict containing basin information from JSON configuration
    """
    # 프로젝트 루트 디렉토리에서 경로 지정 (mlops 디렉토리 상위로 올라감)
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "basins_config.json"))
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Basin configuration file not found: {config_path}")


def find_basin_by_cmd_num(cmd_num: str) -> Tuple[Optional[str], Optional[str]]:
    """Find basin by cmd_num in basins_config.json.
    
    Args:
        cmd_num: Command number to look up
        
    Returns:
        Tuple of (river_basin, downstream) or (None, None) if not found
    """
    try:
        cmd_num = int(cmd_num)
        basins_config = load_basins_json_config()
        
        # Search for the basin with matching cmd_num
        for river_basin, basins in basins_config.items():
            for downstream, basin_info in basins.items():
                if basin_info.get("cmd_num") == cmd_num:
                    return river_basin, downstream
        
        # If not found
        print(f"Warning: No basin found with cmd_num={cmd_num}")
        return None, None
    except Exception as e:
        print(f"Error finding basin by cmd_num: {e}")
        return None, None


def setup_basin_paths(basin_id: Optional[str] = None, step: Optional[str] = None, is_ar_mode: bool = False) -> Dict[str, str]:
    """Setup basin-specific paths based on basin configuration.
    
    Args:
        basin_id: Basin identifier (e.g., "1", "yddd", "2", "dccd")
        is_ar_mode: Whether running in autoregressive mode
        
    Returns:
        Dict containing path configurations
    """
    # 프로젝트 루트 디렉토리에서 경로 지정 (mlops 디렉토리 상위로 올라감)
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "basins_config.json"))

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Basin configuration file not found: {config_path}")

    if basin_id:
        try:
            search_value = int(basin_id) if basin_id.isdigit() else basin_id
            result = find_value(config, search_value)
            river_basin = result[0]["path"][0]
            train_step = result[0]["context"]["train_step"]
            forecast_step = result[0]["context"]["forecast_step"]
            region = result[0]["context"]["region"]
            target_variable = result[0]["context"]["target_variable"]
            cmd_num = result[0]["context"]["cmd_num"]
            mlflow_train = result[0]["context"]["mlflow_train"]
            mlflow_forecast = result[0]["context"]["mlflow_forecast"]
            if is_ar_mode == True:
                base_dir = os.path.abspath(os.path.join("data", river_basin, region))
                if step == "retrain":
                    data_dir = os.path.abspath(os.path.join("data", river_basin, region, "retrain", "ar_data"))
                else:
                    data_dir = os.path.abspath(os.path.join("data", river_basin, region, "forecast", "ar_data"))
            else:
                base_dir = os.path.abspath(os.path.join("data", river_basin, region))
                if step == "retrain":
                    data_dir = os.path.abspath(os.path.join("data", river_basin, region, "retrain", "data"))
                else:
                    data_dir = os.path.abspath(os.path.join("data", river_basin, region, "forecast", "data"))
                
            return {
                "river_basin": river_basin,
                "train_step": train_step,
                "forecast_step": forecast_step,
                "region": region,
                "target_variable": target_variable,
                "cmd_num": cmd_num,
                "mlflow_train": mlflow_train,
                "mlflow_forecast": mlflow_forecast,
                "base_dir": base_dir,
                "data_dir": data_dir
            }
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading basin configuration: {e}")
            raise
    else:
        print(f"Please provide a valid basin identifier")
        raise


def find_yaml_file(data_dir: str) -> str:
    """Find the region yaml file in the specified data directory.
    
    Args:
        data_dir: Directory to search for region yaml files
        
    Returns:
        Path to the region yaml file
        
    Raises:
        FileNotFoundError: If no matching file is found
    """
    # Ensure the directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        raise FileNotFoundError(f"Directory does not exist: {data_dir}. Created it, but no YAML files found.")
    
    # Look for YAML files with "lists" in the name
    input_yaml_file = None
    for filename in os.listdir(data_dir):
        if "lists" in filename and (filename.endswith('.yml') or filename.endswith('.yaml')):
            input_yaml_file = os.path.join(data_dir, filename)
            break  # Stop after finding the first lists file
    
    # If no YAML file found in the specified directory, look in parent directories
    if input_yaml_file is None:
        parent_dir = os.path.dirname(data_dir)
        for root, _, files in os.walk(parent_dir):
            for filename in files:
                if "lists" in filename and (filename.endswith('.yml') or filename.endswith('.yaml')):
                    input_yaml_file = os.path.join(root, filename)
                    print(f"Found YAML file in parent directory: {input_yaml_file}")
                    return input_yaml_file
    
    if input_yaml_file is None:
        raise FileNotFoundError(f"There is no target lists file in {data_dir} or parent directories")
    
    return input_yaml_file

def find_region_file(data_dir: str) -> str:
    """Find the region file in the specified data directory.
    
    Args:
        data_dir: Directory to search for region files
        
    Returns:
        Path to the region file
    """
    region_file = None
    for filename in os.listdir(data_dir):
        if "lists" in filename:
            region_file = os.path.join(data_dir, filename)
            break  # 첫 번째 lists 파일을 찾으면 중단
    
    if region_file is None:
        raise FileNotFoundError(f"There is no target lists file in {data_dir}")
    
    return region_file 