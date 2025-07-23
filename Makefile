# Configuration Variables
PYTHON := python
MLFLOW_HOST := 0.0.0.0
MLFLOW_PORT := 5000
ENV_FILE := $(shell pwd)/.mlflow_env
PID_FILE := mlflow.pid
LOG_FILE := mlflow.log

# Script Names
RETRAIN_SCRIPT := mlops/retrain.py
FORECAST_SCRIPT := mlops/forecast.py

# MLflow Server Management: MLflow 서버 시작
start-floodai:
	@if [ -f $(PID_FILE) ]; then \
		echo "Stopping existing MLflow server..."; \
		pid=`cat $(PID_FILE)`; \
		pkill -P $$pid 2>/dev/null || true; \
		kill -9 $$pid 2>/dev/null || true; \
		rm -f $(PID_FILE); \
		sleep 2; \
	fi
	@echo "Starting MLflow server with basic-auth..."
	@nohup mlflow server \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT) \
		--app-name basic-auth > $(LOG_FILE) 2>&1 & \
	echo $$! > $(PID_FILE)
	@sleep 3  # Wait for server to fully start

# MLflow 서버 종료
stop-floodai:
	@if [ -f $(PID_FILE) ]; then \
		echo "Stopping MLflow server..."; \
		pid=`cat $(PID_FILE)`; \
		pkill -P $$pid 2>/dev/null || true; \
		kill -9 $$pid 2>/dev/null || true; \
		rm -f $(PID_FILE); \
		echo "Checking for remaining MLflow processes..."; \
		pgrep -f "mlflow server" | xargs kill -9 2>/dev/null || true; \
	else \
		echo "MLflow server is not running."; \
	fi
	@echo "Ensuring port $(MLFLOW_PORT) is free..."
	@lsof -ti:$(MLFLOW_PORT) | xargs kill -9 2>/dev/null || true

# 실시간 로그 보기
logs:
	@echo "Showing real-time logs..."
	@tail -f $(LOG_FILE)

# Normal and Autoregressive LSTM Mode Training Command
retrain:
	@echo "=== Starting New LSTM Retrain Run ==="
	@echo "=== Authenticating User ==="
	$(PYTHON) login.py
	@echo "=== Running Retraining Script ==="
	@if [ -z "$(basin)" ]; then \
		echo "Error: basin parameter is required!"; \
		exit 1; \
	fi

	@if [ -f $(ENV_FILE) ]; then \
		set -a; \
		. $(ENV_FILE); \
		echo "Using basin: $(basin)"; \
		if [ "$(AR)" = "true" ] || [ "$(AR)" = "yes" ] || [ "$(AR)" = "1" ] || [ "$(AR)" = "AR" ]; then \
			echo "Running in Autoregressive LSTM Mode"; \
			$(PYTHON) $(RETRAIN_SCRIPT) --basin $(basin) --ar $(if $(only_hwls),--only-hwls $(only_hwls),); \
		else \
			echo "Running in Normal LSTM Mode"; \
			$(PYTHON) $(RETRAIN_SCRIPT) --basin $(basin) $(if $(only_hwls),--only-hwls $(only_hwls),); \
		fi; \
	else \
		echo "Environment file $(ENV_FILE) not found. Please log in first."; \
		exit 1; \
	fi
	@echo "=== Retraining Run Completed ==="


forecast:
	@echo "=== Starting New Forecast run ==="
	@echo "=== Authenticating User ==="
	$(PYTHON) login.py
	@echo "=== Running Forecast Script ==="
	@if [ -f $(ENV_FILE) ]; then \
		set -a; \
		. $(ENV_FILE); \
		if [ "$(AR)" = "true" ] || [ "$(AR)" = "yes" ] || [ "$(AR)" = "1" ] || [ "$(AR)" = "AR" ]; then \
			echo "Running in Autoregressive mode"; \
			if [ -n "$(basin)" ]; then \
				echo "Using basin: $(basin)"; \
				$(PYTHON) $(FORECAST_SCRIPT) --ar --basin $(basin); \
			else \
				$(PYTHON) $(FORECAST_SCRIPT) --ar; \
			fi; \
		else \
			echo "Running in Standard mode"; \
			if [ -n "$(basin)" ]; then \
				echo "Using basin: $(basin)"; \
				$(PYTHON) $(FORECAST_SCRIPT) --basin $(basin); \
			else \
				$(PYTHON) $(FORECAST_SCRIPT); \
			fi; \
		fi; \
	else \
		echo "Environment file $(ENV_FILE) not found. Please log in first."; \
		exit 1; \
	fi
	@echo "=== Forecast run completed ==="

# Help
help:
	@echo "Available commands:"
	@echo "  make start-floodai    - Start MLflow server"
	@echo "  make stop-floodai     - Stop MLflow server"
	@echo "  make logs            - Show real-time logs"
	@echo "  make retrain basin=1 - Run retraining with basin 1 (Normal mode)"
	@echo "  make retrain basin=1 AR=1 - Run retraining with basin 1 (Autoregressive mode)"
	@echo "  make retrain only_hwls=hwl_id1,hwl_id2 - Run retraining with specific hwls"
	@echo "  make retrain basin=2 - Run retraining with basin 2"
	@echo "  make forecast        - Run standard forecast"
	@echo "  make forecast basin=1 - Run forecast with basin 1"
	@echo "  make forecast basin=2 - Run forecast with basin 2"
	@echo "  make help            - Show this help message"

.PHONY: start-floodai stop-floodai logs retrain forecast help