"""MLflow authentication module.

This module handles user authentication for MLflow, including login and signup functionality.
It provides a secure way to manage user credentials and store them in an environment file.
"""

# Standard library imports
import os
from getpass import getpass
from typing import Tuple

# Third-party imports
from mlflow.server.auth.client import AuthServiceClient
from mlflow.exceptions import RestException


# Constants
MLFLOW_TRACKING_URI = "http://localhost:5000"
ENV_FILE = ".mlflow_env"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "password"


def save_env_file(username: str, password: str) -> None:
    """Save authentication credentials to environment file.
    
    Args:
        username: MLflow username
        password: MLflow password
    """
    with open(ENV_FILE, "w") as f:
        f.write(f"MLFLOW_TRACKING_USERNAME={username}\n")
        f.write(f"MLFLOW_TRACKING_PASSWORD={password}\n")
        f.write(f"MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}\n")
    print(f"Environment variables saved to {ENV_FILE}")


def login() -> Tuple[str, str]:
    """Authenticate user with MLflow server.
    
    Returns:
        Tuple[str, str]: Username and password if authentication successful
        
    Raises:
        SystemExit: If login is aborted by user
    """
    print("Welcome to MLflow!")
    while True:
        username = input("Enter your username: ")
        password = getpass("Enter your password: ")

        try:
            client = AuthServiceClient(MLFLOW_TRACKING_URI)
            print("Authenticating...")
            user = client.get_user(username)
            print(f"[✅] Login successful: Welcome, {user.username}.")
            return username, password
        except RestException as e:
            print(f"[❌] Login failed: {e}")
            signup_prompt = input("Would you like to sign up? (y/n): ").strip().lower()
            if signup_prompt in ("y", "ye", "yes"):
                return signup()
            else:
                print("Login process aborted.")
                exit(1)


def signup() -> Tuple[str, str]:
    """Register new user with MLflow server and grant admin privileges.
    
    Returns:
        Tuple[str, str]: Username and password if registration successful
        
    Raises:
        SystemExit: If registration fails
    """
    print("[📝] Sign up")
    username = input("Enter your desired username: ")

    while True:
        password = getpass("Enter your desired password: ")
        confirm_password = getpass("Confirm your password: ")
        if password != confirm_password:
            print("Passwords do not match. Please try again.")
        else:
            break

    try:
        client = AuthServiceClient(MLFLOW_TRACKING_URI)
        user = client.create_user(username=username, password=password)
        print(f"[✅] Registration successful: {user.username}")
        client.update_user_admin(username, True)
        print(f"Updated user: {user.username} as an admin")
        return username, password

    except RestException as e:
        print(f"[❌] Registration failed: {e}")
        exit(1)


def main() -> None:
    """Main entry point for MLflow authentication.
    
    Sets up default admin credentials and initiates the login process.
    """
    # Set default admin credentials
    os.environ["MLFLOW_TRACKING_USERNAME"] = DEFAULT_ADMIN_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DEFAULT_ADMIN_PASSWORD
    
    # Start login process
    username, password = login()
    save_env_file(username, password)


if __name__ == "__main__":
    main()
