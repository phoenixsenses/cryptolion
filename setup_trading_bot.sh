#!/bin/bash

# setup_trading_bot.sh
# This script sets up the trading bot environment by:
# 1. Deactivating any active environments.
# 2. Disabling Conda's base environment auto-activation.
# 3. Creating and activating the project's virtual environment (.venv).
# 4. Installing required dependencies.
# 5. Cleaning up shell configuration files to remove redundant activation commands.

# -------------------------------
# Function Definitions
# -------------------------------

# Function to deactivate all active environments
deactivate_all_envs() {
    echo "Deactivating all active environments..."

    # Deactivate Conda environment if active
    if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
        echo "Deactivating Conda environment: $CONDA_DEFAULT_ENV"
        conda deactivate
    fi

    # Deactivate virtualenv if active
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Deactivating virtual environment: $VIRTUAL_ENV"
        deactivate
    fi

    echo "All environments deactivated."
}

# Function to prevent Conda's base environment from auto-activating
prevent_conda_auto_activate() {
    echo "Configuring Conda to prevent auto-activation of the base environment..."
    conda config --set auto_activate_base false
    echo "Conda auto-activation disabled."
}

# Function to create the virtual environment
create_virtualenv() {
    PROJECT_VENV=".venv"

    if [[ -d "$PROJECT_VENV" ]]; then
        echo "Virtual environment '$PROJECT_VENV' already exists."
    else
        echo "Creating virtual environment '$PROJECT_VENV'..."
        python3 -m venv "$PROJECT_VENV"
        echo "Virtual environment '$PROJECT_VENV' created."
    fi
}

# Function to activate the project's virtual environment
activate_virtualenv() {
    PROJECT_VENV=".venv"

    if [[ -d "$PROJECT_VENV" ]]; then
        echo "Activating virtual environment: $PROJECT_VENV"
        source "$PROJECT_VENV/bin/activate"
        echo "Virtual environment '$PROJECT_VENV' activated."
    else
        echo "Error: Virtual environment '$PROJECT_VENV' not found."
        echo "Please create it using 'python3 -m venv .venv' before proceeding."
        exit 1
    fi
}

# Function to install dependencies from requirements.txt
install_dependencies() {
    REQUIREMENTS_FILE="requirements.txt"

    if [[ -f "$REQUIREMENTS_FILE" ]]; then
        echo "Installing dependencies from '$REQUIREMENTS_FILE'..."
        pip install --upgrade pip
        pip install -r "$REQUIREMENTS_FILE"
        echo "Dependencies installed successfully."
    else
        echo "Error: '$REQUIREMENTS_FILE' not found in the project directory."
        echo "Please ensure it exists and lists all required packages."
        exit 1
    fi
}

# Function to clean up shell configuration files
clean_shell_configs() {
    echo "Cleaning up shell configuration files to remove redundant activation commands..."

    # Define shell config files to clean
    SHELL_CONFIG_FILES=(".bashrc" ".bash_profile" ".zshrc")

    for CONFIG_FILE in "${SHELL_CONFIG_FILES[@]}"; do
        if [[ -f "$HOME/$CONFIG_FILE" ]]; then
            echo "Processing '$CONFIG_FILE'..."

            # Remove lines that activate the virtual environment
            sed -i.bak '/source .*\.venv\/bin\/activate/d' "$HOME/$CONFIG_FILE"

            # Remove lines that activate Conda environments
            sed -i '' '/conda activate/d' "$HOME/$CONFIG_FILE"

            echo "Cleaned '$CONFIG_FILE'. Backup saved as '$CONFIG_FILE.bak'."
        else
            echo "Shell configuration file '$CONFIG_FILE' does not exist. Skipping."
        fi
    done

    echo "Shell configuration files cleaned."
}

# Function to reload shell configuration
reload_shell_config() {
    echo "Reloading shell configuration..."

    # Detect which shell is being used
    current_shell=$(basename "$SHELL")

    if [[ "$current_shell" == "bash" ]]; then
        source "$HOME/.bashrc" || source "$HOME/.bash_profile"
    elif [[ "$current_shell" == "zsh" ]]; then
        source "$HOME/.zshrc"
    else
        echo "Unsupported shell: $current_shell. Please reload your shell manually."
    fi

    echo "Shell configuration reloaded."
}

# -------------------------------
# Main Execution Flow
# -------------------------------

echo "Starting setup_trading_bot.sh script..."

# Step 1: Deactivate any active environments
deactivate_all_envs

# Step 2: Prevent Conda's base environment from auto-activating
prevent_conda_auto_activate

# Step 3: Create the virtual environment
create_virtualenv

# Step 4: Activate the virtual environment
activate_virtualenv

# Step 5: Install required dependencies
install_dependencies

# Step 6: Clean up shell configuration files
clean_shell_configs

# Step 7: Reload shell configuration
reload_shell_config

echo "Setup completed successfully."
echo "Your shell prompt should now display only one virtual environment indicator (e.g., '(.venv)')."

echo "You can now run your trading bot using the 'run_trading_bot.sh' script."
