#!/bin/bash

# ComfyUI/main.py

# Forgive me father, for I have given you this GPT4-Turbo written bash file!

# Change the directory to where your Python script resides, if needed
cd "$(dirname "$0")"

# Define the path to the embedded Python executable
# PYTHON_EMBEDDED_PATH="$(dirname "$0")/python_embedded/python"

# No need to append system directories to PATH, as they are already there in Linux

# Create virtual environment with the embedded Python if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment for Augmentoolkit Node Edition..."
    python3.10 -m venv venv #--always-copy # Make sure the activation file is created.
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Exiting."
        exit 1
    fi
    echo "Virtual environment 'venv' created."
fi

# Activate virtual environment
if source venv/bin/activate; then
    echo "Virtual environment 'venv' activated."
else
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

BATCH_FILE_NAME=run_augmentoolkit_linux.bash

read -p "Welcome to Augmentoolkit Node Edition: Linux Version! Look at the program's arguments before proceeding? y/n: " help

# Check if the help argument is provided. TODO: Actually write the help page.
if [ "$help" = "y" ]; then
    echo "Sorry, but Augmentoolkit Node Edition's help page is still under construction. Sorry about that!"
fi


# Install torch and all its dependencies.
echo "***********************************"
echo "Installing torch version 2.2.0, torchvision, and torchaudio..."
pip show torch > /dev/null 2>&1
if [ $? -ne 0 ]; then
    pip install torch==2.2.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then
        echo "Failed to install torch==2.2.0, torchvision, torchaudio. Exiting."
        exit 1
    fi
else
    echo "torch==2.2.0, torchvision, torchaudio are already installed."
fi


echo "***********************************"
echo "Installing ComfyUI dependencies and misc. dependencies"
# Install the rest of ComfyUI dependencies
pip install -r requirements.txt

# Install aphrodite-engine
echo "***********************************"
echo "Installing aphrodite-engine..."
pip show aphrodite-engine > /dev/null 2>&1
if [ $? -ne 0 ]; then
    pip install aphrodite-engine
    if [ $? -ne 0 ]; then
        echo "Failed to install aphrodite-engine. Exiting."
        exit 1
    fi
else
    echo "aphrodite-engine is already installed."
fi


# Install llama-cpp-python
echo "***********************************"
echo "Installing llama-cpp-python..."
pip show llama-cpp-python > /dev/null 2>&1
if [ $? -ne 0 ]; then
    # llama_cpp_python_setup
    read -p "Would you like to build llama-cpp-python with GPU off-loading support? Only CUBLAS is supported at this time. y/n: " CUDA_YES_NO

    if [ "$CUDA_YES_NO" = "n" ]; then
        echo "Building llama-cpp-python without GPU off-loading..."
        pip install llama-cpp-python
        if [ $? -ne 0 ]; then
            echo "Failed to install llama-cpp-python. Exiting."
            exit 1
        fi
    else
        # Install llama-cpp-python with GPU support
        CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
        if [ $? -ne 0 ]; then
            echo "Failed to install llama-cpp-python with CUBLAS. Exiting."
            exit 1
        fi
    fi
else
    echo "llama-cpp-python is already installed."
fi

pip install torchsde

echo "***********************************"
echo "All dependencies installed successfully."

# Run the Python script
echo "***********************************"
echo "Running ComfyUI main.py script..."

read -p "Please select your run-mode choice, CPU or Nvidia: " CPU_OR_NVIDIA

if [ "$CPU_OR_NVIDIA" = "CPU" ]; then
    python3.10 -s ComfyUI/main.py --cpu
else
    python3.10 -s ComfyUI/main.py
fi

echo "Script execution completed."
