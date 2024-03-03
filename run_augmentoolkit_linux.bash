#!/bin/bash

# ComfyUI/main.py

# Forgive me father, for I have given you this GPT4-Turbo written bash file!

# Change the directory to where your Python script resides, if needed
cd "$(dirname "$0")"

# Define the path to the embedded Python executable
PYTHON_EMBEDDED_PATH="$(dirname "$0")/python_embedded/python"

# No need to append system directories to PATH, as they are already there in Linux

# Create virtual environment with the embedded Python if it doesn't exist
if [ ! -d "myenv" ]; then
    echo "Creating virtual environment for Augmentoolkit Node Edition..."
    "$PYTHON_EMBEDDED_PATH" -m venv myenv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Exiting."
        exit 1
    fi
    echo "Virtual environment 'myenv' created and activated."
fi

# Activate virtual environment
source myenv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

BATCH_FILE_NAME=run_augmentoolkit.sh

read -p "Welcome to Augmentoolkit Node Edition: Linux Version! Look at the program's arguments before proceeding? y/n: " help

# Check if the help argument is provided. TODO: Actually write the help page.
if [ "$help" = "y" ]; then
    echo "Sorry, but Augmentoolkit Node Edition's help page is still under construction. Sorry about that!"
fi

# Install all the dependencies. These are installed manually because python dependencies are picky!
echo "Installing llama-cpp-python dependencies..."
for i in protobuf scikit-build scikit_build_core pyproject-metadata pathspec cmake; do
    pip show "$i" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Installing $i..."
        pip install "$i"
        if [ $? -ne 0 ]; then
            echo "Failed to install $i. Exiting."
            exit 1
        fi
    else
        echo "$i is already installed."
    fi
done

echo "llama-cpp-python dependencies installed."
echo "***********************************"
echo "Searching for existing llama-cpp-python installation..."

# Check whether a copy of llama-cpp-python is already installed.
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
        read -p "Have you installed C++ dependencies and cmake? y/n: " INSTALLED_DEPS

        if [ "$INSTALLED_DEPS" = "n" ]; then
            echo "CUBLAS requires C++ and cmake dependencies."
            exit
        fi

        read -p "Enter the path to your CUDA installation. Default is /usr/local/cuda/bin/nvcc): " CUDACXX

        # Setup default CUDA path.
        if [ -z "$CUDACXX" ]; then
            CUDACXX="/usr/local/cuda/bin/nvcc"
        fi

        # Verify CUDA path existence
        if [ ! -f "$CUDACXX" ]; then
            echo "The provided CUDA path does not exist."
            echo "If you do not have CUDA installed, download the latest version of it."
            exit 1
        fi

        # Install llama-cpp-python with GPU support
        CUDACXX="$CUDACXX"
        CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major"
        FORCE_CMAKE=1
        pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
        if [ $? -ne 0 ]; then
            echo "Failed to install llama-cpp-python with CUBLAS. Exiting."
            exit 1
        fi
    fi
else
    echo "llama-cpp-python is already installed."
fi

# requirements_setup
echo "***********************************"
echo "Installing Torch and additional dependencies..."
while read -r i; do
    pip show "$i" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Installing $i..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Failed to install $i. Exiting."
            exit 1
        fi
        break
    else
        echo "$i is already installed."
    fi
done < requirements.txt

echo "***********************************"
echo "All dependencies installed successfully."

# Run the Python script
echo "***********************************"
echo "Running ComfyUI main.py script..."

read -p "Please select your run-mode choice, CPU or Nvidia: " CPU_OR_NVIDIA

if [ "$CPU_OR_NVIDIA" = "CPU" ]; then
    "$PYTHON_EMBEDDED_PATH" -s ComfyUI/main.py --cpu
else
    "$PYTHON_EMBEDDED_PATH" -s ComfyUI/main.py
fi

echo "Script execution completed."
