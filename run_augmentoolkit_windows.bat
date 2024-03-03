@echo off
REM ComfyUI\main.py

REM Change the directory to where your Python script resides, if needed
cd /D "%~dp0"

REM Define the path to the embedded Python executable
REM Comfy misspelled embedded as embeded. UGUHGUHSUHUGHUH
set PYTHON_EMBEDDED_PATH="%~dp0python_embeded\python.exe"

REM Append the Windows system32 directory to the PATH variable
set PATH=%PATH%;%SystemRoot%\system32

REM Create virtual environment with the embedded Python if it doesn't exist
if not exist "myenv" (
    echo Creating virtual environment for Augmentoolkit Node Edition...
    %PYTHON_EMBEDDED_PATH% -m venv myenv
    if errorlevel 1 (
        echo Failed to create virtual environment. Exiting.
        pause
        exit /b
    )
    echo Virtual environment 'myenv' created and activated.
)

REM Activate virtual environment
call myenv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment. Exiting.
    pause
    exit /b
)

set BATCH_FILE_NAME=run_augmentoolkit.bat

set /p help="Welcome to Augmentoolkit Node Edition: Windows Version! Look at the program's arguments before proceeding? y/n: "

REM Check if the help argument is provided. TODO: Actually write the help page.
if "%help%"=="y" (
    echo Sorry, but Augmentoolkit Node Edition's help page is still under construction. Sorry about that!
)

REM Install all the dependencies. These are installed manually because python dependencies are FUCKING picky!
echo ***********************************
echo Installing llama-cpp-python dependencies...
for %%i in (protobuf scikit-build scikit_build_core pyproject-metadata pathspec cmake) do (
    %PYTHON_EMBEDDED_PATH% -m pip show %%i >nul 2>&1
    if errorlevel 1 (
        echo Installing %%i...
        %PYTHON_EMBEDDED_PATH% -m pip install %%i
        if errorlevel 1 (
            echo Failed to install %%i. Exiting.
            pause
            exit /b
        )
    ) else (
        echo %%i is already installed.
    )
)


echo llama-cpp-python dependencies installed.
echo ***********************************
echo Searching for existing llama-cpp-python installation...

REM Check whether a copy of llama-cpp-python is already installed.
for %%i in (llama-cpp-python) do (
    %PYTHON_EMBEDDED_PATH% -m pip show %%i >nul 2>&1
    if errorlevel 1 (
        goto :llama_cpp_python_setup
    ) else (
        echo %%i is already installed.
    )
)

set /p FORCE_REINSTALL="Force reinstall of llama-cpp-python y/n: "
if "%FORCE_REINSTALL%"=="y" (
    goto :llama_cpp_python_setup
) else (
    echo Existing llama-cpp-python installation selected.
    goto :requirements_setup
)

:llama_cpp_python_setup
REM Check to see if the user wants to build with GPU support.
set /p CUDA_YES_NO="Would you like to build llama-cpp-python with GPU off-loading support? Only CUBLAS is supported at this time. y/n: "

if "%CUDA_YES_NO%"=="n" (
    echo Building llama-cpp-python without GPU off-loading...
    %PYTHON_EMBEDDED_PATH% -m pip install llama-cpp-python
    if errorlevel 1 (
        echo Failed to install llama-cpp-python. Exiting.
        pause
        exit /b
    )
    goto :requirements_setup
)

set /p INSTALLED_MSVS="Have you installed Microsoft Visual Studio with C++ and cmake dependencies? y/n: "

if "%INSTALLED_MSVS%"=="n" (
    echo CUBLAS requires a copy of Microsoft Visual Studio with C++ and cmake dependencies in order to build.
    echo If you do not have Microsoft Visual Studio, go to https://visualstudio.microsoft.com/vs/features/cplusplus/ and download it.
    pause
    exit
)

set /p CUDACXX="If CUDA is installed, enter the path to your CUDA installation. Default is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe:  "

REM Setup default CUDA path.
if "%CUDACXX%"=="" (
    set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe"
) else (
    set "CUDACXX=%CUDACXX%"
)

REM Verify CUDA path existence
if not exist "%CUDACXX%" (
    echo The provided CUDA path does not exist.
    echo If you do not have CUDA installed, go to https://developer.nvidia.com/cuda-downloads and download the latest version of it.
    pause
    exit /b
)

REM Install llama-cpp-python with GPU support
set CUDACXX=%CUDACXX%
set CMAKE_ARGS=-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major
set FORCE_CMAKE=1
%PYTHON_EMBEDDED_PATH% -m pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
if errorlevel 1 (
    echo Failed to install llama-cpp-python with CUBLAS. Exiting.
    pause
    exit /b
)
echo llama-cpp-python building successful.

:requirements_setup
echo *********************************** 
echo Installing Torch and additional dependencies...
for /f %%i in (requirements.txt) do (
    %PYTHON_EMBEDDED_PATH% -m pip show %%i >nul 2>&1
    if errorlevel 1 (
        echo Installing %%i...
        %PYTHON_EMBEDDED_PATH% -m pip install -r requirements.txt
        if errorlevel 1 (
            echo Failed to install %%i. Exiting.
            pause
            exit /b
        )
        goto :end_requirements
    ) else (
        echo %%i is already installed.
    )
)

:end_requirements
echo ***********************************
echo All dependencies installed successfully.

REM Run the Python script
echo ***********************************
echo Running ComfyUI main.py script...

set /p CPU_OR_NVIDIA="Please select your run-mode choice, CPU or Nvidia: "

if "%CPU_OR_NVIDIA%"=="CPU" (
    %PYTHON_EMBEDDED_PATH% -s ComfyUI\main.py --cpu --windows-standalone-build
) else (
    %PYTHON_EMBEDDED_PATH% -s ComfyUI\main.py --windows-standalone-build
)

echo Script execution completed.
pause
