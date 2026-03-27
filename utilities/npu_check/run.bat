@REM Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
@echo off

:continue
echo Building NPU check application...

cmake -S . -B .\build_out -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX=.\build_out\install
if %ERRORLEVEL% NEQ 0 echo "cmake config failed..." & goto :error

cmake --build .\build_out --config Release
if %ERRORLEVEL% NEQ 0 echo "cmake build failed..." & goto :error

cmake --build .\build_out --config Release --target INSTALL
if %ERRORLEVEL% NEQ 0 echo "cmake install failed..." & goto :error

echo.
echo.

build_out\Release\npu_util.exe

:Error
exit /b %ERRORLEVEL%