<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Windows ML Examples </h1>
    </td>
 </tr>
</table>

# Introduction

Windows Machine Learning (WinML) enables developers to run ONNX AI models on PC via ONNX runtime, with automatic execution provider management for different hardwares i.e. CPUs, GPUs and NPUs. For more details [Microsoft Windows ML Documentation](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)

WinML is a lightweight, efficient AI model runtime designed for dynamic execution and broad hardware compatibility.

Key Features:
-  Dynamic Loading - Automatically fetches latest execution providers (EPs) at runtime
-  Shared ONNX Runtime - Reduces application size by eliminating redundant dependencies
-  Optimized Distribution - Smaller downloads and streamlined installations
-  Broader Hardware Support - Seamless compatibility across different vendors and device types

In this document, we discuss how to enable AMD hardware through WinML APIs. This tutorial uses Windows ML APIs to run CNN, Transformer and LLM models on AMD NPU.

## System Requirements

- Windows 10/11 with NPU support
- Visual Studio Code with AI Toolkit extension (for AI model conversion)
- Install compatible version of Windows App SDK
- Visual Studio 2022 (for building the C++ application)
- Python package management (Miniforge) and Python 3.10+

## Model Support

The VitisAI EP within WinML supports input models in the following formats:

  - CNN Models
    - Original float (FP32) model with automatically converted to BF16 during compilation
    - Quantized QDQ model using A8W8 configuration
  - Transformer Models
    - Original float (FP32) model with automatically converted to BF16 during compilation
    - Quantized QDQ model using A16W8 configuration
  - LLM Models:
    - Quantized and pre-compiled LLM models

## WinML Installation Instructions

Install the required python packages in the conda environment `winml_env`

```sh
conda create -n winml_env python==3.11
conda activate winml_env
pip install --pre --upgrade -r .\requirements.txt
```

Check installed wasdk python version and install same version of [Windows App SDK](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads):

```sh
conda list | findstr wasdk
```

Expected Output:

```shell
wasdk-microsoft-windows-ai-machinelearning 2.0.0.dev4               pypi_0    pypi
wasdk-microsoft-windows-applicationmodel-dynamicdependency-bootstrap 2.0.0.dev4               pypi_0    pypi
```

Download the Windows App SDK corresponding to the wasdk version (e.g., 2.0.0.dev4) or latest and install it to ensure the WinML execution providers work correctly.

```shell
curl -L -o windowsappruntimeinstall-x86.exe "https://aka.ms/windowsappsdk/2.0/2.0.0-experimental4/windowsappruntimeinstall-x86.exe"
windowsappruntimeinstall-x86.exe --quiet
```

After installation instructions run the check_winml_setup.py to verify the Windows ML installation

```sh
============================================================
WinML Setup Checker
============================================================
Python: 3.11.0 (<path_to_python_installation>\python.exe)
WASDK Python Packages:
----------------------------------------
  [✓] wasdk-ML: 2.0.0.dev4
  [✓] wasdk-Bootstrap: 2.0.0.dev4
Windows App SDK Runtime:
----------------------------------------
  [✓] Windows App SDK: 2.0-experimental5 (internal: 0.770.2319.0)
Installed runtimes (newest first):
    - 2.0-experimental5 (internal: 0.770.2319.0)
    * 2.0-experimental4 (internal: 0.738.2207.0)
    - 1.8 (internal: 8000.642.119.0)
    - 1.8 (internal: 8000.675.1142.0)
    - 1.8 (internal: 8000.731.1532.0)
    - 1.8 (internal: 8000.770.947.0)
    - 1.8-experimental (internal: 8000.589.1529.0)
    - 1.8-preview (internal: 8000.591.1127.0)
    * Active runtime used by this checker
Expected SDK: 2.0.0-experimental4
============================================================
Status: All components installed. Please, ensure matching Windows App SDK version is Installed.
```

## WinML examples

For detailed step by step tutorials:

- [Getting Started ResNet Tutorial](./CNN/ResNet/README.md)
- [Transformer Tutorial using Google BERT](./Transformers/GoogleBert/README.md)
- [Transformer Tutorial using OpenAI CLIP model](./Transformers/clip-vit-base-patch16/README.md)
- [LLM Examples](./LLM/README.md)

## References

- [Windows ML Documentation](https://learn.microsoft.com/en-us/windows/ai/windows-ml/)
- [AI Toolkit Documentation](https://code.visualstudio.com/docs/intelligentapps/modelconversion)
- [Windows App SDK](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/)
