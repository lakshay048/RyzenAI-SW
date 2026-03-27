# WinML ResNet Tutorial: C++

This tutorial demonstrates how to use Windows Machine Learning (WinML) for ONNX model inference using C++. It covers setup, running models, and sample code for image classification using ResNet.


## Build the C++ Project

Ensure the requirements and model setup instructions mentioned in the [CNN README](../README.md#requirements) are completed before proceeding.

Install NuGet packages and build the executable using Visual Studio or the Visual Studio Developer Command Prompt:


```sh
cd cpp\CppResnetBuildDemo\
nuget.exe restore .\CppResnetBuildDemo.sln
msbuild .\CppResnetBuildDemo.sln /p:Configuration=Release /m
```

### Run Inference

Run inference on NPU (Neural Processing Unit):

```sh
.\x64\Release\CppResnetBuildDemo.exe --model ..\..\model\resnet50.onnx --image_path ..\..\images\dog.jpg --ep_policy NPU
```

Or simply run with defaults (uses NPU policy, resnet50.onnx model, and dog.jpg image):

```sh
.\x64\Release\CppResnetBuildDemo.exe
```

if using the quantized model with AI Toolkit, make sure to give the converted model path.

### Command-Line Arguments

- `--ep_policy <NPU|CPU|GPU|DEFAULT>`: Execution provider policy. Default: NPU
- `--model <path>`: Path to input ONNX model (default: ../model/resnet50.onnx relative to executable)
- `--compiled_output <path>`: Path for compiled output model (default: ../model/resnet50_ctx.onnx)
- `--image_path <path>`: Path to input image (default: dog.jpg in executable directory)
- `--help`: Show help message


## API Walkthrough

### Execution Provider Setup

The Windows ML runtime dynamically discovers and registers available execution providers (EPs):

```cpp
#include <winml/onnxruntime_cxx_api.h>
#include <winrt/Microsoft.Windows.AI.MachineLearning.h>

// Create ONNX Runtime environment
auto env = Ort::Env();

// Use WinML to download and register Execution Providers
auto catalog = winrt::Microsoft::Windows::AI::MachineLearning::ExecutionProviderCatalog::GetDefault();
auto providers = catalog.FindAllProviders();
for (const auto& provider : providers)
{
    provider.EnsureReadyAsync().get();
    provider.TryRegister();
}
```

### Session Configuration

Configure session options to enable automatic EP selection based on device policy:

```cpp
// Set the auto EP selection policy
Ort::SessionOptions sessionOptions;
sessionOptions.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER);
```

**EP Selection Policies:**
- `OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER` - Optimize for power efficiency
- `OrtExecutionProviderDevicePolicy_MAX_PERF` - Optimize for maximum performance


### Model Compilation

Models may need to be compiled for specific EPs. This is a one-time process that stores the compiled model for subsequent runs:

```cpp
bool isCompiledModelAvailable = std::filesystem::exists(compiledModelPath);

if (!isCompiledModelAvailable)
{
    Ort::ModelCompilationOptions compile_options(env, sessionOptions);
    compile_options.SetInputModelPath(modelPath.c_str());
    compile_options.SetOutputModelPath(compiledModelPath.c_str());

    std::cout << "Starting compile, this may take a few moments..." << std::endl;
    Ort::Status compileStatus = Ort::CompileModel(env, compile_options);
    if (compileStatus.IsOK())
    {
        std::cout << "Model compiled successfully!" << std::endl;
        isCompiledModelAvailable = true;
    }
}


### Session Creation

Create an inference session with the model:

```cpp
Ort::Session session(env, modelPathToUse.c_str(), sessionOptions);
std::cout << "ResNet model loaded" << std::endl;
```

## Expected Output
Run the example:
```sh
.\x64\Release\CppResnetBuildDemo.exe
```
output:
```console
ONNX Version string: 1.23.0
Getting available providers...
Provider: MIGraphXExecutionProvider
Provider: VitisAIExecutionProvider
ONNX providers registered:
CPUExecutionProvider
DmlExecutionProvider
MIGraphXExecutionProvider
VitisAIExecutionProvider
image path: "D:\\repos\\RyzenAI-SW\\tutorial\\WinML\\cpp\\CppResnetBuildDemo\\x64\\Release\\dog.jpg"
Using compiled model: "C:\\Users\\Administrator\\Documents\\resnet50\\model-float\\model-compiled.onnx"
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20251009 12:35:33.712430 27416 vitisai_compile_model.cpp:1263] Vitis AI EP Load ONNX Model Success
I20251009 12:35:33.712430 27416 vitisai_compile_model.cpp:1264] Graph Input Node Name/Shape (1)
I20251009 12:35:33.712430 27416 vitisai_compile_model.cpp:1268]          data : [-1x3x224x224]
I20251009 12:35:33.712430 27416 vitisai_compile_model.cpp:1274] Graph Output Node Name/Shape (1)
I20251009 12:35:33.712430 27416 vitisai_compile_model.cpp:1278]          resnetv17_dense0_fwd : [-1x1000]
[Vitis AI EP] No. of Subgraphs supported by Vitis AI EP: VAIML     1
ResNet model loaded
Running inference for 20 iterations
....................
Output for the last iteration
Top Predictions:
-------------------------------------------
Label                           Confidence
-------------------------------------------
207,golden retriever                 97.78%
208,Labrador retriever                1.48%
852,tennis ball                       0.09%
220,Sussex spaniel                    0.08%
219,cocker spaniel                    0.07%
-------------------------------------------
---------------------------------------------
Time taken for 20 iterations: 0 seconds
Avg time per iteration : 19 milliseconds
```


## References

- [AI Toolkit Documentation](https://code.visualstudio.com/docs/intelligentapps/modelconversion)
- [WinML Documentation](https://learn.microsoft.com/en-us/windows/ai/windows-ml/)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [Windows::AI::MachineLearning Namespace](https://learn.microsoft.com/en-us/uwp/api/windows.ai.machinelearning)
- [ResNet-50 Model on HuggingFace](https://huggingface.co/microsoft/resnet-50)
