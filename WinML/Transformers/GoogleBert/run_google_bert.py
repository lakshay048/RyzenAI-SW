import subprocess
import json
import sys
import os
import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import argparse

def register_execution_providers():
    worker_script = os.path.abspath('winml.py')
    result = subprocess.check_output([sys.executable, worker_script], text=True)
    paths = json.loads(result)
    print("paths:", paths)
    for name, lib_path in paths.items():
        if not lib_path or not os.path.exists(lib_path):
            print(f"Skipping execution provider {name}: invalid or missing library path")
            continue
        ort.register_execution_provider_library(name, lib_path)
        print(f"Registered execution provider: {name} with library path: {lib_path}")

def mean_pooling(model_output, attention_mask):
    token_embeddings = torch.tensor(model_output[0])
    input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Google BERT inference with NPU or CPU')
    parser.add_argument('--ep_policy', type=str.upper, default='NPU', choices=['NPU', 'CPU', 'GPU', 'DEFAULT'],
                        help='Set execution provider policy (NPU, CPU, GPU, DEFAULT). Default: NPU')
    parser.add_argument('--model', type=str, default='./model/model.onnx',
                        help='Path to ONNX model (default: ./model_a16w8/model.onnx)')
    args = parser.parse_args()

    onnx_model_path = args.model
    
    # Validate model path exists
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"Model file not found: {onnx_model_path}")
    
    print("Registering execution providers ...")
    register_execution_providers()

    print(f"Model path: {onnx_model_path}")

    inputs = "This is an example sentence."
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
    encoded_input = tokenizer(
        inputs,
        padding="max_length",
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    
    print("Creating session ...")
    session_options = ort.SessionOptions()

    # Set execution provider policy
    policy_map = {
        'NPU': ort.OrtExecutionProviderDevicePolicy.PREFER_NPU,
        'CPU': ort.OrtExecutionProviderDevicePolicy.PREFER_CPU,
        'DEFAULT': ort.OrtExecutionProviderDevicePolicy.DEFAULT
    }
    
    policy = policy_map.get(args.ep_policy)
    if policy:
        session_options.set_provider_selection_policy(policy)
        print(f"Set provider selection policy to: {args.ep_policy}")

    session = ort.InferenceSession(
        onnx_model_path, # a model with QNN EPContext nodes
        sess_options=session_options,
    )

    # Report active execution providers
    try:
        eps = session.get_providers()
        print(f"Active execution providers (priority order): {eps}")
        if eps:
            print(f"Primary provider (highest priority): {eps[0]}")
    except Exception as e:
        print(f"Warning: unable to query active execution providers: {e}")

    print("Running inference ...")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    token_type_ids = encoded_input["token_type_ids"]
    inputs = {
        "input_ids": input_ids.long().cpu().numpy(),
        "attention_mask": attention_mask.long().cpu().numpy(),
        "token_type_ids": token_type_ids.long().cpu().numpy()
    }

    outputs = session.run(None, inputs)
    embeds_1 = mean_pooling(outputs, encoded_input['attention_mask'])
    embeds_1 = F.normalize(embeds_1, p=2, dim=1)

    # get text embedding from orinal model, as ground truth.
    model = AutoModel.from_pretrained('google-bert/bert-base-multilingual-cased').eval()
    with torch.no_grad():
        outputs = model(**encoded_input)
        embeds_2 = mean_pooling(outputs, encoded_input['attention_mask'])
        embeds_2 = F.normalize(embeds_2, p=2, dim=1)

    similarity = F.cosine_similarity(embeds_1, embeds_2).item()
    print(f"Similarity: {similarity}")

if __name__ == "__main__":
    main()