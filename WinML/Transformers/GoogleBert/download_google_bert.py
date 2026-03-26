import torch
import os
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

def export_model_to_onnx():
    """
    Download Google BERT bert-base-multilingual-cased model and export to ONNX format
    """
    # Define paths
    model_dir = Path("./model")
    model_dir.mkdir(exist_ok=True)
    onnx_model_path = model_dir / "model.onnx"
    
    # Model name
    model_name = 'google-bert/bert-base-multilingual-cased'
    batch_size = 1
    sequence_length = 128
    print(f"Downloading model: {model_name}")
    
    # Load the pre-trained model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Model downloaded successfully")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy inputs for export
    dummy_input = "This is a sample sentence for exporting the model."
    encoded_input = tokenizer(
        dummy_input,
        padding="max_length",
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    
    # Extract input tensors
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    token_type_ids = encoded_input["token_type_ids"]
    
    print(f"Exporting model to ONNX format...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        str(onnx_model_path),
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state', 'pooler_output'],
        opset_version=14,
        do_constant_folding=True,
    )
    
    print(f"Model successfully exported to: {onnx_model_path}")
    print(f"Model file size: {onnx_model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: PASSED")
    except ImportError:
        print("Note: Install 'onnx' package to validate the exported model")
    except Exception as e:
        print(f"Warning: Model validation failed: {e}")
    
    return str(onnx_model_path)

if __name__ == "__main__":
    try:
        model_path = export_model_to_onnx()
        print(f"\nDownload and export complete!")
        print(f"You can now use the model with: python run_google_bert.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

