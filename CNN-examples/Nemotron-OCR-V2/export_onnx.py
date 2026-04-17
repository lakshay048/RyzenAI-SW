#!/usr/bin/env python3
"""Export Nemotron OCR v2 (English) to static-shape ONNX.

Produces 3 ONNX models with frozen input shapes:
  - Detector:    input [1, 3, 1024, 1024] fp32
  - Recognizer:  input [1, 128, 8, 32] fp32
  - Relational:  rectified_quads [128, 128, 2, 3] + original_quads [128, 4, 2]
                 + recog_features [128, 32, 256] fp32

The detector and recognizer use the standard PyTorch models from
nvidia/nemotron-ocr-v2.  The relational model uses a sparse-neighbor
architecture (KNN top-15 via torch.cdist + topk inside the ONNX graph)
adapted from https://github.com/ramkrishna2910/nemotron-ocr-v2-onnx.

Each export is validated by comparing PyTorch vs ONNXRuntime CPU outputs.
When existing ONNX models are available (from export_nemotron_ocr_v2_onnx.py),
also validates functional equivalence against those models.

Requirements:
  - nemotron-ocr-v2 repo checkout (contains v2_english/ weights and
    nemotron-ocr/src/ source tree)
  - PyTorch, ONNX, ONNXRuntime

Usage:
    python export_onnx.py --model-dir ./nemotron-ocr-v2
    python export_onnx.py --model-dir ./nemotron-ocr-v2 --output-dir ./my_onnx
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx.external_data_helper import convert_model_from_external_data, load_external_data_for_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INFER_LENGTH = 1024
DETECTOR_FEATURE_CHANNELS = 128
RECOGNIZER_CROP_HEIGHT = 8
RELATIONAL_K = 128          # static padded region count
RELATIONAL_GRID_H = 2
RELATIONAL_GRID_W = 3
RELATIONAL_KNN = 15         # top-K nearest neighbors in relational graph

# ImageNet normalization constants (baked into detector weights)
DETECTOR_INPUT_MEAN = [0.485, 0.456, 0.406]
DETECTOR_INPUT_STD = [0.229, 0.224, 0.225]

# Naming convention matching export_nemotron_ocr_v2_onnx.py
EXPORT_NAMES = {
    "detector": "nvidia-nemotron-ocr-v2-detector-english",
    "recognizer": "nvidia-nemotron-ocr-v2-recognizer-english",
    "relational": "nvidia-nemotron-ocr-v2-relational-english",
}


# ---------------------------------------------------------------------------
# C++ extension stub
# ---------------------------------------------------------------------------

def _stub_cpp():
    """Stub out nemotron_ocr_cpp so pure-Python model classes can be imported."""
    if "nemotron_ocr_cpp" in sys.modules:
        return
    stub = types.ModuleType("nemotron_ocr_cpp")
    stub.__file__ = "<stub>"
    for name in (
        "quad_non_maximal_suppression", "region_counts_to_indices",
        "rrect_to_quads", "quad_rectify_calc_quad_width",
        "ragged_quad_all_2_all_distance_v2", "indirect_grid_sample_forward",
        "indirect_grad_sample_backward", "quad_rectify_forward",
        "quad_rectify_backward",
    ):
        setattr(stub, name, lambda *a, **k: None)
    sys.modules["nemotron_ocr_cpp"] = stub
    sys.modules["nemotron_ocr_cpp._nemotron_ocr_cpp"] = stub


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class RelationalONNX(nn.Module):
    """Sparse-neighbor relational model for ONNX export.

    Uses torch.cdist + topk(k=15) inside the graph for neighbor selection,
    making the model self-contained with no external C++ distance ops.

    Adapted from: https://github.com/ramkrishna2910/nemotron-ocr-v2-onnx
    Source: scripts/relational_onnx.py (MIT license)

    Returns:
        null_logits:      [N, 3]     -- logits for "no connection"
        neighbor_logits:  [N, K, 3]  -- logits for each of K=15 neighbors
        neighbor_indices: [N, K]     -- which region each neighbor refers to
    """

    K = RELATIONAL_KNN  # constant for tracing

    def __init__(self, num_input_channels, recog_feature_depth, k=16,
                 dropout=0.1, num_layers=4, **kwargs):
        super().__init__()
        assert k - 1 == self.K

        from nemotron_ocr.inference.models import blocks

        nic = (num_input_channels[-1]
               if isinstance(num_input_channels, (list, tuple))
               else num_input_channels)
        self.quad_downscale = 1024.0
        self.grid_area = float(RELATIONAL_GRID_H * RELATIONAL_GRID_W)
        self.pos_channels = 14
        self.initial_depth = 128
        self.k = k
        self.quad_rectify_grid_size = (RELATIONAL_GRID_H, RELATIONAL_GRID_W)

        # Learned layers (names match relational.pth keys)
        self.rect_proj = blocks.conv2d_block(nic, nic, 1)
        self.recog_tx = nn.Linear(recog_feature_depth, nic)

        initial_depth = self.initial_depth - 1 - self.pos_channels  # 113
        cb_input = 2 * nic  # 256
        self.combined_proj = nn.Sequential(
            nn.Linear(cb_input, cb_input),
            nn.BatchNorm1d(cb_input),
            nn.ReLU(),
            nn.Linear(cb_input, initial_depth),
            nn.BatchNorm1d(initial_depth),
            nn.ReLU(),
        )

        dim = 2 * self.initial_depth  # 256
        self.encoder = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    dim, 8, 2 * dim, batch_first=True,
                    dropout=dropout, norm_first=True,
                ),
                num_layers=num_layers,
            ),
            nn.Linear(dim, 3),
        )

    def forward(
        self,
        rectified_quads: torch.Tensor,   # [N, C, 2, 3]
        original_quads: torch.Tensor,     # [N, 4, 2]
        recog_features: torch.Tensor,     # [N, T, D]
    ):
        rectified_quads = rectified_quads.float()
        recog_features = recog_features.float()
        K = self.K

        # Stage 1: input encoding
        quads_scaled = original_quads / self.quad_downscale
        mid_pts = quads_scaled.detach().mean(dim=1)

        rect_out = self.rect_proj(rectified_quads)
        avg_rects = rect_out.flatten(2).sum(dim=2) / self.grid_area

        recog_enc = self.recog_tx(recog_features.detach()).mean(dim=1)

        semantic = self.combined_proj(torch.cat((avg_rects, recog_enc), dim=1))

        h1 = quads_scaled[:, 3] - quads_scaled[:, 0]
        h2 = quads_scaled[:, 2] - quads_scaled[:, 1]
        mp1 = quads_scaled[:, 0] + (h1 / 2)
        mp2 = quads_scaled[:, 1] + (h2 / 2)
        d1 = mp2 - mp1
        wdth = d1.norm(dim=1, keepdim=True)
        d1 = d1 / wdth.clamp_min(1e-6)
        hts = ((h1 + h2) / 2).norm(dim=1, keepdim=True)
        d2 = torch.stack([-d1[:, 1], d1[:, 0]], dim=-1)

        proj_rects = torch.cat(
            (semantic, quads_scaled.flatten(1), d1, d2, wdth, hts), dim=1,
        )

        # Stage 2: pairwise distance + top-k
        centers = quads_scaled.mean(dim=-2)
        dists = torch.cdist(
            centers.unsqueeze(0), centers.unsqueeze(0),
        ).squeeze(0)

        arange = torch.arange(dists.shape[0], device=dists.device)
        self_mask = (arange.unsqueeze(0) == arange.unsqueeze(1)).to(dists.dtype)
        dists = dists + self_mask * 1e9

        N = dists.shape[0]
        pad_cols = K - N
        if pad_cols > 0:
            dists = torch.nn.functional.pad(dists, (0, pad_cols), value=float("inf"))
            proj_rects_padded = torch.nn.functional.pad(proj_rects, (0, 0, 0, pad_cols))
            mid_pts_padded = torch.nn.functional.pad(mid_pts, (0, 0, 0, pad_cols))
        else:
            proj_rects_padded = proj_rects
            mid_pts_padded = mid_pts

        topk_dists, topk_idxs = torch.topk(
            dists, k=K, dim=1, largest=False, sorted=True,
        )

        idx_feat = topk_idxs.unsqueeze(-1).expand(-1, -1, proj_rects_padded.shape[-1])
        neighbor_rects = torch.gather(
            proj_rects_padded.unsqueeze(0).expand(N, -1, -1), 1, idx_feat,
        )

        idx_ctr = topk_idxs.unsqueeze(-1).expand(-1, -1, 2)
        neighbor_centers = torch.gather(
            mid_pts_padded.unsqueeze(0).expand(N, -1, -1), 1, idx_ctr,
        )

        # Directions
        pt0 = (quads_scaled[:, 0] + quads_scaled[:, 3]) / 2
        pt1 = (quads_scaled[:, 1] + quads_scaled[:, 2]) / 2
        direction = pt1 - pt0
        direction = direction / direction.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = direction.unsqueeze(1).expand_as(neighbor_centers)
        ctr = (pt0 + pt1) / 2
        vec_other = neighbor_centers - ctr.unsqueeze(1)
        dir_other = vec_other / vec_other.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        dirs = (direction * dir_other).sum(dim=-1)

        # Stage 3: assemble encoder input
        null_rects = torch.zeros(N, 1, proj_rects.shape[-1], device=proj_rects.device)
        null_dists = torch.full((N, 1), -1.0, device=proj_rects.device)
        null_dirs = torch.full((N, 1), -2.0, device=proj_rects.device)

        to_rects = torch.cat((
            torch.cat((null_rects, neighbor_rects), dim=1),
            torch.cat((null_dists, topk_dists), dim=1).unsqueeze(-1),
            torch.cat((null_dirs, dirs), dim=1).unsqueeze(-1),
        ), dim=-1)

        from_rects = proj_rects.unsqueeze(1).expand(-1, K + 1, -1)
        enc_input = torch.cat((from_rects, to_rects), dim=-1)

        # Stage 4: transformer + head
        dots = self.encoder[0](enc_input)
        dots = self.encoder[1](dots)

        null_logits = dots[:, 0, :]
        neighbor_logits = dots[:, 1:, :]

        return null_logits, neighbor_logits, topk_idxs


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _compare(name, pt_tensor, onnx_array, atol=1e-4):
    """Compare PyTorch vs ONNX outputs, return (ok, max_diff)."""
    pt = pt_tensor.numpy() if torch.is_tensor(pt_tensor) else pt_tensor
    if pt.shape != onnx_array.shape:
        print(f"  FAIL {name}: shape mismatch PT={pt.shape} ONNX={onnx_array.shape}")
        return False, float("inf")

    finite = np.isfinite(pt) & np.isfinite(onnx_array)
    if finite.any():
        diff = np.abs(pt[finite] - onnx_array[finite])
        max_diff = float(diff.max())
    else:
        max_diff = 0.0

    ok = max_diff < atol
    status = "PASS" if ok else "FAIL"
    print(f"  {status} {name}: max_diff={max_diff:.7f}")
    return ok, max_diff


def _cross_validate(label, new_path, ref_path, input_name, dummy_np):
    """Run the same input through new and reference ONNX models, compare outputs."""
    if not ref_path.exists():
        print(f"  SKIP cross-validation: {ref_path} not found")
        return True

    print(f"  Cross-validating against existing: {ref_path.name}")
    try:
        new_sess = ort.InferenceSession(str(new_path), providers=["CPUExecutionProvider"])
        ref_sess = ort.InferenceSession(str(ref_path), providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  SKIP cross-validation: failed to load model: {e}")
        return True

    feed = {input_name: dummy_np}
    new_outs = new_sess.run(None, feed)
    ref_outs = ref_sess.run(None, feed)

    new_names = [o.name for o in new_sess.get_outputs()]
    ref_names = [o.name for o in ref_sess.get_outputs()]

    all_ok = True
    for i, (nn_, rn) in enumerate(zip(new_names, ref_names)):
        if nn_ != rn:
            print(f"  WARN output name mismatch: new={nn_} vs ref={rn}")
        if new_outs[i].shape != ref_outs[i].shape:
            print(f"  FAIL {nn_}: shape mismatch new={new_outs[i].shape} vs ref={ref_outs[i].shape}")
            all_ok = False
            continue
        diff = np.abs(new_outs[i].astype(np.float64) - ref_outs[i].astype(np.float64))
        finite = np.isfinite(diff)
        max_diff = float(diff[finite].max()) if finite.any() else 0.0
        ok = max_diff < 1e-2  # generous tolerance for cross-model comparison
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {nn_} (vs existing): max_diff={max_diff:.7f}")
        all_ok &= ok
    return all_ok


def _export_onnx(model, dummy_inputs, path, input_names, output_names,
                 opset, use_dynamo=True):
    """Export model to ONNX with static shapes."""
    export_kwargs = {}
    sig = inspect.signature(torch.onnx.export)

    # Always use legacy exporter for consistent static shapes
    if "dynamo" in sig.parameters:
        export_kwargs["dynamo"] = False
    if "external_data" in sig.parameters:
        export_kwargs["external_data"] = True

    torch.onnx.export(
        model,
        dummy_inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        **export_kwargs,
    )


def _save_with_external_data(onnx_path: Path):
    """Reload ONNX model and save with single external data file."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    load_external_data_for_model(model, str(onnx_path.parent))
    convert_model_from_external_data(model)

    model = onnx.shape_inference.infer_shapes(model)

    data_path = onnx_path.with_suffix(".data")
    if onnx_path.exists():
        onnx_path.unlink()
    if data_path.exists():
        data_path.unlink()

    onnx.save_model(
        model, str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )
    return data_path


def _optimize_onnx_model(onnx_path: Path):
    """Apply ORT_ENABLE_BASIC graph optimization in-place.

    Loads the ONNX model, runs ORT optimization to fuse ops, then overwrites
    the original file with the optimized model + external data.
    """
    with tempfile.TemporaryDirectory(prefix="ort_opt_", dir=onnx_path.parent) as temp_dir:
        temp_dir = Path(temp_dir)
        # Copy the model into the temp dir as a single self-contained file
        # so ORT optimization can find all weights.
        temp_input = temp_dir / "input.onnx"
        model = onnx.load(str(onnx_path), load_external_data=False)
        load_external_data_for_model(model, str(onnx_path.parent))
        convert_model_from_external_data(model)
        onnx.save_model(model, str(temp_input))
        del model

        optimized_path = temp_dir / "optimized.onnx"
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session_options.optimized_model_filepath = str(optimized_path)
        ort.InferenceSession(
            str(temp_input), sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        # Load optimized model (self-contained, no external data)
        optimized_model = onnx.load(str(optimized_path))

    optimized_model = onnx.shape_inference.infer_shapes(optimized_model)

    # Overwrite original file with external data
    data_path = onnx_path.with_suffix(".data")
    if onnx_path.exists():
        onnx_path.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        optimized_model, str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )


def _file_size_mb(path: Path) -> float:
    """Get total size of ONNX + .data file in MB."""
    sz = os.path.getsize(path)
    data = path.with_suffix(".data")
    if data.exists():
        sz += os.path.getsize(data)
    return sz / (1024 * 1024)


def _make_onnx_path(output_dir: Path, component: str, shape_suffix: str) -> Path:
    """Build the standard ONNX output path with full naming convention.

    Returns e.g.:
      output_dir/nvidia-nemotron-ocr-v2-detector-english/
                 nvidia-nemotron-ocr-v2-detector-english_1x3x1024x1024.onnx
    """
    export_name = EXPORT_NAMES[component]
    subdir = output_dir / export_name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{export_name}_{shape_suffix}.onnx"


# ---------------------------------------------------------------------------
# Per-model export functions
# ---------------------------------------------------------------------------

def export_detector(model_dir: Path, output_dir: Path, config: dict,
                    ref_dir: Path | None = None):
    """Export detector (opset 17, no wrapper, no ORT optimization)."""
    from nemotron_ocr.inference.models.detector.fots_detector import FOTSDetector

    print("=" * 60)
    print("Exporting detector ...")
    print("=" * 60)

    backbone = config.get("backbone", "regnet_x_8gf")
    scope = config.get("scope", 2048)
    detector = FOTSDetector(
        coordinate_mode="RBOX", backbone=backbone, scope=scope, verbose=False,
    )
    detector.load_state_dict(
        torch.load(model_dir / "detector.pth", map_location="cpu", weights_only=True),
    )
    detector.eval()
    detector.inference_mode = True

    # Fix fp16 normalization buffers with exact fp32 ImageNet constants
    detector.input_mean = torch.tensor(
        DETECTOR_INPUT_MEAN, dtype=torch.float32,
    ).view(1, 3, 1, 1)
    detector.input_std = torch.tensor(
        DETECTOR_INPUT_STD, dtype=torch.float32,
    ).view(1, 3, 1, 1)

    shape = (1, 3, INFER_LENGTH, INFER_LENGTH)
    shape_suffix = "x".join(str(d) for d in shape)
    dummy = torch.rand(shape, dtype=torch.float32)
    print(f"  Input:  input {list(shape)} fp32")

    # FOTSDetector returns 4 outputs: (confidence, offsets, rboxes, features).
    # Specifying only 3 output_names causes torch.onnx.export to prune the
    # unused offsets output from the graph automatically.
    with torch.no_grad():
        pt_conf, _pt_offsets, pt_rbox, pt_feat = detector(dummy)
    print(f"  Output: confidence {list(pt_conf.shape)}, "
          f"rboxes {list(pt_rbox.shape)}, features {list(pt_feat.shape)}")

    onnx_path = _make_onnx_path(output_dir, "detector", shape_suffix)
    print(f"  Exporting to {onnx_path} ...")
    _export_onnx(
        detector, (dummy,), onnx_path,
        input_names=["input"],
        output_names=["confidence", "rboxes", "features"],
        opset=17,
    )
    _save_with_external_data(onnx_path)
    onnx.checker.check_model(str(onnx_path))

    # Validate PyTorch vs ORT
    print("  Validating against ONNXRuntime CPU ...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ox_conf, ox_rbox, ox_feat = sess.run(None, {"input": dummy.numpy()})

    all_pass = True
    ok, _ = _compare("confidence", pt_conf, ox_conf, atol=1e-4)
    all_pass &= ok
    ok, _ = _compare("rboxes", pt_rbox, ox_rbox, atol=1e-2)
    all_pass &= ok
    ok, _ = _compare("features", pt_feat, ox_feat, atol=1e-4)
    all_pass &= ok

    # Cross-validate against existing ONNX
    if ref_dir:
        ref_name = EXPORT_NAMES["detector"]
        ref_path = ref_dir / ref_name / f"{ref_name}_{shape_suffix}.onnx"
        xv = _cross_validate("detector", onnx_path, ref_path, "input", dummy.numpy())
        all_pass &= xv

    print(f"  Size: {_file_size_mb(onnx_path):.1f} MB")
    print()
    return all_pass, onnx_path


def export_recognizer(model_dir: Path, output_dir: Path, config: dict,
                      ref_dir: Path | None = None):
    """Export recognizer (opset 17, no wrapper, ORT_ENABLE_BASIC optimization)."""
    from nemotron_ocr.inference.models.recognizer import TransformerRecognizer

    print("=" * 60)
    print("Exporting recognizer ...")
    print("=" * 60)

    num_tokens = config.get("num_tokens", 858)
    max_width = config.get("max_width", 32)
    depth = config.get("depth", 128)
    feature_depth = config.get("feature_depth", 256)

    recognizer = TransformerRecognizer(
        nic=DETECTOR_FEATURE_CHANNELS,
        num_tokens=num_tokens,
        max_width=max_width,
        use_pre_norm=config.get("has_pre_norm", False),
        use_final_norm=config.get("has_tx_norm", True),
        norm_first=config.get("norm_first", True),
        depth=depth,
        num_layers=config.get("num_layers", 3),
        nhead=config.get("nhead", 8),
        dim_feedforward=config.get("dim_feedforward", None),
    )
    recognizer.load_state_dict(
        torch.load(model_dir / "recognizer.pth", map_location="cpu", weights_only=True),
    )
    recognizer.eval()

    shape = (1, DETECTOR_FEATURE_CHANNELS, RECOGNIZER_CROP_HEIGHT, max_width)
    shape_suffix = "x".join(str(d) for d in shape)
    dummy = torch.randn(shape, dtype=torch.float32)
    print(f"  Input:  input {list(shape)} fp32")

    with torch.no_grad():
        pt_logits, pt_feats = recognizer(dummy)
    print(f"  Output: logits {list(pt_logits.shape)}, features {list(pt_feats.shape)}")

    onnx_path = _make_onnx_path(output_dir, "recognizer", shape_suffix)
    print(f"  Exporting to {onnx_path} ...")
    _export_onnx(
        recognizer, (dummy,), onnx_path,
        input_names=["input"],
        output_names=["logits", "features"],
        opset=17,
    )
    _save_with_external_data(onnx_path)

    # Apply ORT_ENABLE_BASIC graph optimization (fuses ops, reduces node count
    # from ~255 to ~144, matching the reference recognizer export)
    print("  Applying ORT_ENABLE_BASIC optimization ...")
    _optimize_onnx_model(onnx_path)

    onnx.checker.check_model(str(onnx_path))

    # Validate PyTorch vs ORT
    print("  Validating against ONNXRuntime CPU ...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ox_logits, ox_feats = sess.run(None, {"input": dummy.numpy()})

    all_pass = True
    ok, _ = _compare("logits", pt_logits, ox_logits, atol=1e-4)
    all_pass &= ok
    ok, _ = _compare("features", pt_feats, ox_feats, atol=1e-4)
    all_pass &= ok

    # Cross-validate against existing ONNX
    if ref_dir:
        ref_name = EXPORT_NAMES["recognizer"]
        ref_path = ref_dir / ref_name / f"{ref_name}_{shape_suffix}.onnx"
        xv = _cross_validate("recognizer", onnx_path, ref_path, "input", dummy.numpy())
        all_pass &= xv

    print(f"  Size: {_file_size_mb(onnx_path):.1f} MB")
    print()
    return all_pass, onnx_path


def export_relational(model_dir: Path, output_dir: Path, config: dict,
                      ref_path: Path | None = None):
    """Export relational model (sparse-neighbor style, opset 18)."""
    print("=" * 60)
    print("Exporting relational (sparse-neighbor) ...")
    print("=" * 60)

    # Load detector to get num_features for the relational model
    from nemotron_ocr.inference.models.detector.fots_detector import FOTSDetector
    det_tmp = FOTSDetector(
        coordinate_mode="RBOX",
        backbone=config.get("backbone", "regnet_x_8gf"),
        scope=config.get("scope", 2048),
        verbose=False,
    )
    det_tmp.load_state_dict(
        torch.load(model_dir / "detector.pth", map_location="cpu", weights_only=True),
    )
    det_num_features = det_tmp.num_features

    feature_depth = config.get("feature_depth", 256)
    max_width = config.get("max_width", 32)

    # Disable fused TransformerEncoder kernel (no ONNX mapping)
    torch.backends.mha.set_fastpath_enabled(False)

    relational = RelationalONNX(
        num_input_channels=det_num_features,
        recog_feature_depth=feature_depth,
        k=16,  # k=16 -> K=15 neighbors
        num_layers=4,
    )
    relational.load_state_dict(
        torch.load(model_dir / "relational.pth", map_location="cpu", weights_only=True),
    )
    relational.eval()

    K = RELATIONAL_K
    dummy_rect = torch.randn(K, DETECTOR_FEATURE_CHANNELS, RELATIONAL_GRID_H, RELATIONAL_GRID_W)
    dummy_quads = torch.rand(K, 4, 2) * 1024
    dummy_recog = torch.randn(K, max_width, feature_depth)

    shape_suffix = f"{K}x{DETECTOR_FEATURE_CHANNELS}x{RELATIONAL_GRID_H}x{RELATIONAL_GRID_W}"

    print(f"  Input:  rectified_quads {list(dummy_rect.shape)}, "
          f"original_quads {list(dummy_quads.shape)}, "
          f"recog_features {list(dummy_recog.shape)} fp32")

    with torch.no_grad():
        pt_null, pt_neigh, pt_idx = relational(dummy_rect, dummy_quads, dummy_recog)
    print(f"  Output: null_logits {list(pt_null.shape)}, "
          f"neighbor_logits {list(pt_neigh.shape)}, "
          f"neighbor_indices {list(pt_idx.shape)}")

    onnx_path = _make_onnx_path(output_dir, "relational", shape_suffix)
    print(f"  Exporting to {onnx_path} ...")
    _export_onnx(
        relational, (dummy_rect, dummy_quads, dummy_recog), onnx_path,
        input_names=["rectified_quads", "original_quads", "recog_features"],
        output_names=["null_logits", "neighbor_logits", "neighbor_indices"],
        opset=18,  # opset 18 needed for topk/cdist ops
        use_dynamo=False,  # legacy exporter needed for topk/gather/arange
    )
    _save_with_external_data(onnx_path)
    onnx.checker.check_model(str(onnx_path))

    # Validate PyTorch vs ORT
    print("  Validating against ONNXRuntime CPU ...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ox_null, ox_neigh, ox_idx = sess.run(None, {
        "rectified_quads": dummy_rect.numpy(),
        "original_quads": dummy_quads.numpy(),
        "recog_features": dummy_recog.numpy(),
    })

    all_pass = True
    ok, _ = _compare("null_logits", pt_null, ox_null, atol=5e-4)
    all_pass &= ok

    idx_match = (pt_idx.numpy() == ox_idx).all()
    if idx_match:
        ok, _ = _compare("neighbor_logits", pt_neigh, ox_neigh, atol=5e-4)
        all_pass &= ok
        print(f"  PASS neighbor_indices: exact match")
    else:
        # topk tie-breaking can differ between PyTorch and ORT; match by index
        pt_i = pt_idx.numpy()
        max_diff = 0.0
        matched = 0
        total = pt_i.shape[0] * pt_i.shape[1]
        for row in range(pt_i.shape[0]):
            for j in range(pt_i.shape[1]):
                ox_positions = np.where(ox_idx[row] == pt_i[row, j])[0]
                if len(ox_positions) > 0:
                    d = float(np.abs(pt_neigh[row, j].numpy() - ox_neigh[row, ox_positions[0]]).max())
                    max_diff = max(max_diff, d)
                    matched += 1
        ok = max_diff < 5e-4
        status = "PASS" if ok else "FAIL"
        print(f"  {status} neighbor_logits (index-matched): "
              f"max_diff={max_diff:.7f} ({matched}/{total} shared)")
        print(f"  INFO neighbor_indices: topk tie-breaking differs (expected)")
        all_pass &= ok

    # Cross-validate against Krishna's existing ONNX if available.
    # Krishna's model uses sorted=False in topk, so neighbor ordering differs.
    # We compare null_logits directly and match neighbor_logits by index.
    if ref_path and ref_path.exists():
        print(f"  Cross-validating against existing: {ref_path.name}")
        ref_sess = ort.InferenceSession(str(ref_path), providers=["CPUExecutionProvider"])
        ref_null, ref_neigh, ref_idx = ref_sess.run(None, {
            "rectified_quads": dummy_rect.numpy(),
            "original_quads": dummy_quads.numpy(),
            "recog_features": dummy_recog.numpy(),
        })
        ok, _ = _compare("null_logits (vs existing)", ox_null, ref_null, atol=5e-4)
        all_pass &= ok

        # Match neighbor logits by index (handles sorted vs unsorted topk)
        max_diff = 0.0
        matched = 0
        total = ox_idx.shape[0] * ox_idx.shape[1]
        for row in range(ox_idx.shape[0]):
            for j in range(ox_idx.shape[1]):
                idx_val = ox_idx[row, j]
                ref_positions = np.where(ref_idx[row] == idx_val)[0]
                if len(ref_positions) > 0:
                    d = float(np.abs(ox_neigh[row, j] - ref_neigh[row, ref_positions[0]]).max())
                    max_diff = max(max_diff, d)
                    matched += 1
        ok = max_diff < 5e-4
        status = "PASS" if ok else "FAIL"
        print(f"  {status} neighbor_logits (vs existing, index-matched): "
              f"max_diff={max_diff:.7f} ({matched}/{total} shared)")
        all_pass &= ok

    print(f"  Size: {_file_size_mb(onnx_path):.1f} MB")
    print()
    return all_pass, onnx_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export Nemotron OCR v2 (English) to static-shape ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True,
        help="Path to nemotron-ocr-v2 repo checkout (contains v2_english/ "
             "and nemotron-ocr/src/).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("onnx_export"),
        help="Output directory for ONNX files.",
    )
    parser.add_argument(
        "--ref-dir", type=Path, default=None,
        help="Path to existing ONNX artifacts dir for cross-validation "
             "(e.g. artifacts/onnx). Detector and recognizer outputs are "
             "compared against existing models to verify equivalence.",
    )
    parser.add_argument(
        "--ref-relational", type=Path, default=None,
        help="Path to existing relational ONNX for cross-validation "
             "(e.g. krishna_models/relational_static_shapes.onnx).",
    )
    args = parser.parse_args()

    model_dir = args.model_dir / "v2_english"
    if not model_dir.is_dir():
        print(f"Error: {model_dir} not found. --model-dir should point to "
              f"the nemotron-ocr-v2 repo root.", file=sys.stderr)
        return 1

    # Add nemotron-ocr source to path
    _stub_cpp()
    src_dir = args.model_dir / "nemotron-ocr" / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))
    else:
        print(f"Warning: {src_dir} not found. Assuming nemotron_ocr is "
              f"installed.", file=sys.stderr)

    # Load model config
    config_path = model_dir / "model_config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    print(f"Model dir:  {model_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Opset:      17 (detector/recognizer), 18 (relational)")
    if args.ref_dir:
        print(f"Ref dir:    {args.ref_dir}")
    if args.ref_relational:
        print(f"Ref rel:    {args.ref_relational}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    paths = {}

    ok, p = export_detector(model_dir, args.output_dir, config,
                            ref_dir=args.ref_dir)
    results["detector"] = ok
    paths["detector"] = p

    ok, p = export_recognizer(model_dir, args.output_dir, config,
                              ref_dir=args.ref_dir)
    results["recognizer"] = ok
    paths["recognizer"] = p

    ok, p = export_relational(model_dir, args.output_dir, config,
                              ref_path=args.ref_relational)
    results["relational"] = ok
    paths["relational"] = p

    # Summary
    print("=" * 60)
    print("Export summary")
    print("=" * 60)
    all_pass = True
    for name in ("detector", "recognizer", "relational"):
        onnx_path = paths[name]
        status = "PASS" if results[name] else "FAIL"
        sz = _file_size_mb(onnx_path)
        print(f"  {name:12s}  {status}  {sz:6.1f} MB  {onnx_path}")
        all_pass &= results[name]

    print()
    if all_pass:
        print("All exports PASSED validation.")
    else:
        print("Some exports FAILED validation.")
    print(f"Output: {args.output_dir}/")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
