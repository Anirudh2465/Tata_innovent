# Think Fast: Reflex Architecture

"Think Fast" is a real-time, safety-critical perception and reflex system for autonomous vehicles.
It performs **Early Fusion** of 6 RGB cameras + 5 Radar sensors using a modified **YOLOv11-Nano**
model that accepts **5-channel tensors** (RGB + Radar Depth + Radar Velocity), batched across all
6 cameras into a `[6, 5, 640, 640]` GPU tensor. 

It evaluates a **Threat Matrix** (TTC) to trigger either a **Physical Reflex** (hard brake/steer) 
or an asynchronous **Semantic Dispatch** to the "Think Slow" VLA model via MCP.

## Features

- **nuScenes Integration**: Full data pipeline wrapper for 6-cam + 5-radar synchronisation.
- **Ego-Motion Compensation**: Uses IMU data to align radar sweeps to camera frames to avoid ghosting.
- **5-Channel YOLOv11-Nano**: Modified YOLO architecture incorporating depth and velocity features.
- **Weight Transfer**: Initialises 5-channel model using pre-trained 3-channel RGB YOLOv11 weights.
- **TensorRT Optimization**: Batched FP16 inference pipeline for edge deployment (Jetson Orin).
- **Threat Matrix**: C++ and Python modules to compute Enhanced Time-to-Collision (ETTC).
- **MCP Dispatcher**: FastAPI stub and async MCP client for semantic agentic handoff.

## Installation

```bash
# Create a conda environment
conda create -n thinkfast python=3.10
conda activate thinkfast

# Install dependencies
pip install -r requirements.txt

# Install TensorRT (hardware specific - refer to NVIDIA docs)
```

## Data Preparation

Extract the nuScenes dataset (`v1.0-trainval03_blobs.tgz` and others) to a directory (e.g., `/data/nuscenes`).
Ensure the `v1.0-trainval` JSON metadata is present.

## Pipeline Usage

### 1. Training

Train the 5-channel YOLOv11 model with weight transfer from a standard `yolo11n.pt` checkpoint:

```bash
python -m think_fast.model.train \
    --dataroot /data/nuscenes \
    --pretrained yolo11n.pt \
    --epochs 100 \
    --batch 4
```

### 2. ONNX Export

Export the trained model to ONNX with dynamic batch sizing:

```bash
python -m think_fast.inference.export_onnx \
    --weights runs/think_fast/best.pt \
    --output model_5ch.onnx
```

### 3. TensorRT Build (Production Only)

Compile an FP16 engine tailored to your GPU architecture:

```python
from think_fast.inference.tensorrt_engine import TRTEngineBuilder
TRTEngineBuilder.build_from_onnx("model_5ch.onnx", "model_5ch_fp16.trt", fp16=True)
```

### 4. Running the End-to-End Pipeline

Run the orchestrator on a set of nuScenes samples. You can use `--mode dev` (PyTorch) or `--mode prod` (TensorRT).

```bash
# Start the MCP Stub Server (VLA simulator) in a separate terminal:
python -m think_fast.mcp.mcp_server_stub

# Run the pipeline (Development mode)
python -m think_fast.pipeline.think_fast_pipeline \
    --dataroot /data/nuscenes \
    --weights runs/think_fast/best.pt \
    --mode dev \
    --demo \
    --n_samples 5
```

## C++ Integration

The `threat/` and `actuation/` directories contain header-only and standalone C++ implementations designed to be compiled natively into ROS2 nodes or embedded controllers.

Compile the threat matrix standalone demo:
```bash
g++ -std=c++17 -O2 -o threat_demo think_fast/threat/threat_matrix.cpp -DTHINK_FAST_STANDALONE_DEMO
./threat_demo
```
