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

## Installation & Setup (Windows PowerShell)

We use **`uv`** as our lightning-fast package and Python version manager. It ensures your environment is exactly identical to the required setup without conflicting with your system.

Open a PowerShell window in the `think_fast` directory:

```powershell
# 1. Install uv (if you haven't already)
irm https://astral.sh/uv/install.ps1 | iex

# 2. Sync the environment (uv will automatically download Python and all dependencies!)
uv sync

# 3. Activate the virtual environment
.\.venv\Scripts\activate
```

*(Note: If you plan to compile the TensorRT engine, TensorRT must be installed directly from NVIDIA's developer portal).*

## Data Preparation

Extract the nuScenes dataset (`v1.0-trainval03_blobs.tgz` and others) to a directory (e.g., `D:\TATA\data 1\v1.0-trainval03_blobs`).
Ensure the `v1.0-trainval` JSON metadata is present in that root directory.

## Pipeline Usage (PowerShell)

**Important:** Before running any of the commands below, make sure your virtual environment is active (`.\think_fast\.venv\Scripts\activate`) and you are running these commands from the **parent directory** (`D:\TATA\data 1`) so that Python recognizes `think_fast` as a module.

```powershell
cd "D:\TATA\data 1"
```

### 1. Run the Unit Tests
Make sure everything is wired up correctly:
```powershell
pytest think_fast/tests/
```

### 2. Training
Train the 5-channel YOLOv11 model with weight transfer from a standard `yolo11n.pt` checkpoint. 
*(Adjust the `--dataroot` path to match your exact nuScenes extraction folder).*
```powershell
python -m think_fast.model.train --dataroot "D:\TATA\data 1\v1.0-trainval03_blobs" --pretrained yolo11n.pt --epochs 100 --batch 4
```

### 3. ONNX Export
Export the trained model to ONNX with dynamic batch sizing:
```powershell
python -m think_fast.inference.export_onnx --weights runs/think_fast/best.pt --output model_5ch.onnx
```

### 4. TensorRT Build (Production Only)
Compile an FP16 engine tailored to your GPU architecture:
```python
from think_fast.inference.tensorrt_engine import TRTEngineBuilder
TRTEngineBuilder.build_from_onnx("model_5ch.onnx", "model_5ch_fp16.trt", fp16=True)
```

### 5. Running the End-to-End Pipeline

Run the orchestrator on a set of nuScenes samples. You can use `--mode dev` (PyTorch) or `--mode prod` (TensorRT).

**Step A:** Start the MCP Stub Server (VLA simulator) in a separate PowerShell window (remember to activate your venv there too):
```powershell
cd "D:\TATA\data 1"
.\think_fast\.venv\Scripts\activate
python -m think_fast.mcp.mcp_server_stub
```

**Step B:** Run the pipeline in your original window:
```powershell
python -m think_fast.pipeline.think_fast_pipeline --dataroot "D:\TATA\data 1\v1.0-trainval03_blobs" --weights runs/think_fast/best.pt --mode dev --demo --n_samples 5
```

## C++ Integration

The `threat/` and `actuation/` directories contain header-only and standalone C++ implementations designed to be compiled natively into ROS2 nodes or embedded controllers.

Compile the threat matrix standalone demo (requires MinGW/GCC):
```powershell
cd think_fast
g++ -std=c++17 -O2 -o threat_demo.exe threat/threat_matrix.cpp -DTHINK_FAST_STANDALONE_DEMO
.\threat_demo.exe
```
