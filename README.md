# Real-Time On-Device Multi-Vessel Tracking for UAV with Trapezoid Convolution (IROS 2026)


 **Zijie Zhang<sup>1</sup>, Changhong Fu<sup>1,†</sup>, Mengyuan Li<sup>1</sup>, Haobo Zuo<sup>2</sup>, Guangze Zheng<sup>2</sup>, Bowen Li<sup>3</sup>**

<sup>1</sup>Tongji University, Shanghai, China
<sup>2</sup>University of Hong Kong, Hong Kong, China
<sup>3</sup>Carnegie Mellon University, Pittsburgh, USA
<sup>†</sup>Corresponding author

[![IROS](https://img.shields.io/badge/IROS-2026-blue)](https://2026.ieee-iros.org/)
[![MobileMVT](https://img.shields.io/badge/MobileMVT-green)](https://github.com/vision4robotics/MobileMVT)
[![VesselMOT](https://img.shields.io/badge/VesselMOT-red)](https://pan.baidu.com/s/1d9XinUwxi1fNIoVyoI94aA?pwd=vfax)
## Abstract
Visual multi-vessel tracking is critical for intelligent maritime surveillance yet challenging due to the complexity of efficiently modeling rigid vessel structures across diverse scales and viewpoints. Existing methods struggle to reconcile fine-grained geometric modeling with the real-time constraints of resource-constrained UAV edge platforms, where performance is further hindered by scale variations and viewpoint perturbations. To address these challenges, a novel visual multi-vessel tracking framework (MobileMVT) is proposed for real-time deployment on UAV-mounted edge electro-optical devices. At its core, the Trapezoid Convolution (TConv) encodes vessel-specific geometric priors through trapezoid-aligned sampling, enhancing shape preservation and feature discrimination. Building upon it, TConvNet serves as a compact backbone to improve recognition accuracy and tracking consistency with minimal computational cost. Furthermore, an uncertainty-aware viewpoint compensator is integrated to mitigate UAV-induced shifts and motion uncertainties, ensuring stable trajectory estimation. To facilitate comprehensive evaluation, we introduce VesselMOT, a large-scale benchmark comprising 150 sequences and over 180k annotated frames featuring diverse scenarios. Extensive experiments and real-world tests demonstrate that MobileMVT achieves superior tracking accuracy and real-time performance in dynamic maritime environments.

---

## 🔥 Highlights

- **Trapezoid Convolution (TConv)** — encodes vessel-specific trapezoidal contour priors into the sampling topology via affine-constrained offsets, improving contour fidelity at marginal overhead.
- **TConvNet** — a compact detection backbone integrating TConv into a hierarchical feature pyramid for fine-grained, scale-aware vessel perception.
- **UAVC tracker** — adaptively rectifies UAV-induced perturbations, ensuring stable trajectory estimation in unstructured environments.
- **Edge-ready** — runs at 43.75 FPS on the Hi3403V100 edge AI chip with 3.75 M params and 5.11 G MACs.
- **Large-scale VMVT benchmark (VesselMOT)** — a challenging benchmark comprising 150 sequences and over 180k annotated frames. Serving as the largest UAV-captured multi-vessel tracking benchmark, it covers diverse waterways and environmental conditions, providing a comprehensive testbed for practical maritime monitoring.

## 🏗️ Repository Structure

```
├── train.py                     # TConvNet training entry
├── requirements.txt
├── vesselmot-detection.yaml     # Dataset config
├── weights/
├── tracker/                     # UAVC
├── detector/                    # Homography mapper
└── ultralytics/                 # Detector pipeline
```

---

## 🚀 Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.6 (for GPU acceleration)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/vision4robotics/MobileMVT.git
cd MobileMVT

# Create conda environment
conda create -n mobilemvt python=3.8
conda activate mobilemvt

# Install dependencies
pip install -r requirements.txt
```
## 📦 Dataset Preparation

### VesselMOT Dataset

**Download (Baidu Netdisk):** https://pan.baidu.com/s/1d9XinUwxi1fNIoVyoI94aA?pwd=vfax

The default VesselMOT benchmark follows the standard MOT data formats.

**Expected directory structure:**
```text
vesselmot
+-- test/
    +-- <sequence_name>/
        +-- img1/
        |   +-- 000001.jpg
        |   +-- ...
        +-- gt/
            +-- gt.txt
+-- train/
+-- val/
```
---

## 📊 Quick Start

```bash
python demo.py \
    --video    demo/vesselmot-demo.mp4 \
    --cam_para demo/vesselmot.txt
```

The annotated video is written to `output/output.mp4`. Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--video` | `demo/vesselmot-demo.mp4` | Input video file |
| `--cam_para` | `demo/vesselmot.txt` | Camera parameter file |
| `--wx`, `--wy` | `4.20`, `2.60` | Ground-plane process-noise factors |
| `--vmax` | `1.0` | Maximum expected speed (m/frame) |
| `--a` | `5.0` | Association gating threshold |
| `--cdt` | `3.0` | Coasted-track deletion time (frames) |
| `--high_score` | `0.58` | High-confidence detection threshold |
| `--conf_thresh` | `0.29` | Detection confidence threshold |


UAVC projects detections onto a ground reference plane, requiring a camera
parameter file in the following format:

```
RotationMatrices
r11 r12 r13
r21 r22 r23
r31 r32 r33

TranslationVectors
tx ty tz

IntrinsicMatrix
fx 0  cx
0  fy cy
0  0  1
```

The bundled `demo/vesselmot.txt` can be used directly for the demo. For new
sequences, estimate the parameters from a representative frame (rotation matrix,
translation vector in millimeters, focal length, and principal point).

---

## 🏋️ Training

MobileMVT follows the tracking-by-detection paradigm, so training amounts to
training the geometry-aware **TConvNet** detector; the UAVC tracker is
training-free and runs on top of the detector at inference time.

### 1. Prepare the detection dataset

Convert the VesselMOT sequences into the Ultralytics detection format (one
`.txt` label per image, `<class> <cx> <cy> <w> <h>` normalized), then edit
[`vesselmot-detection.yaml`](vesselmot-detection.yaml) so that `path` points to
your dataset root:

```text
vesselmot_detection/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt   # YOLO format: <cls> <cx> <cy> <w> <h>
```

### 2. Launch training

```bash
python train.py \
    --model ultralytics/cfg/models/tconv/tconv3win-r18-320.yaml \
    --data  vesselmot-detection.yaml \
    --epochs 200 --imgsz 320 --batch 32 --device 0 \
    --name tconv-w3-r18-320-vesselmot
```

Key arguments (see `train.py --help` for the full list):

| Argument | Default | Description |
|---|---|---|
| `--model` | `.../tconv3win-r18-320.yaml` | Model config |
| `--data` | `vesselmot-detection.yaml` | Dataset config |
| `--epochs` | `200` | Training epochs |
| `--imgsz` | `320` | Input resolution |
| `--batch` | `32` | Batch size |
| `--lr0` | `1e-4` | Initial learning rate (AdamW) |
| `--weight_decay` | `5e-4` | Weight decay |
---

## 📖 Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{zhang2026mobilemvt,
  title     = {Real-Time On-Device Multi-Vessel Tracking for UAV
               with Trapezoid Convolution},
  author    = {Zhang, Zijie and Fu, Changhong and Li, Mengyuan and
               Zuo, Haobo and Zheng, Guangze and Li, Bowen},
  booktitle = {Proceedings of the IEEE/RSJ International Conference on
               Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgements
This work builds upon several excellent open-source projects:

- [UCMCTrack](https://github.com/corfyi/UCMCTrack) - Tracker association pipeline
- [Ultralytics](https://github.com/ultralytics/ultralytics) - Detector pipeline

## 📄 License
This project is licensed under the AGPL-3.0 License.

