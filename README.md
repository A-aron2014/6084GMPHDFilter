# GM-PHD Multi-Target Tracking

A **multi-target optical tracking framework** utilizing a **Finite Set Statistics (FISST) Gaussian Mixture Probability Hypothesis Density (GM-PHD) filter** with **Meta SAM3** object segmentation for UAV-based target tracking in **RGB and infrared (IR)** imagery.

This project demonstrates a probabilistic multi-target tracking pipeline that combines modern vision-based object detection with principled Bayesian estimation. The system uses **SAM3 detections to generate measurement sets**, which are then processed through a **GM-PHD filter** to estimate target centroids, manage clutter, and maintain target hypotheses under uncertainty.

The resulting outputs visualize:

* **True object centroids**
* **Predicted target centroids**
* **Uncertainty ellipsoids** derived from covariance propagation and Mahalanobis distance thresholds
* **Track birth and maintenance behavior** under uncertain detections

---

## Features

* **GM-PHD multi-target tracking**

  * FISST-based random finite set formulation
  * Handles clutter, missed detections, and varying target cardinality
  * Gaussian mixture representation of target intensity

* **Meta SAM3 object detection**

  * Optical segmentation-based target detection
  * Supports both **RGB and IR UAV imagery**

* **Dynamic birth model**

  * Incorporates **SAM3 detection confidence scores**
  * Enables adaptive target initialization

* **Uncertainty-aware tracking**

  * Mahalanobis-distance-based covariance reasoning
  * Predicted uncertainty ellipsoids visualized on image outputs

* **GPU acceleration**

  * CUDA-enabled inference pipeline
  * Designed for NVIDIA hardware

---

## Example Pipeline

```text
UAV RGB / IR Imagery
          ↓
   Meta SAM3 Detection
          ↓
 Measurement Generation
 (centroids + confidence)
          ↓
     GM-PHD Filter
  (predict / update /
 prune / merge / birth)
          ↓
Tracking Visualization
(predictions + covariance)
```

---

## Methodology

### Gaussian Mixture Probability Hypothesis Density (GM-PHD)

This project implements a **GM-PHD filter**, a **Finite Set Statistics (FISST)** approach for multi-target tracking.

Unlike traditional **Track-Oriented Multiple Hypothesis Tracking (TO-MHT)** or **Hypothesis-Oriented MHT (HO-MHT)** methods, the GM-PHD filter propagates the **first-order moment (intensity)** of the multi-target posterior rather than enumerating individual target hypotheses. This provides a mathematically principled framework for:

* Unknown and time-varying target counts
* False detections / clutter
* Missed detections
* Target birth and death

Targets are represented as a **Gaussian mixture**, allowing efficient recursive prediction and update operations while maintaining uncertainty estimates.

### Dynamic Birth Model

A custom **dynamic target birthing model** is used to initialize new targets.

Rather than relying on static priors, the filter incorporates **SAM3 detection confidence scores** to probabilistically determine when and where new target hypotheses should be created.

This improves robustness in cluttered optical scenes and reduces false-positive track creation.

### Uncertainty Visualization

Tracking outputs include:

* Measured object centroids
* Estimated target positions
* Covariance-derived uncertainty ellipsoids

Ellipsoids are generated using **Mahalanobis-distance thresholds**, enabling visualization of state uncertainty and filter confidence.

---

## System Requirements

### Hardware

This project requires:

* **NVIDIA GPU**
* **CUDA-capable hardware**

GPU acceleration is required for efficient SAM3 inference.

### Operating System

Recommended environment:

* **WSL2**
* **Ubuntu 22.04**

Windows users should install and configure WSL2 before setup.

---

## Installation

### 1. Install WSL2 + Ubuntu 22.04

Install:

* WSL2
* Ubuntu 22.04

Then install Python and `uv`.

### 2. Install `uv`

```bash
pip install uv
```

### 3. Create the Environment

From the project root:

```bash
uv sync
```

This will install the required dependencies and generate the virtual environment.

### 4. Activate the Environment

Linux / WSL:

```bash
source .venv/bin/activate
```

---

## Model Weights Setup

Download the required **SAM3 Hugging Face model weights** and place them in the project toot directory

Example:

```text
GM-PHD-Multi-Target-Tracking/
├── sam3.pt_here
```

**Note:** Model weights are not included in this repository as they were acquired from huggingg-face.

---

## Dataset Setup

Training / evaluation imagery may be stored anywhere on your machine.

Update the dataset file path in the codebase to point to your local image directory.

You can also configure:

* Number of images to process
* Input dataset location
* Tracking runtime parameters

Dataset download link:

**[INSERT DATASET LINK HERE]**
https://www.nature.com/articles/s41597-025-06204-0
---

## Running the Tracker

After activating the environment:

1. Open the project in **VS Code**
2. Select the correct Python interpreter from `.venv`
3. Update:

   * dataset path
   * image count
   * runtime parameters (if desired)
4. Run the target tracking script

Example:

```bash
python image_detection.py run --data ~/data/M3OT/2/ir/test/2-03T --n-frames 300 --visualise
```

*(Replace with the correct entrypoint if different.)*

---

## Example Outputs

The tracker generates visualizations showing:

* **Ground-truth target centroids**
* **GM-PHD predicted centroids**
* **Covariance uncertainty ellipsoids**
* **Multi-target tracking evolution across image sequences**

Example visualization:

```text
True Centroid      → observed target
Predicted Centroid → GM-PHD estimate
Ellipsoid          → covariance uncertainty
```

---

## Lessons Learned / Future Improvements

Potential future extensions include:

* **ROS2 integration**
* **Real-time streaming support**
* **C++ GM-PHD implementation for performance**
* **Multi-sensor fusion**
* **Improved target birth/death modeling**
* **Extended UAV autonomy workflows**
* **Interactive Multiple Model Incorporation**

---

## References
This implementation was developed using concepts from:

* **Statistical Multisource-Multitarget Information Fusion by Ronald Mahler**
* **Tracking and Data Fusion by Yaakov Bar-Shalom**
* **Vo and Ma's 2006 Paper on the GM-PHD** 
* **Meta SAM3 segmentation models**
* **The M3OT Dataset**
* Research literature and open-source tracking references like stone-soup

Please cite relevant papers if extending this work for academic purposes.

