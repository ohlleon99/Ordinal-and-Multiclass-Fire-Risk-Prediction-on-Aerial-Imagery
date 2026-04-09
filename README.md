# Ordinal-and-Multiclass-Fire-Risk-Prediction-on-Aerial-Imagery


This project investigates **wildfire risk prediction from high-resolution aerial imagery** using deep learning, hybrid models, and ordinal regression.

It compares multiple approaches on the **FireRisk dataset**, including:

- CNN: ResNet18  
- Vision Transformer: Swin Transformer V2-B  
- Hybrid model: Swin feature extractor + SVC  
- Ordinal regression using CORN  

### Results

Best-performing model (Swin + SVC):

- **Accuracy:** 61.25%  
- **F1-score:** 58.44%  

Ordinal regression improves prediction consistency:

- **MAE reduced from 0.59 → 0.54**

---

## Installation

Recommended: run on **Google Colab (A100 GPU)**

Install dependencies:

```bash
pip install torch torchvision lightning coral-pytorch datasets scikit-learn matplotlib einops tensorboard
```
## Dataset

This project uses the **FireRisk dataset** from Hugging Face:

- Dataset: `blanchon/FireRisk`  
- Source: NAIP aerial imagery  
- 7 fire risk classes (imbalanced)  

### Download

The dataset is automatically downloaded in the code:

```python
from datasets import load_dataset
dataset = load_dataset("blanchon/FireRisk")
```
Notes:

- The dataset is not included in this repository
- A 60/20/20 split (train/val/test) is created during runtime

### Usage Notes:
- Run chapters 1 (Imports) and 3 (Dataset) before all other chapters
- Individual sections depend on previous components (see comments in beginning of chapters)
- If not running individual chapters, the script can be executed as a whole
