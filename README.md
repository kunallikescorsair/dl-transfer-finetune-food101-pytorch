# Transfer Learning and Fine-Tuning on Food101

This project explores the use of transfer learning and fine-tuning with deep convolutional neural networks on the Food-101 image classification dataset.  
Developed for the Deep Learning (94691) course in the Master of Data Science and Innovation program at the University of Technology Sydney (UTS).

---

## Project Structure

- `transfer-learning-food101.ipynb`: Transfer learning using ResNet-50, MobileNet V3, and GoogLeNet with frozen convolutional layers
- `finetuning-resnet50-food101.ipynb`: Fine-tuning ResNet-50 by unfreezing selective layers
- `requirements.txt`: Python dependencies for running both notebooks

---

## Dataset: Food-101

- 101,000 images across 101 food categories
- Pre-split into 750 training and 250 test images per class
- High-resolution RGB images with natural variation
- Download: [http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)

### Preprocessing

- Train: RandomResizedCrop (224×224), Horizontal Flip
- Test: Resize → CenterCrop (224×224)
- Normalization: ImageNet mean and std

---

## Part A: Transfer Learning (Frozen Models)

### Models Evaluated

| Model        | Params (M) | Best Val Accuracy |
|--------------|------------|-------------------|
| ResNet-50    | 25.6       | 63.91%            |
| MobileNet V3 | 5.4        | 61.89%            |
| GoogLeNet    | 6.8        | 48.30%            |

### Setup

- Replaced final head with custom 3-layer classifier
- Frozen all pretrained layers
- Optimizer: Adam (lr = 0.001), Batch Size: 64, Epochs: 5

---

## Part B: Fine-Tuning ResNet-50

- Unfroze `layer4` of ResNet-50
- Reduced learning rate to 1e-4
- Reused training pipeline and evaluation strategy

### Fine-Tuned Performance

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 5     | 71.00%    | **81.89%** |

### Gains

- +18% over frozen model
- Higher F1 across most classes
- Better separation of visually similar foods (e.g., cheesecake vs. carrot_cake)

---

## Key Takeaways

- ResNet-50 is best for general performance
- Fine-tuning specific layers significantly boosts accuracy
- Shallow models (e.g., GoogLeNet) underperform on complex, fine-grained datasets

---

## Future Improvements

- Gradually unfreeze more layers (e.g., layer3)
- Add learning rate schedulers (cosine annealing, step decay)
- Use advanced augmentations (MixUp, RandAugment)
- Try newer models (EfficientNet, ConvNeXt)
- Build ensemble for further performance gain

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Academic Integrity Notice
This repository is shared for educational and demonstration purposes only.
Do not copy or submit this work for academic credit. Unauthorized use may be considered academic misconduct.

Author
Kunal Gurung
Master of Data Science and Innovation
University of Technology Sydney
