"""
Day 10 - Script 3: Grad-CAM Visualization
Shows WHERE the model looks to make its decision — one image per class.

IMPORTANT: Run this AFTER 02_transfer_learning_strategies.py
It loads the best saved model from outputs/

Dataset path: /Users/alvi/OpenCV/data/NEU Metal Surface Defects Data/
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# ============================================================
# YOUR DATASET PATH
# ============================================================
DATA_ROOT = Path('/Users/alvi/OpenCV/data/NEU Metal Surface Defects Data')

# ============================================================
# LOAD MODEL — tries partial_finetune first, falls back to others
# ============================================================
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Detect which saved model exists
model_priority = ['partial_finetune', 'full_finetune', 'feature_extractor']
MODEL_PATH = None
STRATEGY   = None

for strategy in model_priority:
    candidate = Path(f'output/best_{strategy}.pt')
    if candidate.exists():
        MODEL_PATH = candidate
        STRATEGY   = strategy
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        "No saved model found in outputs/. "
        "Please run 02_transfer_learning_strategies.py first."
    )

print(f"Loading model: {MODEL_PATH}")
print(f"Strategy: {STRATEGY}")

# Reconstruct model architecture
CLASS_NAMES = sorted([
    f.name for f in (DATA_ROOT / 'train').iterdir() if f.is_dir()
])
N_CLASSES = len(CLASS_NAMES)
print(f"Classes ({N_CLASSES}): {CLASS_NAMES}")

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, N_CLASSES)
)
model.load_state_dict(
    torch.load(str(MODEL_PATH), map_location=DEVICE)
)
model = model.to(DEVICE)
model.eval()
print(f"Model loaded successfully on {DEVICE}")

# ============================================================
# GRAD-CAM IMPLEMENTATION FROM SCRATCH
# Hooks into the last conv layer to capture gradients
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Shows which spatial regions contributed most to the prediction.
    Critical for explaining model decisions to factory operators.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        # Register hooks — intercept forward and backward passes
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward for target class only
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize to [0, 1] and resize to 224×224
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        pred_conf = torch.softmax(output, dim=1)[0, class_idx].item()
        return cam, class_idx, pred_conf


# Hook onto the last feature block
target_layer = model.features[-1]
grad_cam     = GradCAM(model, target_layer)

# ============================================================
# TRANSFORMS FOR GRADCAM INPUT
# ============================================================
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def denormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(t * std + mean, 0, 1)

# ============================================================
# COLLECT ONE SAMPLE IMAGE PER CLASS FROM TEST SET
# ============================================================
test_samples = {}
for class_name in CLASS_NAMES:
    class_folder = DATA_ROOT / 'test' / class_name
    if not class_folder.exists():
        # Fall back to train folder if test is empty
        class_folder = DATA_ROOT / 'train' / class_name
    imgs = sorted(class_folder.glob('*.*'))
    if imgs:
        test_samples[class_name] = imgs[0]
    else:
        print(f"  WARNING: No images found for class {class_name}")

print(f"\nRunning Grad-CAM on {len(test_samples)} classes...")

# ============================================================
# GENERATE GRAD-CAM FOR EACH CLASS
# 4 columns: Original | Heatmap | Overlay | Contour
# ============================================================
n_classes    = len(test_samples)
fig, axes    = plt.subplots(n_classes, 4, figsize=(16, 4 * n_classes))
fig.suptitle(
    f'Grad-CAM: Where Does the Model Look?\n'
    f'Model: EfficientNet-B0 ({STRATEGY}) | '
    f'Red = high attention | Blue = low attention',
    fontsize=13, fontweight='bold'
)

# Column headers
col_titles = ['Original image', 'Grad-CAM heatmap', 'Overlay', 'Defect region (CAM > 0.5)']
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=11, fontweight='bold', pad=10)

for row, (class_name, img_path) in enumerate(test_samples.items()):
    raw_img = Image.open(img_path).convert('RGB')
    input_t = val_transform(raw_img).unsqueeze(0).to(DEVICE)

    # Generate CAM
    cam, pred_idx, confidence = grad_cam.generate(input_t)

    # Prepare display image (denormalized)
    img_display = denormalize(val_transform(raw_img)).permute(1, 2, 0).numpy()

    # Heatmap
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    # Overlay: 50% image + 50% heatmap
    overlay = np.clip(0.5 * img_display + 0.5 * heatmap_rgb, 0, 1)

    # Contour: find regions where CAM > 0.5 threshold
    # contour_img    = (img_display * 255).astype(np.uint8)
    # cam_threshold  = (cam > 0.5).astype(np.uint8)
    # contours, _    = cv2.findContours(
    #     cam_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    # )
    # cv2.drawContours(contour_img, contours, -1, (255, 50, 50), 2)

    # Contour: find regions where CAM > 0.5 threshold
    contour_img    = (img_display * 255).astype(np.uint8)
    contour_img    = np.ascontiguousarray(contour_img)  # <-- THIS IS THE FIX
    
    cam_threshold  = (cam > 0.5).astype(np.uint8)
    contours, _    = cv2.findContours(
        cam_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(contour_img, contours, -1, (255, 50, 50), 2)

    # Determine if prediction is correct
    correct    = CLASS_NAMES[pred_idx] == class_name
    pred_color = 'green' if correct else 'red'
    pred_label = f"Pred: {CLASS_NAMES[pred_idx]} ({confidence:.0%})"
    icon       = '✓' if correct else '✗'

    # Column 0: original image
    axes[row][0].imshow(img_display)
    axes[row][0].set_ylabel(
        class_name, fontsize=11, fontweight='bold',
        rotation=90, va='center'
    )
    axes[row][0].axis('off')

    # Column 1: heatmap only
    axes[row][1].imshow(cam, cmap='jet')
    axes[row][1].axis('off')

    # Column 2: overlay with prediction label
    axes[row][2].imshow(overlay)
    axes[row][2].set_title(
        f'{icon} {pred_label}',
        color=pred_color, fontsize=9, pad=3
    )
    axes[row][2].axis('off')

    # Column 3: contour on original
    axes[row][3].imshow(contour_img)
    n_contours = len(contours)
    axes[row][3].set_title(
        f'{n_contours} region(s) detected',
        fontsize=9, pad=3
    )
    axes[row][3].axis('off')

    print(f"  {class_name:20s} → Predicted: {CLASS_NAMES[pred_idx]:20s} "
          f"({confidence:.0%}) {'✓' if correct else '✗'}")

plt.tight_layout()
plt.savefig('output/03_gradcam_visualization.jpg', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: output/03_gradcam_visualization.jpg")

# ============================================================
# PRINT INTERPRETATION GUIDE
# ============================================================
print("\n" + "="*55)
print("HOW TO INTERPRET GRAD-CAM RESULTS")
print("="*55)
print("""
✓ GOOD signs (model is learning correctly):
  - Heatmap concentrates on the actual defect region
  - Correct predictions with high confidence (>80%)
  - Contour region matches the visible defect

✗ WARNING signs (model may need more data):
  - Heatmap focuses on background or edges of image
  - High confidence but wrong class (overconfident errors)
  - Heatmap is diffuse/spread across entire image

WHAT TO DO if results look bad:
  1. Add more training data for the failing class
  2. Use stronger augmentation for that class
  3. Try full fine-tune instead of partial
  4. Check annotation quality in your dataset
""")
print("✓ Script 3 complete — Day 10 finished!")
print("  Commit your work: git add . && git commit -m 'day-10: transfer learning complete'")