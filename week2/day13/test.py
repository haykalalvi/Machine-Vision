from anomalib.data import MVTec
from pathlib import Path

datamodule = MVTec(
    root='/Users/alvi/OpenCV/data',
    category='leather',
    train_batch_size=8,
    eval_batch_size=8,
)
datamodule.setup()
print(f"Train samples: {len(datamodule.train_data)}")
print(f"Test samples:  {len(datamodule.test_data)}")