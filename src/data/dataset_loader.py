"""Dataset loading and preprocessing for industrial anomaly detection."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    """MVTec AD compatible dataset loader supporting category subdirectories."""

    def __init__(self, root: str | Path, category: str = "bottle", split: str = "train", image_size: int = 224) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.samples: list[tuple[Path, int, str]] = []
        self._load_data()

    def _load_data(self) -> None:
        cat_split = self.root / self.category / self.split
        split_dir = cat_split if cat_split.exists() else (self.root / self.split)
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = 0 if class_dir.name == "good" else 1
            for img_path in sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, label, class_dir.name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, defect_type = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label, str(img_path), defect_type


def get_dataloader(
    root: str | Path,
    category: str = "bottle",
    split: str = "train",
    image_size: int = 224,
    batch_size: int = 8,
) -> DataLoader:
    dataset = MVTecDataset(root=root, category=category, split=split, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
