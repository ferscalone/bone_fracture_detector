"""
Загрузка датасета bone-fracture-detection в формате YOLO.

Для Faster R-CNN нужен формат "pascal-style":
    boxes: тензор [N, 4] с координатами [xmin, ymin, xmax, ymax] в пикселях
    labels: тензор [N] с индексами классов (от 1 до 6, 0 — это фон)
"""

from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from .model import YOLO_NAME_TO_MODEL_INDEX, NUM_CLASSES


def _load_yolo_class_mapping(data_yaml_path: Path) -> dict[int, int]:
    """
    Читает data.yaml, выясняет, какие имена классов и в каком порядке идут
    в YOLO-разметке, и строит словарь {yolo_index: model_index}.
    """
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Не нашёл {data_yaml_path}. Без data.yaml не могу понять, "
            f"как соотнести индексы классов в .txt-файлах с моделью."
        )

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config.get("names")
    if names is None:
        raise ValueError(f"В {data_yaml_path} нет ключа 'names'.")

    # В YOLO-формате names может быть либо списком (старый стиль),
    # либо словарём {0: 'name', 1: 'name'} (новый стиль ultralytics).
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]

    mapping: dict[int, int] = {}
    unknown: list[str] = []
    for yolo_idx, name in enumerate(names):
        key = str(name).strip().lower()
        if key in YOLO_NAME_TO_MODEL_INDEX:
            mapping[yolo_idx] = YOLO_NAME_TO_MODEL_INDEX[key]
        else:
            unknown.append(f"{yolo_idx}: {name}")

    if not mapping:
        raise ValueError(
            f"Ни одно имя класса из {data_yaml_path} не сопоставилось "
            f"со списком ожидаемых классов в model.py. "
            f"Имена в датасете: {names}"
        )

    if unknown:
        print(
            f"[dataset] Внимание: эти классы из {data_yaml_path} не сопоставлены "
            f"и будут проигнорированы в разметке: {unknown}"
        )

    return mapping


class BoneFractureDataset(Dataset):
    """Датасет в формате YOLO."""

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms=None,
        skip_empty: bool = True,
    ):
        """
        Параметры
        ---------
        root : str | Path
            Путь к корневой папке датасета (там, где лежит data.yaml).
        split : str
            "train", "valid" или "test".
        transforms : callable | None
            Аугментации/преобразования. На вход берут (image, target),
            на выходе тоже (image, target).
        skip_empty : bool
            Пропускать ли изображения без размеченных рамок. Faster R-CNN
            не любит пустые таргеты при обучении и иногда падает с
            device-side assert. Поэтому для train/valid лучше True.
        """
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Не нашёл папку с изображениями: {self.images_dir}. "
                f"Проверь, что датасет распакован правильно."
            )

        # Маппинг yolo_index → model_index строится автоматически по data.yaml.
        self.yolo_to_model = _load_yolo_class_mapping(self.root / "data.yaml")

        # Сначала собираем все изображения, затем (при skip_empty=True)
        # выкидываем те, у которых нет ни одной валидной рамки.
        all_images = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS
        )

        if skip_empty:
            self.image_paths = [p for p in all_images if self._has_valid_labels(p)]
            skipped = len(all_images) - len(self.image_paths)
            if skipped:
                print(
                    f"[dataset/{split}] Пропущено {skipped} изображений без "
                    f"размеченных переломов (Faster R-CNN не учится на пустых таргетах)."
                )
        else:
            self.image_paths = all_images

    def _label_path(self, img_path: Path) -> Path:
        return self.labels_dir / (img_path.stem + ".txt")

    def _has_valid_labels(self, img_path: Path) -> bool:
        """Быстрая проверка: есть ли в label-файле хотя бы одна валидная рамка."""
        label_path = self._label_path(img_path)
        if not label_path.exists() or label_path.stat().st_size == 0:
            return False
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id not in self.yolo_to_model:
                        continue
                    _, _, w, h = map(float, parts[1:5])
                    if w > 0 and h > 0:
                        return True
        except (ValueError, OSError):
            return False
        return False

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_labels(self, label_path: Path, img_w: int, img_h: int):
        """Читает YOLO-разметку и переводит её в формат Faster R-CNN."""
        boxes = []
        labels = []

        if not label_path.exists():
            return boxes, labels

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                # Игнорируем неизвестные классы — предупреждение уже выдали
                # в _load_yolo_class_mapping.
                if cls_id not in self.yolo_to_model:
                    continue
                model_label = self.yolo_to_model[cls_id]
                # Защита от переполнения — если модель собрана с меньшим
                # числом классов, чем встретилось в датасете.
                if not (0 < model_label < NUM_CLASSES):
                    continue

                cx, cy, w, h = map(float, parts[1:5])

                # YOLO → XYXY в пикселях.
                xmin = (cx - w / 2) * img_w
                ymin = (cy - h / 2) * img_h
                xmax = (cx + w / 2) * img_w
                ymax = (cy + h / 2) * img_h

                # Обрезаем рамку по краям изображения.
                xmin = max(0.0, min(xmin, img_w - 1))
                ymin = max(0.0, min(ymin, img_h - 1))
                xmax = max(0.0, min(xmax, img_w - 1))
                ymax = max(0.0, min(ymax, img_h - 1))

                # Пропускаем вырожденные рамки.
                if xmax - xmin < 1.0 or ymax - ymin < 1.0:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(model_label)

        return boxes, labels

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        boxes, labels = self._read_labels(self._label_path(img_path), image.width, image.height)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    """
    Свой collate для DataLoader.
    В батче картинки могут быть разного размера, поэтому tuple() лучше,
    чем torch.stack, который требует одинаковых форм.
    """
    return tuple(zip(*batch))
