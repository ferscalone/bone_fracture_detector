"""
Модель для детекции переломов костей.

Используется Faster R-CNN с backbone ResNet50 + FPN из torchvision.
Базовая модель предобучена на датасете COCO; меняется только финальный
классификационный слой под наши 7 классов (6 частей тела + фон).
"""

from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Шесть анатомических областей + фон. Индекс 0 в Faster R-CNN всегда
# зарезервирован под фон, поэтому собственно классов переломов 6,
# а на вход модели подаётся num_classes = 7.
CLASS_NAMES_RU = [
    "фон",             # 0 — служебный класс
    "локоть",          # 1
    "пальцы",          # 2
    "предплечье",      # 3
    "плечевая кость",  # 4
    "плечо",           # 5
    "запястье",        # 6
]

# Сопоставление "название класса в YOLO-датасете" → "индекс в модели".
# Датасет на Kaggle (pkdarabi/bone-fracture-detection-computer-vision-project)
# на самом деле содержит 7 классов: и "humerus", и "humerus fracture"
# идут отдельно. С точки зрения анатомии это одна и та же кость, поэтому
# мы объединяем их в один класс "плечевая кость" (индекс 4).
YOLO_NAME_TO_MODEL_INDEX = {
    "elbow positive":    1,  # локоть
    "fingers positive":  2,  # пальцы
    "forearm fracture":  3,  # предплечье
    "humerus fracture":  4,  # плечевая кость
    "humerus":           4,  # плечевая кость (та же кость, что и выше)
    "shoulder fracture": 5,  # плечо
    "wrist positive":    6,  # запястье
}

NUM_CLASSES = len(CLASS_NAMES_RU)  # 7 = 6 классов + фон


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> torch.nn.Module:
    """
    Собирает Faster R-CNN с заменённой "головой" под нужное число классов.

    Параметры
    ---------
    num_classes : int
        Общее число классов вместе с фоном.
    pretrained : bool
        Загружать ли веса, предобученные на COCO. При обучении в Colab
        ставится True, при инференсе из чекпойнта — можно поставить False,
        потому что свои веса всё равно перезапишут предобученные.

    Возвращает
    ----------
    torch.nn.Module — готовая модель Faster R-CNN.
    """
    # Загружаем базовую модель. Параметр weights="DEFAULT" подгружает
    # последние доступные предобученные веса COCO.
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # У стандартной модели последний слой рассчитан на 91 класс (COCO).
    # Заменяем его на свой, под наши классы.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_model(weights_path: str | Path, device: str = "cpu") -> torch.nn.Module:
    """
    Загружает обученные веса в модель и переводит её в режим eval.

    Параметры
    ---------
    weights_path : str | Path
        Путь к файлу .pth с весами, обученными в Colab.
    device : str
        "cpu" или "cuda" — куда положить модель.

    Возвращает
    ----------
    torch.nn.Module — модель, готовая к инференсу.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Не найден файл с весами модели: {weights_path}. "
            f"Запусти scripts/download_weights.py или положи файл вручную."
        )

    # pretrained=False — COCO-веса нам не нужны, мы тут же загрузим свои.
    model = build_model(num_classes=NUM_CLASSES, pretrained=False)

    # map_location позволяет загрузить веса с GPU на CPU и наоборот.
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()  # отключает dropout и batchnorm в train-режиме
    return model