"""
Инференс модели на одном изображении.

Здесь только "чистая" логика — без UI и без сети.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from .model import CLASS_NAMES_RU


# Те же трансформации, что и в transforms.py, но завернуты как обычная функция.
# Дублирую, чтобы инференс не тянул лишних зависимостей.
_inference_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: str = "cpu",
    confidence_threshold: float = 0.5,
):
    """
    Применяет модель к одному PIL-изображению.

    Параметры
    ---------
    model : torch.nn.Module
        Обученная Faster R-CNN.
    image : PIL.Image
        Входной рентгеновский снимок.
    device : str
        "cpu" или "cuda".
    confidence_threshold : float
        Рамки с уверенностью ниже этого значения отбрасываются.

    Возвращает
    ----------
    list[dict] — список найденных переломов. Каждый элемент:
        {"box": [xmin, ymin, xmax, ymax], "label": "локоть", "score": 0.92}
    """
    # На всякий случай переводим в RGB (модель ждёт три канала).
    image = image.convert("RGB")

    # Готовим тензор для модели.
    tensor = _inference_transform(image)
    tensor = tensor.to(device)

    # torch.inference_mode — быстрее, чем no_grad, и явно говорит, что мы не учимся.
    with torch.inference_mode():
        # Модель принимает список тензоров (так задумано torchvision).
        predictions = model([tensor])

    # У нас одно изображение -> один результат.
    pred = predictions[0]

    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score < confidence_threshold:
            continue

        # Защита от случая, если модель вдруг вернёт неизвестный индекс.
        label_name = (
            CLASS_NAMES_RU[label]
            if 0 <= label < len(CLASS_NAMES_RU)
            else f"класс_{label}"
        )

        detections.append({
            "box": [float(b) for b in box],
            "label": label_name,
            "score": float(score),
        })

    return detections


def main():
    """
    Консольный режим: python -m src.predict <путь_до_картинки>.
    """
    import argparse

    from .model import load_model

    parser = argparse.ArgumentParser(description="Детекция переломов на одном снимке.")
    parser.add_argument("image_path", type=str, help="Путь к рентгеновскому снимку.")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/model.pth",
        help="Путь к файлу с весами модели.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Минимальная уверенность для отображения рамки.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Использую устройство: {device}")

    model = load_model(args.weights, device=device)
    image = Image.open(Path(args.image_path))

    detections = predict_image(model, image, device=device, confidence_threshold=args.threshold)

    if not detections:
        print("Переломов не обнаружено (или уверенность ниже порога).")
        return

    print(f"Найдено переломов: {len(detections)}")
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["box"]
        print(
            f"  {i}. {det['label']}: уверенность {det['score']:.2f}, "
            f"рамка [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
        )


if __name__ == "__main__":
    main()
