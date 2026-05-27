"""
Преобразования и аугментации для рентгеновских снимков.

torchvision.transforms.v2 умеет работать с (image, target) одновременно,
так что рамки автоматически пересчитываются при отражении/обрезке.
"""

import torch
from torchvision.transforms import v2


def get_train_transforms():

    return v2.Compose([
        v2.ToImage(),  # PIL -> tv_tensors.Image
        v2.RandomHorizontalFlip(p=0.5),  # отражение влево-вправо
        v2.ToDtype(torch.float32, scale=True),  # 0..255 -> 0..1
    ])


def get_eval_transforms():
    """Преобразования для инференса и валидации — без аугментаций."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


class ApplyTransforms:
    """
    Обёртка, которая применяет преобразования к (image, target) одновременно.
    Нужна, чтобы рамки автоматически пересчитывались при отражении.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        # Чтобы v2 знал, что это bounding box'ы, заворачиваем их в tv_tensors.BoundingBoxes.
        from torchvision import tv_tensors

        boxes = tv_tensors.BoundingBoxes(
            target["boxes"],
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(image.height, image.width),
        )
        target = dict(target)
        target["boxes"] = boxes

        image, target = self.transforms(image, target)

        # Обратно в обычный тензор — Faster R-CNN ждёт именно torch.Tensor.
        target["boxes"] = target["boxes"].as_subclass(torch.Tensor)
        return image, target
