"""
Локальный скрипт обучения. Дублирует логику из notebooks/train_colab.ipynb,
но без визуализаций и без скачивания датасета с Kaggle

Запуск:
    python -m src.train \
        --data data/bone_fracture \
        --epochs 15 \
        --batch_size 4 \
        --lr 0.005
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import BoneFractureDataset, collate_fn
from .model import NUM_CLASSES, build_model
from .transforms import ApplyTransforms, get_eval_transforms, get_train_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение детектора переломов.")
    parser.add_argument("--data", type=str, required=True, help="Путь к корню датасета.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=str,
        default="models/model.pth",
        help="Куда сохранить лучшие веса.",
    )
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device, epoch: int):
    """
    Один проход по обучающей выборке.
    Faster R-CNN в режиме train() возвращает словарь с разными loss-ами
    (classification, box regression, objectness, rpn), мы их просто складываем.
    """
    model.train()
    total_loss = 0.0
    for step, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 20 == 0:
            print(f"  [epoch {epoch}] step {step}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.inference_mode()
def evaluate_loss(model, loader, device):
    """
    Считает loss на валидации.
    Faster R-CNN не возвращает loss в режиме eval, поэтому временно ставим train(),
    но градиенты не считаем — это безопасно и стандартный приём для валидации
    детекторов.
    """
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()

    return total_loss / len(loader)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Использую устройство: {device}")

    # Сборка датасетов.
    train_ds = BoneFractureDataset(
        args.data, split="train", transforms=ApplyTransforms(get_train_transforms())
    )
    valid_ds = BoneFractureDataset(
        args.data, split="valid", transforms=ApplyTransforms(get_eval_transforms())
    )
    print(f"Train: {len(train_ds)} образцов, Valid: {len(valid_ds)} образцов")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Модель и оптимизатор.
    model = build_model(num_classes=NUM_CLASSES, pretrained=True)
    model.to(device)

    # Оптимизируем только параметры, у которых requires_grad=True.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Lr-scheduler: уменьшаем lr в 10 раз каждые 5 эпох — стандартная практика.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate_loss(model, valid_loader, device)
        lr_scheduler.step()

        print(f"[epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"  -> сохранил лучшие веса в {output_path}")

    print(f"Готово. Лучший val_loss = {best_val_loss:.4f}")


if __name__ == "__main__":
    main()