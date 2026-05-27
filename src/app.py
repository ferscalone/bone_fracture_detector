"""
Веб-интерфейс на Gradio для детектора переломов.

Запуск:
    python -m src.app
или через Docker — см. docker-compose.yml.
"""

import os
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont

from .model import load_model
from .predict import predict_image


CLASS_COLORS = {
    "локоть":          (255,  99, 132),  # розовый
    "пальцы":          ( 54, 162, 235),  # синий
    "предплечье":      (255, 206,  86),  # жёлтый
    "плечевая кость":  ( 75, 192, 192),  # бирюзовый
    "плечо":           (153, 102, 255),  # фиолетовый
    "запястье":        (255, 159,  64),  # оранжевый
}
DEFAULT_COLOR = (200, 200, 200)


_MODEL = None
_DEVICE = None


def _init_model():

    global _MODEL, _DEVICE
    if _MODEL is not None:
        return

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[app] Загружаю модель на устройство: {_DEVICE}")

    weights_path = Path(os.getenv("MODEL_PATH", "models/model.pth"))
    _MODEL = load_model(weights_path, device=_DEVICE)
    print(f"[app] Модель загружена из {weights_path}")


def _draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """
    Рисует рамки и подписи на копии изображения и возвращает её.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    # Толщина рамки зависит от размера снимка
    line_width = max(2, min(out.width, out.height) // 200)

    # Пытаемся подгрузить TTF-шрифт. Если не получилось — fallback на дефолт.
    try:
        font_size = max(14, min(out.width, out.height) // 40)
        font = None
        for candidate in ("DejaVuSans.ttf", "arial.ttf"):
            try:
                font = ImageFont.truetype(candidate, font_size)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["label"], DEFAULT_COLOR)

        # Сама рамка.
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Подпись над рамкой: класс и уверенность в процентах.
        caption = f"{det['label']} {det['score'] * 100:.0f}%"

        # Прямоугольная "плашка" под текстом
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        padding = 2
        # Если рамка близко к верху — пишем подпись внутри рамки, а не над ней.
        text_y = y1 - text_h - 2 * padding
        if text_y < 0:
            text_y = y1 + padding

        draw.rectangle(
            [x1, text_y, x1 + text_w + 2 * padding, text_y + text_h + 2 * padding],
            fill=color,
        )
        draw.text((x1 + padding, text_y + padding), caption, fill=(255, 255, 255), font=font)

    return out


def _format_text_report(detections: list[dict]) -> str:
    """Текстовый отчёт о найденных переломах"""
    if not detections:
        return (
            "Переломы не обнаружены.\n\n"
            "Это не означает, что их точно нет — модель — лишь вспомогательный "
            "инструмент, окончательное заключение даёт врач."
        )

    # Группируем найденное по классам
    by_class: dict[str, list[float]] = {}
    for d in detections:
        by_class.setdefault(d["label"], []).append(d["score"])

    lines = [f"Найдено объектов: {len(detections)}.", ""]
    for label, scores in by_class.items():
        scores_str = ", ".join(f"{s * 100:.0f}%" for s in sorted(scores, reverse=True))
        lines.append(f"• {label} — {len(scores)} шт. (уверенность: {scores_str})")

    lines.append("")
    return "\n".join(lines)


def _classify(image: Image.Image, threshold: float):
    """Обработчик кнопки 'Анализировать'."""
    if image is None:
        return None, "Сначала загрузите рентгеновский снимок."

    _init_model()

    try:
        detections = predict_image(
            _MODEL, image, device=_DEVICE, confidence_threshold=float(threshold)
        )
    except Exception as e:
        # Ошибки модели не должны валить весь UI.
        return None, f"Ошибка при анализе снимка: {e}"

    annotated = _draw_detections(image, detections)
    report = _format_text_report(detections)
    return annotated, report


def create_ui() -> gr.Blocks:
    """Собирает интерфейс Gradio."""
    with gr.Blocks(title="Детектор переломов костей", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Детектор переломов костей на рентгеновских снимках\n"
            "Загрузите рентгеновский снимок — приложение покажет, где и какой "
            "перелом обнаружен (локоть, пальцы, предплечье, плечевая кость, плечо или запястье)."
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Рентгеновский снимок",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
                    step=0.05,
                    label="Порог уверенности",
                    info="Рамки с уверенностью ниже этого значения не отображаются.",
                )
                analyze_btn = gr.Button("Анализировать", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Результат", type="pil")
                output_text = gr.Textbox(
                    label="Описание",
                    lines=10,
                    interactive=False,
                )

        analyze_btn.click(
            fn=_classify,
            inputs=[input_image, threshold],
            outputs=[output_image, output_text],
        )


    return demo


def main():
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo = create_ui()
    # server_name="0.0.0.0" — чтобы интерфейс был доступен изнутри Docker-контейнера.
    demo.launch(server_name="0.0.0.0", server_port=port, show_api=False)


if __name__ == "__main__":
    main()
