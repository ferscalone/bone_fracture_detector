# Bone Fracture Detector

Детектор переломов костей на рентгеновских снимках. Загружаешь снимок —
приложение обводит места переломов рамками и подписывает, в какой части тела
обнаружен перелом (локоть, пальцы, предплечье, плечевая кость, плечо или запястье).

В основе — модель Faster R-CNN (ResNet50-FPN) из torchvision, дообученная на
[bone fracture detection dataset](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project).

## Состав

- `src/` — исходники модели, инференса и веб-интерфейса.
- `notebooks/train_colab.ipynb` — ноутбук обучения в Google Colab.
- `docs/` — `user_guide.docx`, `developer_guide.docx`, `отчёт.docx`.
- `Dockerfile`, `docker-compose.yml` — для запуска одной командой.

## Требования

- Docker Engine ≥ 24.0 и Docker Compose ≥ 2.0.
- 8 ГБ оперативной памяти.
- 5 ГБ свободного места на диске.

## Быстрый запуск

```bash
cd bone_fracture_detector.

docker compose up --build
```

После запуска интерфейс будет доступен на http://localhost:7860.

Если ссылку на веса не задавать, нужно положить готовый файл вручную в
`models/model.pth` — его можно получить, запустив ноутбук обучения в Colab.

## Подробные инструкции

- Установка и работа с интерфейсом — `docs/user_guide.docx`.
- Архитектура, обучение, расширение — `docs/developer_guide.docx`.
- Полный отчёт по НИРС — `docs/отчёт.docx`.