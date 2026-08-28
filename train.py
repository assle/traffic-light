import os
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    BASE = Path(r'c:\Users\Di Wang\Desktop\已筛选数据')
    DATASET = BASE / 'dataset'
    MODEL_PATH = BASE / 'yolo26x-cls.pt'

    # Count classes
    classes = sorted([d.name for d in DATASET.iterdir() if d.is_dir()])
    print(f'Classes ({len(classes)}): {classes}')
    for c in classes:
        n = len(list((DATASET / c).glob('*')))
        print(f'  {c}: {n} images')

    # Load pretrained model
    model = YOLO(str(MODEL_PATH))

    # Train
    results = model.train(
        data=str(DATASET),
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        dropout=0.2,
        val=True,
        save=True,
        save_period=10,
        project=str(BASE / 'runs'),
        name='cls_train',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        amp=True,
        device=0,
        workers=4,
        verbose=True,
    )

    print('\nTraining complete!')
    print(f'Best model: {results.save_dir}')
