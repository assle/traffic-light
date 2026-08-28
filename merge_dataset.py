import os
import shutil
from pathlib import Path

BASE = Path(r'c:\Users\Di Wang\Desktop\已筛选数据')
SOURCES = {
    'a': BASE / 'a系数据(已筛选)',
    'b': BASE / 'b系数据(已筛选)',
}
DST = BASE / 'dataset'

# Clear existing dataset
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir()

stats = {}
for prefix, src_root in SOURCES.items():
    for class_dir in sorted(src_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        dst_class = DST / class_name
        dst_class.mkdir(exist_ok=True)

        count = 0
        for img in sorted(class_dir.iterdir()):
            if img.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                new_name = f'{prefix}_{img.name}'
                shutil.copy2(img, dst_class / new_name)
                count += 1

        key = f'{prefix}_{class_name}'
        stats[key] = count
        print(f'[{key}] copied {count} images')

print('\n--- Summary ---')
total = 0
for k, v in sorted(stats.items()):
    print(f'  {k}: {v}')
    total += v
print(f'Total: {total} images')

# Count per class
print('\n--- Per class ---')
class_totals = {}
for prefix, src_root in SOURCES.items():
    for class_dir in sorted(src_root.iterdir()):
        if not class_dir.is_dir():
            continue
        cn = class_dir.name
        class_totals[cn] = class_totals.get(cn, 0) + len([f for f in DST.glob(f'{cn}/*')])

for cn, cnt in sorted(class_totals.items()):
    print(f'  {cn}: {cnt}')
