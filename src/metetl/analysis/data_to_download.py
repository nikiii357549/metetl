import csv
import json
from pathlib import Path
from metetl.logging_config import logger

def prepare_download_json(csv_path: str, output_json: str) -> None:
    csv_path = Path(csv_path)
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    painting_ids = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Classification') == 'Paintings':
                painting_ids.append(row['Object ID'])

    logger.info(f"Найдено картин: {len(painting_ids)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"total_paintings": len(painting_ids), "object_ids": painting_ids}, f, indent=2)

    logger.info(f"Сохранено в {output_path}")