import json
import logging.config
from pathlib import Path


def _setup_logging(config_file: str = "logging_config.json"):
    # Ищем файл в той же папке, где лежит logging_config.py
    config_path = Path(__file__).parent / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации {config_file} не найден в {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Создаём директорию для логов (относительно текущей рабочей директории)
    log_file = config.get('handlers', {}).get('file', {}).get('filename', 'logs/metetl.log')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)
    return logging.getLogger("metetl")


logger = _setup_logging()