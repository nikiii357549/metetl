import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        prog="metetl",
        description="MET ETL: загрузка и обработка изображений"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Подготовка JSON")
    prepare_parser.add_argument("--csv", required=True)
    prepare_parser.add_argument("--output", required=True)

    process_parser = subparsers.add_parser("process", help="Запуск обработки")
    process_parser.add_argument("--input", required=True)
    process_parser.add_argument("--output", required=True)
    process_parser.add_argument("--num", type=int, default=5)
    process_parser.add_argument("--parallel", type=int, default=4)

    analyze_parser = subparsers.add_parser("analyze", help="Анализ датасета")
    analyze_parser.add_argument("--csv", required=True)
    analyze_parser.add_argument("--output-dir", required=True)

    return parser