import asyncio
import sys
from metetl.cli import create_parser
from metetl.logging_config import logger
from metetl.analysis.data_to_download import prepare_download_json
from metetl.analysis.aggregations import analyze_dataset
from metetl.images.processing import run_processing_pipeline

def main():
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            logger.info(f"Подготовка: {args.csv} -> {args.output}")
            prepare_download_json(args.csv, args.output)
            logger.info("Готово")
        elif args.command == "process":
            logger.info(f"Обработка: {args.input} -> {args.output}, num={args.num}")
            asyncio.run(run_processing_pipeline(args.input, args.output, args.num, args.parallel))
            logger.info("Готово")
        elif args.command == "analyze":
            logger.info(f"Анализ: {args.csv} -> {args.output_dir}")
            analyze_dataset(args.csv, args.output_dir)
            logger.info("Готово")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()