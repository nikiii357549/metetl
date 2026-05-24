
import unittest
from pathlib import Path
import sys
import json
import tempfile
import shutil
import asyncio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metetl.images.processing import ImageProcessor


class TestImageProcessor(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.mkdtemp()
        self.json_path = Path(self.temp_dir) / "test_ids.json"

        test_data = {"total_paintings": 2, "object_ids": ["436947", "437123"]}
        with open(self.json_path, 'w') as f:
            json.dump(test_data, f)

    def tearDown(self):
        """Удаляет временную папку после каждого теста"""
        shutil.rmtree(self.temp_dir)

    # ========== ЮНИТ-ТЕСТ: ЗАГРУЗКА ID ИЗ JSON ==========
    def test_load_ids_from_json(self):
        """Проверяет чтение ID из JSON файла"""
        processor = ImageProcessor(num_images=1)
        processor.load_ids_from_json(str(self.json_path))

        self.assertEqual(len(processor._painting_ids), 2)
        self.assertIn("436947", processor._painting_ids)
        self.assertIn("437123", processor._painting_ids)

        print("✅ test_load_ids_from_json: OK")

    # ========== ИНТЕГРАЦИОННЫЙ ТЕСТ: РЕАЛЬНОЕ СКАЧИВАНИЕ ==========
    async def _async_download_test(self):
        processor = ImageProcessor(num_images=1, output_folder=self.temp_dir)
        processor._painting_ids = ["436947"]
        await processor.download_paintings_async()
        return processor

    def test_download_one_image(self):
        """Интеграционный тест: реальное скачивание изображения через API"""
        processor = asyncio.run(self._async_download_test())

        self.assertEqual(len(processor._artworks), 1)

        artwork = processor._artworks[0][1]
        self.assertIsNotNone(artwork.image)
        self.assertTrue(artwork.image.size > 0)
        self.assertNotEqual(artwork.title, "Unknown")

        print("✅ test_download_one_image: OK")

    # ========== ИНТЕГРАЦИОННЫЙ ТЕСТ: ПОЛНАЯ ОБРАБОТКА ==========
    async def _async_processing_test(self):
        processor = ImageProcessor(num_images=1, output_folder=self.temp_dir)
        processor._painting_ids = ["436947"]
        await processor.download_paintings_async()
        processor.process_artworks_parallel()
        return processor

    def test_processing_downloaded_image(self):
        """Интеграционный тест: полная обработка скачанного изображения"""
        processor = asyncio.run(self._async_processing_test())

        self.assertEqual(len(processor._artworks), 1)

        output_dir = Path(processor.OUTPUT_FOLDER)
        files = list(output_dir.iterdir())
        self.assertGreater(len(files), 0)

        has_original = any('original' in f.name for f in files)
        has_gray = any('gray' in f.name for f in files)
        has_blur = any('blur' in f.name for f in files)
        has_sobel = any('sobel' in f.name for f in files)
        has_gamma = any('gamma' in f.name for f in files)
        has_hist = any('hist' in f.name for f in files)
        has_sharpen = any('sharpen' in f.name for f in files)
        has_mixed = any('mixed' in f.name for f in files)

        self.assertTrue(has_original, "Оригинал не создан")
        self.assertTrue(has_gray, "Чёрно-белая версия не создана")
        self.assertTrue(has_blur, "Размытая версия не создана")
        self.assertTrue(has_sobel, "Sobel (границы) не создан")
        self.assertTrue(has_gamma, "Гамма-коррекция не создана")
        self.assertTrue(has_hist, "Выравнивание гистограммы не создано")
        self.assertTrue(has_sharpen, "Резкость не создана")
        self.assertTrue(has_mixed, "Смешивание не создано")

        print("✅ test_processing_downloaded_image: OK")

    # ========== ИНТЕГРАЦИОННЫЙ ТЕСТ: API ==========
    async def _async_api_test(self):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = f"{ImageProcessor.API_ENDPOINT}/436947"
            async with session.get(url) as response:
                return await response.json()

    def test_api_call(self):
        """Интеграционный тест: реальный вызов API музея"""
        data = asyncio.run(self._async_api_test())

        self.assertIsNotNone(data)
        self.assertIn('objectID', data)
        self.assertEqual(data['objectID'], 436947)
        self.assertIn('primaryImageSmall', data)
        self.assertIn('title', data)

        print("✅ test_api_call: OK")


if __name__ == "__main__":
    unittest.main()