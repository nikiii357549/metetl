"""
Юнит-тесты для класса Artwork
Проверяют методы преобразования изображений
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metetl.images.models import Artwork


class TestArtwork(unittest.TestCase):
    """Тестирование класса Artwork"""

    def setUp(self):
        """Создаёт тестовые данные перед каждым тестом"""
        self.color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.metadata = {
            "objectID": "12345",
            "title": "Test Painting",
            "artistDisplayName": "Test Artist"
        }
        self.artwork = Artwork(image=self.color_image, metadata=self.metadata)

    def test_grayscale_conversion(self):
        """Проверяет преобразование в оттенки серого"""
        gray = self.artwork.convert_to_grayscale()

        self.assertEqual(len(gray.image.shape), 2)
        self.assertEqual(gray.image.shape[0], 100)
        self.assertEqual(gray.image.shape[1], 100)
        self.assertEqual(gray.image.dtype, np.uint8)
        self.assertEqual(gray.object_id, "12345")

        print("✅ test_grayscale_conversion: OK")

    def test_is_color_property(self):
        """Проверяет свойство is_color"""
        self.assertTrue(self.artwork.is_color)

        gray = self.artwork.convert_to_grayscale()
        self.assertFalse(gray.is_color)

        print("✅ test_is_color_property: OK")

    def test_metadata_extraction(self):
        """Проверяет извлечение метаданных"""
        self.assertEqual(self.artwork.object_id, "12345")
        self.assertEqual(self.artwork.title, "Test Painting")
        self.assertEqual(self.artwork.artist, "Test Artist")

        print("✅ test_metadata_extraction: OK")

    def test_apply_filter(self):
        """Проверяет применение фильтра (свёртку)"""
        kernel = np.ones((3, 3)) / 9
        filtered = self.artwork.apply_filter(kernel)

        self.assertEqual(filtered.image.shape, self.color_image.shape)

        print("✅ test_apply_filter: OK")

    def test_gaussian_blur(self):
        """Проверяет размытие по Гауссу"""
        blurred = self.artwork.apply_gaussian_blur(size=5, sigma=1.0)

        self.assertEqual(blurred.image.shape, self.color_image.shape)

        print("✅ test_gaussian_blur: OK")

    def test_sobel(self):
        """Проверяет выделение границ алгоритмом Собеля"""
        sobel = self.artwork.apply_sobel()

        self.assertEqual(len(sobel.image.shape), 2)

        print("✅ test_sobel: OK")

    def test_gamma_correction(self):
        """Проверяет гамма-коррекцию"""
        gamma = self.artwork.gamma_correction(gamma=1.5)

        self.assertEqual(gamma.image.shape, self.color_image.shape)

        print("✅ test_gamma_correction: OK")

    def test_histogram_equalization(self):
        """Проверяет выравнивание гистограммы"""
        hist_eq = self.artwork.histogram_equalization()

        self.assertEqual(hist_eq.image.shape, self.color_image.shape)

        print("✅ test_histogram_equalization: OK")

    def test_mixed(self):
        """Проверяет смешивание двух изображений"""
        other_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        other_artwork = Artwork(image=other_image, metadata={})

        mixed = self.artwork + other_artwork

        self.assertEqual(mixed.image.shape, self.color_image.shape)

        print("✅ test_mixed: OK")


if __name__ == "__main__":
    unittest.main()