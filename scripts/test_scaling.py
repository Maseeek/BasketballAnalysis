import unittest
import numpy as np

def calculate_scale_and_hoops(width, height, hoop_left, hoop_right, target_width=640):
    """Refactored logic from server.py for testing"""
    scale = target_width / width
    scaled_h_l = (int(hoop_left[0] * scale), int(hoop_left[1] * scale))
    scaled_h_r = (int(hoop_right[0] * scale), int(hoop_right[1] * scale))
    scaled_height = int(height * scale)
    return scale, target_width, scaled_height, scaled_h_l, scaled_h_r

class TestScaling(unittest.TestCase):
    def test_4k_to_640(self):
        # 3840x2160 -> 640x360
        scale, tw, th, hl, hr = calculate_scale_and_hoops(3840, 2160, (1000, 500), (2000, 500))
        self.assertAlmostEqual(scale, 640/3840)
        self.assertEqual(tw, 640)
        self.assertEqual(th, 360)
        self.assertEqual(hl, (int(1000 * 640/3840), int(500 * 640/3840)))
        
    def test_1080p_to_640(self):
        # 1920x1080 -> 640x360
        scale, tw, th, hl, hr = calculate_scale_and_hoops(1920, 1080, (100, 100), (200, 100))
        self.assertEqual(scale, 640/1920)
        self.assertEqual(tw, 640)
        self.assertEqual(th, 360) # Aspect ratio is 16:9
        self.assertEqual(hl, (33, 33)) # 100 / 3 roughly
        
    def test_vertical_video(self):
        # 1080x1920 -> 640x1137
        scale, tw, th, hl, hr = calculate_scale_and_hoops(1080, 1920, (100, 100), (200, 100))
        self.assertEqual(tw, 640)
        self.assertEqual(th, 1137) # 1920 * 640/1080 = 1137.77...
        
if __name__ == '__main__':
    unittest.main()
