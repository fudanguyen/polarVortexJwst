import unittest
from src.rendering.visualization import Visualization

class TestVisualization(unittest.TestCase):

    def setUp(self):
        self.visualization = Visualization()

    def test_render_frame(self):
        # Assuming render_frame returns an image or similar output
        frame_output = self.visualization.render_frame(some_input_data)
        self.assertIsNotNone(frame_output)
        self.assertIsInstance(frame_output, np.ndarray)  # Check if output is a numpy array

    def test_save_image(self):
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Dummy image
        save_path = "test_output.png"
        self.visualization.save_image(test_image, save_path)
        
        # Check if the file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Clean up
        os.remove(save_path)

    def test_visualization_with_invalid_data(self):
        with self.assertRaises(ValueError):
            self.visualization.render_frame(None)  # Test with invalid input

if __name__ == '__main__':
    unittest.main()