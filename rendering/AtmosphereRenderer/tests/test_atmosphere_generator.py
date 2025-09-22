import unittest
from src.rendering.atmosphere_generator import AtmosphereGenerator

class TestAtmosphereGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = AtmosphereGenerator()

    def test_initialization(self):
        self.assertIsNotNone(self.generator)

    def test_create_spherical_mesh(self):
        mesh = self.generator.create_spherical_mesh()
        self.assertEqual(mesh.shape[0], 200)  # Check if mesh has expected shape
        self.assertEqual(mesh.shape[1], 200)

    def test_polar_change(self):
        result = self.generator.polar_change(value=0.5, amplitude=0.25, t=30, f=0, RP=60)
        self.assertAlmostEqual(result, 0.5 + 0.25 * np.sin(2 * np.pi / 60 * 30))

    def test_limb_darkening(self):
        arr = np.ones((100, 100))
        mask = self.generator.limb_darkening(arr, u_coefficient=0.75)
        self.assertEqual(mask.shape, arr.shape)

    def test_circle_vortice(self):
        recmap = np.zeros((200, 200))
        coord = [[-30, -20], [-40, -50]]
        updated_map = self.generator.circle_vortice(recmap, coord, t=0, number=5)
        self.assertTrue(np.any(updated_map > 0))  # Check if any values were updated

if __name__ == '__main__':
    unittest.main()