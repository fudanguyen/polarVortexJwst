import unittest
from src.rendering.mesh_generator import create_spherical_mesh

class TestMeshGenerator(unittest.TestCase):

    def test_create_spherical_mesh(self):
        # Test the creation of a spherical mesh
        radius = 1
        phi_points = 200
        theta_points = 200
        
        x, y, z = create_spherical_mesh(radius, phi_points, theta_points)
        
        # Check the shape of the output arrays
        self.assertEqual(x.shape, (phi_points, theta_points))
        self.assertEqual(y.shape, (phi_points, theta_points))
        self.assertEqual(z.shape, (phi_points, theta_points))
        
        # Check that the mesh is centered around the origin
        self.assertAlmostEqual(x.mean(), 0, delta=1e-5)
        self.assertAlmostEqual(y.mean(), 0, delta=1e-5)
        self.assertAlmostEqual(z.mean(), 0, delta=1e-5)

    def test_mesh_boundary(self):
        # Test that the mesh points are within the expected boundaries
        radius = 1
        phi_points = 200
        theta_points = 200
        
        x, y, z = create_spherical_mesh(radius, phi_points, theta_points)
        
        # Check that all points are within the sphere of given radius
        distances = (x**2 + y**2 + z**2)**0.5
        self.assertTrue((distances <= radius).all())

if __name__ == '__main__':
    unittest.main()