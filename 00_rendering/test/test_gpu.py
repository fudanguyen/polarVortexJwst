import os

# Set debug flags for VTK/OpenGL
os.environ["VTK_DEBUG_OPENGL"] = "1"
os.environ["VTK_REPORT_OPENGL_ERRORS"] = "1"

import vtk
print("VTK Version:", vtk.vtkVersion.GetVTKVersion())
print("OpenGL2 Enabled:", hasattr(vtk, 'vtkOpenGLRenderWindow'))

# Now import PyVista/VTK
import pyvista as pv
from pyvista import examples

# Test with a plot
mesh = examples.load_airplane()
pl = pv.Plotter()
pl.add_mesh(mesh)
pl.show()
print("Rendering Backend:", pl.render_window.GetRenderingBackend(),'\n')
print(pl.render_window.ReportCapabilities())  # Look for GPU vendor/OpenGL version