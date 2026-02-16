import open3d as o3d
import numpy as np

mesh_path = "../data/replica/apartment_0/mesh.ply"

mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()

pcd = mesh.sample_points_uniformly(number_of_points=150000)

print("Número de pontos:", np.asarray(pcd.points).shape)

o3d.visualization.draw_geometries([pcd])
