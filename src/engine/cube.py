import numpy as np

class Cube:
    def __init__(self, size=0.5):
        s = size
        self.base_vertices = np.array([
            [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
            [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        ], dtype=float)

        self.edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]

        self.faces = [
            (0,1,2,3),
            (4,5,6,7),
            (0,1,5,4),
            (2,3,7,6),
            (0,3,7,4),
            (1,2,6,5),
        ]

        self.face_colors = [
            (0.8, 0.2, 0.2),
            (0.2, 0.8, 0.2),
            (0.2, 0.2, 0.8),
            (0.8, 0.8, 0.2),
            (0.8, 0.2, 0.8),
            (0.2, 0.8, 0.8),
        ]

        self.position  = np.array([0.0, 0.0, 0.0])
        self.rotation  = np.array([0.0, 0.0, 0.0])
        self.scale     = 1.0

    def get_transform_matrix(self):
        rx, ry, rz = np.radians(self.rotation)
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        return Rz @ Ry @ Rx

    def get_transformed_vertices(self):
        R = self.get_transform_matrix()
        verts = (self.base_vertices * self.scale) @ R.T
        return verts + self.position