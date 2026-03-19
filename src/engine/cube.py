import numpy as np

class Cube:
    def __init__(self, size=0.5):
        s = size
        self.base_vertices = np.array([
            [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
            [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        ], dtype=float)

        self.faces = [
            (0,1,2,3),
            (4,5,6,7), 
            (0,1,5,4),   
            (2,3,7,6),  
            (0,3,7,4), 
            (1,2,6,5),
        ]

        self.face_normals = np.array([
            [ 0,  0, -1],
            [ 0,  0,  1],
            [ 0, -1,  0],
            [ 0,  1,  0],
            [-1,  0,  0],
            [ 1,  0,  0],
        ], dtype=float)

        self.face_colors = [
            (0.8, 0.2, 0.2),
            (0.2, 0.8, 0.2),
            (0.2, 0.2, 0.8),
            (0.8, 0.8, 0.2),
            (0.8, 0.2, 0.8),
            (0.2, 0.8, 0.8),
        ]

        self.edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]

        self.position  = np.array([0.0, 0.0, 0.0])
        self.rotation  = np.array([0.0, 0.0, 0.0])
        self.scale     = 1.0

        self.selected_face = None 
        self.extruding     = False 

        self.vertex_offsets = np.zeros_like(self.base_vertices)

    def get_transform_matrix(self):
        rx, ry, rz = np.radians(self.rotation)
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        return Rz @ Ry @ Rx

    def get_transformed_vertices(self):
        verts = (self.base_vertices + self.vertex_offsets) * self.scale
        R = self.get_transform_matrix()
        return verts @ R.T + self.position

    def get_face_center(self, face_idx):
        """World-space center of a face."""
        verts = self.get_transformed_vertices()
        face  = self.faces[face_idx]
        return np.mean(verts[list(face)], axis=0)

    def get_face_world_normal(self, face_idx):
        """Face normal rotated into world space."""
        R = self.get_transform_matrix()
        return R @ self.face_normals[face_idx]

    def select_nearest_face(self, finger_world_pos):
        """
        Pick the face whose center is closest to the finger tip,
        but only if the finger is on the correct side (facing the normal).
        Returns face index or None.
        """
        best_idx  = None
        best_dist = 0.6

        for i in range(len(self.faces)):
            center = self.get_face_center(i)
            normal = self.get_face_world_normal(i)
            to_finger = finger_world_pos - center

            if np.dot(to_finger, normal) < 0:
                continue

            dist = np.linalg.norm(to_finger)
            if dist < best_dist:
                best_dist = best_idx = None  # reset
                best_dist = dist
                best_idx  = i

        return best_idx

    def extrude_face(self, face_idx, delta):
        """
        Push the 4 vertices of face_idx outward by delta units
        along the face's local normal.
        """
        normal   = self.face_normals[face_idx]
        for vi in self.faces[face_idx]:
            self.vertex_offsets[vi] += normal * delta