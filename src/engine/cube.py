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
        verts = self.get_transformed_vertices()
        face  = self.faces[face_idx]
        return np.mean(verts[list(face)], axis=0)

    def get_face_world_normal(self, face_idx):
        R = self.get_transform_matrix()
        return R @ self.face_normals[face_idx]

    def select_nearest_face(self, finger_world_pos):
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
                best_dist = dist
                best_idx  = i

        return best_idx

    def select_face_at_pointer(self, pointer_xy, margin=0.18):
        pointer = np.array(pointer_xy, dtype=float)
        verts = self.get_transformed_vertices()
        candidates = []

        for i, face in enumerate(self.faces):
            face_verts = verts[list(face)]
            poly = face_verts[:, :2]
            center = np.mean(face_verts, axis=0)
            normal = self.get_face_world_normal(i)
            min_edge_dist = min(
                self._point_segment_distance(pointer, poly[j], poly[(j + 1) % len(poly)])
                for j in range(len(poly))
            )

            if self._point_in_polygon(pointer, poly) or min_edge_dist <= margin:
                facing_score = normal[2]
                candidates.append((center[2], facing_score, -min_edge_dist, i))

        if candidates:
            return max(candidates)[3]

        best_idx = None
        best_dist = margin * 1.75
        for i, face in enumerate(self.faces):
            center = np.mean(verts[list(face)], axis=0)
            dist = np.linalg.norm(pointer - center[:2])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _point_in_polygon(self, point, polygon):
        inside = False
        x, y = point
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def _point_segment_distance(self, point, a, b):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom == 0.0:
            return float(np.linalg.norm(point - a))
        t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
        closest = a + t * ab
        return float(np.linalg.norm(point - closest))

    def extrude_face(self, face_idx, delta):
        normal   = self.face_normals[face_idx]
        for vi in self.faces[face_idx]:
            self.vertex_offsets[vi] += normal * delta
