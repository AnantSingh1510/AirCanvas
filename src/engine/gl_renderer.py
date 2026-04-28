from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class GLRenderer:
    def __init__(self):
        self.lines = []
        self.current_stroke = []
        self.cube = None

    def init_gl(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (w / h), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        glLightfv(GL_LIGHT0, GL_POSITION, [ 2.0,  4.0,  3.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [ 0.2,  0.2,  0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [ 0.9,  0.9,  0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [ 0.5,  0.5,  0.5, 1.0])
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [-2.0, -2.0, -1.0, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  [ 0.45, 0.45, 0.45, 1.0])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.35, 0.35, 0.35, 1.0])
        glClearColor(0.05, 0.05, 0.1, 1.0)

    FACE_NORMALS = [
        ( 0,  0, -1), ( 0,  0,  1),
        ( 0, -1,  0), ( 0,  1,  0),
        (-1,  0,  0), ( 1,  0,  0),
    ]

    def draw_cube_faces(self, vertices, faces, face_colors, selected_face):
        for i, face in enumerate(faces):
            r, g, b = face_colors[i]
            if i == selected_face:
                glColor4f(min(r * 1.8, 1.0), min(g * 1.8, 1.0), min(b * 1.8, 1.0), 1.0)
            else:
                glColor4f(r, g, b, 0.9)
            glNormal3fv(self.FACE_NORMALS[i])
            glBegin(GL_QUADS)
            for vi in face:
                glVertex3fv(vertices[vi])
            glEnd()

    def draw_selected_face_outline(self, vertices, face_idx, faces):
        """Draw a bright white outline around the selected face."""
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        face = faces[face_idx]
        glBegin(GL_LINE_LOOP)
        for vi in face:
            glVertex3fv(vertices[vi])
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_extrude_arrow(self, face_center, face_normal):
        """Draw a direction arrow showing which way the face will extrude."""
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glColor4f(1.0, 0.9, 0.2, 0.9)
        tip = face_center + face_normal * 0.25
        glBegin(GL_LINES)
        glVertex3fv(face_center)
        glVertex3fv(tip)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_cube_edges(self, vertices, edges):
        glDisable(GL_LIGHTING)
        glLineWidth(2.5)
        glColor4f(1.0, 1.0, 1.0, 0.95)
        glBegin(GL_LINES)
        for i, j in edges:
            glVertex3fv(vertices[i])
            glVertex3fv(vertices[j])
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_grid(self):
        glDisable(GL_LIGHTING)
        glLineWidth(0.5)
        glColor4f(0.3, 0.3, 0.4, 0.35)
        glBegin(GL_LINES)
        for i in range(-5, 6):
            glVertex3f(i * 0.5, -1.2, -3.0)
            glVertex3f(i * 0.5, -1.2,  1.0)
            glVertex3f(-2.5, -1.2, (i-5)*0.5+0.5)
            glVertex3f( 2.5, -1.2, (i-5)*0.5+0.5)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_scene(self):
        glClear(GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -4.0)

        self.draw_grid()

        if self.cube is not None:
            verts    = self.cube.get_transformed_vertices()
            selected = self.cube.selected_face

            self.draw_cube_faces(verts, self.cube.faces,
                                 self.cube.face_colors, selected)
            self.draw_cube_edges(verts, self.cube.edges)

            if selected is not None:
                self.draw_selected_face_outline(verts, selected, self.cube.faces)
                center = self.cube.get_face_center(selected)
                normal = self.cube.get_face_world_normal(selected)
                self.draw_extrude_arrow(center, normal)
