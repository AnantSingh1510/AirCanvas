from OpenGL.GL import *
from OpenGL.GLU import *

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
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def draw_cube_faces(self, vertices, faces, face_colors):
        glEnable(GL_BLEND)
        for i, face in enumerate(faces):
            r, g, b = face_colors[i]
            glColor4f(r, g, b, 0.3)
            glBegin(GL_QUADS)
            for vi in face:
                glVertex3fv(vertices[vi])
            glEnd()

    def draw_cube_edges(self, vertices, edges):
        glLineWidth(2)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        for i, j in edges:
            glVertex3fv(vertices[i])
            glVertex3fv(vertices[j])
        glEnd()

    def draw_scene(self, rotation_y):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3.0)

        if self.cube is not None:
            verts = self.cube.get_transformed_vertices()
            self.draw_cube_faces(verts, self.cube.faces, self.cube.face_colors)
            self.draw_cube_edges(verts, self.cube.edges)

        glRotatef(rotation_y, 0, 1, 0)
        glLineWidth(3)
        for stroke in self.lines:
            glBegin(GL_LINE_STRIP)
            for p in stroke:
                glColor3f(0, 1, 0)
                glVertex3f((p[0]-0.5)*2, (0.5-p[1])*2, -p[2]*10)
            glEnd()

        if self.current_stroke:
            glBegin(GL_LINE_STRIP)
            for p in self.current_stroke:
                glColor3f(1, 1, 1)
                glVertex3f((p[0]-0.5)*2, (0.5-p[1])*2, -p[2]*10)
            glEnd()