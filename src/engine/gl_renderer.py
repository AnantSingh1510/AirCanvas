from OpenGL.GL import *
from OpenGL.GLU import *

class GLRenderer:
    def __init__(self):
        self.lines = []
        self.current_stroke = []

    def init_gl(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (w / h), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    def draw_scene(self, rotation_y):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3.0)
        glRotatef(rotation_y, 0, 1, 0)

        # Draw the 3D strokes
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