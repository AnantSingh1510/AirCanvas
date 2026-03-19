from OpenGL.GL import *
from OpenGL.GLU import *
import pygame

class ThreeDCanvas:
    def __init__(self):
        self.points = []
        self.rotation_x = 0
        self.rotation_y = 0

    def add_point(self, x, y, z, color):
        nx = (x - 0.5) * 2
        ny = (0.5 - y) * 2
        nz = z * 5 # Scale depth
        self.points.append((nx, ny, nz, color))

    def render(self):
        glPushMatrix()
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        glBegin(GL_LINE_STRIP)
        for p in self.points:
            glColor3f(p[3][0], p[3][1], p[3][2])
            glVertex3f(p[0], p[1], p[2])
        glEnd()
        glPopMatrix()