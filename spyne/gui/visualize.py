## ---------------------------------------------------------------- ##
## VISUALIZE.PY
## ---------------------------------------------------------------- ##
## This file provides some code to visualize neural networks in
## Python, using my code.
## ---------------------------------------------------------------- ##
## Requires:
##    * wxPython
##    * PyOpenGL
## ---------------------------------------------------------------- ##

import sys, multiprocessing, operator, copy
import numpy     as np
import colorsys  as col
import threading as th
#import glFreeType2
#import glFreeType
#import matplotlib.cm as cm
from   ..neural   import *
from   space    import *
from   wx.glcanvas  import *


try:
    from OpenGL.GLUT import *
    from OpenGL.GL   import *
    from OpenGL.GLU  import *
except ImportError:
    print """ERROR: PyOpenGL not installed properly.  """

try:
    from OpenGLContext.scenegraph.text import *
except ImportError:
    print """ERROR: OpenGLContext not installed properly.  """

try:
    import wx
    from   wx import glcanvas
except ImportError:
    print """ERROR: Cannot find wxPython. """

        
X_PLANE_PADDING = .1
Y_PLANE_PADDING = .1
Z_PLANE_PADDING = .5
PLANE_ALPHA     = .85


X_GROUP_SPACE   = .15
Y_GROUP_SPACE   = .25
Z_GROUP_SPACE   = .25

X_GAP           = .15
Y_GAP           = .3
Z_GAP           = .15

NEURON_WIDTH    = .05
NEURON_HEIGHT   = .10
NEURON_ALPHA    = .85
NEURON_PADDING  = .01
GROUP_PADDING   = .02
WINDOW          = None

## ---------------------------------------------------------------- ##
## COLOR FUNCS
## ---------------------------------------------------------------- ##

#JET=cm.jet(range(0,200,1))

def RedToYellow(v):
    c = col.hsv_to_rgb(0.1666*v, 1, 0.3+0.7*v)
    #print v, c
    return c

def RedToYellowRGB(v):
    if v > 1:
        a = 1
    else:
        a = v
    return (0.7+0.3*(a**2), a**2, 0) 


def BlackToRed(v):
    if v > 1:
        a = 1
    else:
        a = v
    return (a**2, 0, 0) 

def BlueBlackRed(v):
    if v > 1:
        a = 1
    elif v < -1:
        a = -1
    else:
        a = v
    
    if a > 0:
        return (a**2, 0, 0)
    else:
        return (0,0,a**2)

def BlackRedYellow(v):
    if v>=0:
        r = min(v/.6, 1)
        g = max(0, (v-.6)*2)
        b = 0
    else:
        b = min(abs(v)/.6, 1)
        r = max(0, (abs(v)-.6)*2)*0.6
        g = r
    return (r, g, b)

#COLOR_FUNCTION  = BlueBlackRed

#COLOR_FUNCTION  = RedToYellow

#COLOR_FUNCTION  = Jet

COLOR_FUNCTION  = BlackRedYellow

## ---------------------------------------------------------------- ##
## PRIMITIVES
## ---------------------------------------------------------------- ##

def DrawNeuron(x, y, z, a):
    """Draws a neuron at a specified location"""
    y_offset= a*NEURON_WIDTH/2
    glPushMatrix()
    glTranslate(x, y+y_offset+0.001, z)
    #if a == 0:
    #    a = 0.0001
    glScalef(1., a, 1.)
    c = COLOR_FUNCTION(a)
    #glColor4f(1., a, 0., NEURON_ALPHA)
    glColor4f(c[0], c[1], c[2], NEURON_ALPHA)
    glutSolidCube(NEURON_WIDTH)
    #glColor3f(0., 0., 0.)
    #glScalef(1.005, 1, 1.005)
    #glTranslate(0,0.001,0)
    #glutWireCube(NEURON_WIDTH)
    glPopMatrix()

def DrawPlane(x, y, z, width, depth, color=(0.67, 0.67, 1, PLANE_ALPHA)):
    """Draws a plane at the specified location"""
    # The plane area
    glColor4f(color[0], color[1], color[2], color[3])
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glBegin(GL_POLYGON)
    glNormal3f(0.,1.,0.)
    glVertex3f(x+width, y, z)
    glVertex3f(x, y, z)
    glVertex3f(x, y, z+depth)
    glVertex3f(x+width, y, z+depth)
    glEnd()

    glBegin(GL_POLYGON)
    glNormal3f(0.,-1.,0.)
    glVertex3f(x+width, y, z)
    glVertex3f(x, y, z)
    glVertex3f(x, y, z+depth)
    glVertex3f(x+width, y, z+depth)
    glEnd()
    # The border
    glColor3f(0.,0.,0.)
    glBegin(GL_LINE_LOOP)
    glNormal3f(0.,1.,0.)
    glVertex3f(x, y, z+depth)
    glVertex3f(x+width, y, z+depth)
    glVertex3f(x+width, y, z)
    glVertex3f(x, y, z)
    glEnd()

def GroupVolumeSize(g):
    G      = g.geometry
    width  = 2.0*GROUP_PADDING + G[0]*NEURON_WIDTH + max(0, G[0]-1)*NEURON_PADDING
    depth  = 2.0*GROUP_PADDING + G[1]*NEURON_WIDTH + max(0, G[1]-1)*NEURON_PADDING
    height = 0.
    return Volume(width, height, depth)


## ---------------------------------------------------------------- ##
## WINDOWING
## ---------------------------------------------------------------- ##


def DisplayText(text, point, font=GLUT_BITMAP_HELVETICA_12):
    p=copy.copy(point)
    glColor3f(.0,.0,.0)
    for c in text:
        glRasterPos2f(p.x, p.y)
        glutBitmapCharacter(font, ord(c))
        p.Translate(glutBitmapWidth(font, ord(c))/500.0, 0, 0)
    return


def DisplayText_(text, point, font=GLUT_BITMAP_HELVETICA_12):
    #print "Display", text
    f = glFreeType.font_data("Test.ttf", 8)
    glColor3f(1.0, .0, .0)
    f.glPrint(50, 50, text)
    return


## ---------------------------------------------------------------- ##
## WXPYTHON-BASED GUI
## ---------------------------------------------------------------- ##

#class SPyNECanvas(glcanvas.GLCanvas):
class SPyNECanvas(wx.glcanvas.GLCanvas):
    INPUTS      = 101
    ACTIVATIONS = 102
    THRESHOLDS  = 103
    
    def __init__(self, parent, circuit=None, runFunction=None):
        attribList = (wx.glcanvas.WX_GL_DOUBLEBUFFER,
                      wx.glcanvas.WX_GL_RGBA, 0)
        wx.glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribList)
        self.pov  = Point(1.0, 1.0, 2.0)     # Where the camera is
        #self.pov  = Point(0., 2., 2.)      # Where the camera is
        self.pos  = Point(0., 0., 0.)      # Where the model is
        self.up   = Point(0., 1., 0.)      # Upwards normal
        self.rot  = Point(0., 0., 0.)      # Rotation matrix
        self.step = .1
        self.init = False
        # initial mouse position
        # self.lastx = self.x = 30
        # self.lasty = self.y = 30
        self.size  = None
        self.__pinlist = None
        self.__selected = None
        self.__vvalues = self.ACTIVATIONS
        self.context = wx.glcanvas.GLContext(self)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        
        # The internal circuit to visualize
        self.circuit = circuit
        circuit.AddUpdateListener(self.OnUpdate)

#    def OnSize(self, arg):
#        w,h = self.GetClientSize()
#        self.w = w
#        self.h = h
#        dc = wx.ClientDC(self)
        #self.Render(dc)
#        self.OnDraw()

    def GetCenter(self):
        """Returns the center of the circuit (for rotation)"""
        return Point(0, -0.875, 0)

    def SetVisibleValues(self, val):
        self.__vvalues=val

    def GetVisibleValues(self):
        return self.__vvalues

    def AddToSelected(self, obj):
        if obj not in self.__selected:
            self.__selected.append(obj) 

    def RemoveFromSelected(self, obj):
        if obj in self.__selected:
            self.selected.remove(obj)

    def GetSelected(self, obj):
        return self.__selected

    def OnUpdate(self, x, y):
        """When the circuit is updated, simply redraw"""
        self.OnDraw()
        
    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.


    def ArrangeCircuit(self, circuit):
        """
        Arranges the groups and the projections in a circuit in a
        way that is convenient for visualization. Returns a 'pin-point'
        list, i.e. a dictionary where every object in the circuit is
        associated with a 'point' (or more points in case of
        projections) in 3D space.
        """
        K = {}  # the group/depth table
        O = {}  # The object table
    
        A = circuit.GetGroups()
        G = copy.copy(circuit.GetInput())  # groups
        P = []                             # alerady processed

        for i in G:
            K[i] = 0

        # First, we proceed forward
        #
        while len(G) > 0:
            g = G.pop()
            P.append(g)
            l = K[g]+1 # Putative level
            C = [x.groupTo for x in g.outgoingProjections]
            C = [c for c in C if c in A]
            C = [c for c in C if c not in P]
            C.sort(key=lambda x: x.name)
            for c in C:
                K[c] = l
            G = C+G   # Append children to the front

        # Now we proceed backward, looking for missed groups
        # among the incoming pathways.
        G = copy.copy(sorted(K.keys(), key=lambda x: x.name))
        for g in G:
            l = max(K[g]-1, 0)   # Putative level
            C = [p.groupFrom for p in g.incomingProjections]
            C = [c for c in C if c in A]
            C = [c for c in C if c not in G]
            C.sort(key=lambda x: x.name)
            for c in C:
                if c in K.keys() and K[c] < l:
                    pass
                else:
                    K[c] = l

        # --- This is some kind of bubble sort, pushing groups down
        # --- as much as possible.
        NOC = 1   # Number of changes
        EPO = 0   # Number of epochs (to avoid infinite recursions)
        G   = copy.copy(sorted(K.keys(), key=lambda x: x.name))
        G   = [g for g in G if g not in circuit.GetInput()]
        G.sort(key=lambda x: x.name)
        while NOC > 0 and EPO < 100:
            NOC  = 0
            EPO += 1
            for g in G: 
                # We make a list of groups that send projections, and divide it
                # in two: Those that are directly connected in feedback loop (C)
                # and those that are not (Z).
                l =  K[g]   # level of G
                d =  circuit.GetGroupDepth(g, fromTop=False)
                C =  [p.groupFrom for p in g.incomingProjections]
                C =  [c for c in C if c in G]
                Z =  [c for c in C if c in [x.groupTo for x in g.outgoingProjections]]
                C =  [c for c in C if c not in [x.groupTo for x in g.outgoingProjections]]

                # --- Now we consider only those whose level is > then our group's
                CL = [c for c in C if K[c] >= l]
                ZL = [z for z in Z if circuit.GetGroupDepth(z, False) >= d and K[z] >=l]
                if len(CL) > 0 or len(ZL):
                    NOC+=1
                    K[g]+=1
                
                if g in circuit.GetOutput():
                    K[g] = max(K.values())+1

        # After bubble-sorting the groups, we re-number the levels to make
        # sure they are all continuous and no level is skipped.
        l = copy.copy(K.values())
        ideal = range(max(l)+1)
        diff = [x for x in ideal if x not in l]
        while len(diff) > 0:
            for missed in diff:
                for g, l in K.iteritems(): 
                    if l > missed:
                        K[g]-=1
            l = copy.copy(K.values())
            ideal = range(max(l)+1)
            diff = [x for x in ideal if x not in l]
        
        # Now, we create the opposite structure, a dictionary that
        # associates each level L to a list of groups [G1, G2...Gn]
        V = list(set(K.values()))  # the list of levels
        L = {}
        for v in V:
            L[v] = [k for k in K.keys() if K[k]==v]

        # ------------------------------------------------------------
        # Finally, we create a dictionary that associates each element
        # of the circuit (group or projection) with one or more points
        # in space.
        for l, G in L.iteritems(): #zip(L.keys(), L.values()):
            G.sort(key=lambda x: x.name )
            W   = [GroupVolumeSize(g).width for g in G]
            off = -1*(reduce(operator.add, W) + X_GROUP_SPACE * (len(G)-1))/2
            y   = -1*l*Y_GROUP_SPACE
            z   = 0
            for i in range(len(G)):
                x = off + W[i]/2
                O[G[i]] = Point(x, y, z)
                off = off + W[i] + X_GROUP_SPACE

        # ------------------------------------------------------------
        # Now the difficult part: the projections. Each projection
        # needs to be associated with a list of 2 or 3 points, depending
        # on whether it projects up/down more than one layer or not.
        FRONT = {}
        BACK  = {}
        P     = {}    # empty dictionary of proejctions
        G = O.keys()  # so far we have only groups in our keys.
        for g in G:
            gFront = []
            gBack  = []
            IN  = g.incomingProjections
            OUT = g.outgoingProjections
            for i in IN:
                if O[i.groupFrom].y > O[g].y:
                    gBack.append((i, O[i.groupFrom].x))
                else:
                    gFront.append((i, O[i.groupFrom].x))
            for o in OUT:
                if O[o.groupTo].y > O[g].y:
                    gBack.append((o, O[o.groupTo].x))
                else:
                    gFront.append((o, O[o.groupTo].x))
            gFront.sort(key=lambda x: x[1])
            gBack.sort(key=lambda x: x[1])
                
            FRONT[g] = [x[0] for x in gFront]
            BACK[g]  = [x[0] for x in gBack]
                
        # Now we trace all the incoming projections
        for g in G:
            space = 0.03
            for p in g.incomingProjections:
                # --- decide whether it is a back or front projections
                pfrom = copy.copy(O[p.groupFrom])
                pto   = copy.copy(O[g])
                if p in FRONT[g]:
                    n = len(FRONT[g])
                    i = FRONT[g].index(p)
                    x_off = -1*(n-1)*space/2 + i*space
                    z_off = GroupVolumeSize(g).depth/2
                    pto.Translate(x_off, 0, z_off)
            
                elif p in BACK[g]:
                    n = len(BACK[g])
                    i = BACK[g].index(p)
                    x_off = -1*(n-1)*space/2 + i*space
                    z_off = -1*GroupVolumeSize(g).depth/2
                    pto.Translate(x_off, 0, z_off)
                else:
                    print "****** ALERT: %s ******"
                    
                if p in FRONT[p.groupFrom]:
                    n = len(FRONT[p.groupFrom])
                    i = FRONT[p.groupFrom].index(p)
                    x_off = -1*(n-1)*space/2 + i*space
                    z_off = GroupVolumeSize(p.groupFrom).depth/2
                    pfrom.Translate(x_off, 0, z_off)
                        
                elif p in BACK[p.groupFrom]:
                    n = len(BACK[p.groupFrom])
                    i = BACK[p.groupFrom].index(p)
                    x_off = -1*(n-1)*space/2 + i*space
                    z_off = -1*GroupVolumeSize(p.groupFrom).depth/2
                    pfrom.Translate(x_off, 0, z_off)
                else:
                    print "****** ALERT: %s ******"


                # if the projection crosses an in-between layer,
                # we need to deviate it away from it. This is signaled
                # by introducing a third point that represents the distance
                # to go to the back of the circuit.

                pout = None
                R    = []
                if K[g] - K[p.groupFrom] < -1:
                    R = range(K[g]+1, K[p.groupFrom])
                elif K[g] - K[p.groupFrom] > 1:
                    R = range(K[p.groupFrom]+1, K[g])
                if len(R) > 0:
                    B  = []    # List of all the groups in-between
                    for lyr in R:
                        B.extend(L[lyr])
                    z = min([O[x].z - GroupVolumeSize(x).depth/2 for x in B]) - 0.05
                    y = (O[g].y + O[p.groupFrom].y)/2
                    x = (O[g].x + O[p.groupFrom].x)/2
                    pout=Point(x, y, z)
                    
                
                P[p]=(pfrom, pto, pout)

        # --- Add all projections to the O dictionary ----------------
        for p, pnts in P.iteritems():
            O[p] = pnts
        return O


    #def OnSize(self, event):
    #    size = self.size = self.GetClientSize()
    #    if self.GetContext():
    #        self.SetCurrent()
    #        h = min([size.width, size.height])
    #        #glViewport(0, 0, size.width, size.height)
    #        glViewport(0, 0, h, h)
    #    event.Skip()

    def OnPaint(self, evt):
        dc = wx.PaintDC(self)
        #self.Render(dc)

        #def OnPaint(self, event):
        #"""Repaints the scene"""
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()

#    def Render(self):  #, dc):
#        self.SetCurrent()
#        glrender(self.w, self.h)
#        self.SwapBuffers()

    def DrawArrow(self, p1, p2, p3=None, context=None, color=(0., 0., 0.)):
        """
        Draws an arrow from point p1 to p2, passing (in case) thorugh
        a third point p3 to avoid colliding with other groups.
        """
        glColor3f(color[0], color[1], color[2])
        if p1.y-p2.y > 0:
            v =  1
            d =  1
            
        elif p1.y == p2.y:
            v =  1
            d = -1
            
        else:
            v = -1
            d = -1
    
        glBegin(GL_LINE_STRIP)
        glVertex3f(p1.x, p1.y, p1.z)
        glVertex3f(p1.x, p1.y-.075*v, p1.z)

        # if we have a third point, we need to go backwards.
        if p3 != None:
            z = min([p3.z, p2.z])
            glVertex3f(p1.x, p1.y+.075*v-v*Y_GROUP_SPACE, z)
            #glVertex3f(p2.x, p2.y+.075*d, z)
            glVertex3f(p2.x, p2.y+d*Y_GROUP_SPACE/2, p2.z)
        glVertex3f(p2.x, p2.y+.075*d, p2.z)
        
        glVertex3f(p2.x, p2.y+.075*d, p2.z)
        glVertex3f(p2.x, p2.y, p2.z)
        glEnd()
        #Front
        glBegin(GL_TRIANGLES)
        glNormal(0., 0., 1.)
        glVertex3f(p2.x - .02,  p2.y + 0.04*d, p2.z)
        glVertex3f(p2.x, p2.y, p2.z)
        glVertex3f(p2.x + .02, p2.y + 0.04*d, p2.z)
        glNormal(0., 0., -1.)
        glVertex3f(p2.x - .02,  p2.y + 0.04*d, p2.z)
        glVertex3f(p2.x, p2.y, p2.z)
        glVertex3f(p2.x + .02, p2.y + 0.04*d, p2.z)
        glEnd()


    def DrawGroup(self, g, x, y, z):
        """Draws a neural group in space"""
        if self.__vvalues == self.ACTIVATIONS:
            A    = g.GetActivations()
        else:
            A    = g.inputs
        V        = GroupVolumeSize(g)
        w        = V.width
        d        = V.depth
        x_offset = w/2.
        z_offset = d/2.
        x_ref    = x - x_offset
        z_ref    = z - z_offset

        if g.GetClamped():
            DrawPlane(x_ref, y, z_ref, w, d, (.3, .3, .3, PLANE_ALPHA))
        else:
            DrawPlane(x_ref, y, z_ref, w, d)
            
        z_ref = z + z_offset   # New offset to start drawing neurons from front
        
        for i in xrange(g.geometry[0]):
            for j in xrange(g.geometry[1]):
                nx = x_ref + GROUP_PADDING + i*NEURON_PADDING + (i+.5)*NEURON_WIDTH
                nz = z_ref - GROUP_PADDING - j*NEURON_PADDING - (j+.5)*NEURON_WIDTH
                DrawNeuron(nx, y, nz, A[j*g.geometry[0]+i])

    
    def OnSize(self, event):
        wx.CallAfter(self.DoSetViewport)
        event.Skip()

    def DoSetViewport(self):
        size = self.size = self.GetClientSize()
        self.SetCurrent(self.context)
        glViewport(0, 0, size.width, size.height)

                
    def InitGL(self):
        """Initializes the OpenGL library"""
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glutInitDisplayMode(GLUT_MULTISAMPLE)

        glMatrixMode(GL_PROJECTION)
        # --- Should use Frustum or Ortho depending on a flag
        glFrustum(-0.5, 0.5, -0.5, 0.5, 1.0, 3.0)
        #glOrtho(-.3, .3, -2.25, 0.25, 1.0, 4.0)
        # Currently uses only ORTHO
        #glOrtho(-1, 1, -1, 1, -4, 4)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)

        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(1.2)
        #glSampleCoverage(0.99, GL_TRUE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glShadeModel (GL_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        #print glIsEnabled(GL_MULTISAMPLE)
        glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE)
        glEnable(GL_SAMPLE_ALPHA_TO_ONE)
        glEnable(GL_SAMPLE_COVERAGE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        #print glIsEnabled(GL_SAMPLE_ALPHA_TO_COVERAGE)
        #print glIsEnabled(GL_SAMPLE_ALPHA_TO_ONE)
        #print glIsEnabled(GL_SAMPLE_COVERAGE)

        ## Now the Prospective
        glMatrixMode(GL_MODELVIEW)

        self.init=True
        glutInit(sys.argv)
        

    def RotateScene(self):
        """
        Rotates the model along the X, Y, Z axis according to the
        values specified in the ROT (= Rotation) object.
        """
        # --- Work on the modelview matrix -----------------
        # There should be no other modality than ModelView
        # at this point, but it's better be safe than sorry. 
        glMatrixMode(GL_MODELVIEW)

        # --- Go to the circuit center ---------------------
        c = self.GetCenter()
        glTranslatef(c.x, c.y, c.z)
        glRotated(self.rot.x, 1, 0, 0)
        glRotated(self.rot.y, 0, 1, 0)
        glRotated(self.rot.z, 0, 0, 1)

        # --- Back to the origin ---------------------------
        glTranslatef(-c.x, -c.y, -c.z)


    def TranslateScene(self):
        """
        Modifies the position of the object according to the values
        specified in the POS object (current unimplemented)
        """
        c = self.GetCenter()
        # There should be no other modality than ModelView
        # at this point, but it's better be safe than sorry. 
        glMatrixMode(GL_MODELVIEW)
        
        # --- Go to the circuit center ---------------------
        glTranslatef(c.x, c.y, c.z)

        # --- Now translate to simulate a change in POV ----
        glTranslatef(self.pos.x, -self.pos.y, -self.pos.z)

        # --- Back to the origin ---------------------------
        glTranslatef(-c.x, -c.y, -c.z)


    def OnDraw(self):
        """Redraws the objects on the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        print self.pov, self.rot, self.pos

        # --- Setting up the camera ------------------------
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        c = self.GetCenter()
        gluLookAt(self.pov.x, self.pov.y, self.pov.z,
                  #self.pos.x, self.pos.y, self.pos.z,
                  c.x, c.y, c.z,
                  self.up.x, self.up.y, self.up.z)

        
        # --- Applies the rotation -------------------------
        #self.TranslateScene()

        # --- Draws the circuit ----------------------------
        glPushMatrix()

        # --- First, all the necessary transformations -----
        self.RotateScene()
        self.TranslateScene()

        # --- Then, we draw the objects in the pinlist -----
        if self.circuit is not None:
            if self.__pinlist == None:
                self.__pinlist = self.ArrangeCircuit(self.circuit)
            O = self.__pinlist
            for obj, pnts in O.iteritems():
                if type(obj) == Group:
                    p = pnts
                    self.DrawGroup(obj, p.x, p.y, p.z)
                    v = GroupVolumeSize(obj)
                    xyz = Point(p.x + v.width/2,
                                p.y,
                                p.z + v.depth/2)
                    DisplayText(obj.name, xyz)
                elif type(obj) == Projection:
                    # Sets the color
                    
                    if (obj.weights<=0).all():
                        glEnable(GL_LINE_STIPPLE)
                        glLineStipple(2, 0xAAAA)
                    self.DrawArrow(pnts[0], pnts[1], pnts[2])
                    glDisable(GL_LINE_STIPPLE)
                    
        # --- Pops the scene matrix
        glPopMatrix()
        

        # Sets the camera
        
        self.SwapBuffers()
        #self.Render() #SwapBuffers()
        

