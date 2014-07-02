## ---------------------------------------------------------------- ##
## NEURAL.PY
## ---------------------------------------------------------------- ##
## Defines the basic classes for simulating neural networks.
## ---------------------------------------------------------------- ##

import math  as m
import numpy as np
from collections import deque
from selection   import Boltzmann
from basic       import SPyNEObject, ParametrizedObject, ContextContainer

## ---------------------------------------------------------------- ##
## CONSTANTS
## ---------------------------------------------------------------- ##

PY_NN_RECORD_NEVER  = 0     # Never record activations
PY_NN_RECORD_FINAL  = 1     # During updates, record only at the end 
PY_NN_RECORD_ALWAYS = 2     # During updates, record all the intermediates


## ---------------------------------------------------------------- ##
## NAMING MECHANISM
## ---------------------------------------------------------------- ##

# --- Gentemp dictionary --------------------------------------------- 
__GT_NAMES__ = {}

def GenTemp(root="obj-"):
    """Generates a new unique name based on the given root"""
    global __GT_NAMES__
    if root in __GT_NAMES__.keys():
        __GT_NAMES__[root]+=1
    else:
        __GT_NAMES__[root]=0
    return "%s%d" % (root, __GT_NAMES__[root])

## ---------------------------------------------------------------- ##
## Scalar functions
## ---------------------------------------------------------------- ##

def SLinear (x):
    """Linear Function"""
    return x

def SLBLinear (x, bound=0):
    """Lower bound linear"""
    return max(x, bound)

def SStep(x, threshold=0):
    """Step function (0,1)"""
    if (x > threshold):
        return 1
    else:
        return 0

def SBinary(x, threshold=0):
    """Binary function (-1, 1)"""
    if (x > threshold):
        return 1
    else:
        return -1

def SSigmoid(x, gain=-1):
    """Sigmoid function"""
    return 1 / (1 + m.exp(x*gain))

def STanh(x, gain=1):
    """Hyperbolic tangent"""
    return m.tanh(gain*x)

def STanh_plus(x, gain=1):
    """Positive hyperbolic tangent"""
    return max([0, m.tanh(gain*x)])

## --- VECTOR VERSIONS -------------------------------------------- ##

Linear    = np.vectorize(SLinear)
LBLinear  = np.vectorize(SLBLinear)
Step      = np.vectorize(SStep)
Binary    = np.vectorize(SBinary)
Sigmoid   = np.vectorize(SSigmoid)
Tanh      = np.vectorize(STanh)
#Tanh_plus = np.vectorize(STanh_plus)
Tanh_plus = np.frompyfunc(STanh_plus, 1,1)


## ---------------------------------------------------------------- ##
## ERROR FUNCTIONS
## ---------------------------------------------------------------- ##
def Error(a1, a2):
    """Classical error function for a vector and its target value"""
    return np.sum((a1-a2)**2)


## ---------------------------------------------------------------- ##
## EXCEPTIONS
## ---------------------------------------------------------------- ##
def NeuralException(Exception):
    """An exception occurring in a Neural object"""
    def __init__(source, msg):
        self.source=source
        self.msg=msg

## ---------------------------------------------------------------- ##
## BASIC NEURAL OBJECTS
## ---------------------------------------------------------------- ##

class NeuralObject (SPyNEObject):
    """The root of all objects"""
    def __init__(self):
        self.name=GenTemp("NeuralObject-")


def GenericUpdate(group, context=None):
    """
    The basic update function for a group. When invoked, this function
    will:
      1. Clear the group's input values;
      2. Propagate through all the incoming projections (so that each
         projections adds to the group's input values
      3. Finally, calculate the activations of the new input values
    """
    group.ClearInputs()
    for p in group.incomingProjections:
        p.PropagateThrough()
    group.CalculateActivations()

    ## Here learning functions
    pass

## ---------------------------------------------------------------- ##
## GROUP
## ---------------------------------------------------------------- ##
## A group is the basic class of the module.  A group is a
## homogeneus collection of neurons, i.e. they all share the same
## activation function, behave in similar ways, and are updated
## together.   A group of N neurons contains the following members:
##
## (a) inputs: a Nx1 matrix containing the most recent inputs.
## (b) thresholds: a Nx1 matrix containing the neurons' thresholds
## (c) activations: a Nx1 matrix containing the neurons' current
##     activation values
## (d) baselines: a Nx1 matrix containing the neurons' baselines
##     (required by some algorithms; can be updated independently)
## (e) mask: a Nx1 binary matrix detailing whether each neuron is
##     'working' or not.  Masks provide an easy way to virtually
##     damage a circuit.
##   
## ---------------------------------------------------------------- ##
class Group(NeuralObject, ContextContainer):
    """A group of neurons"""
    def __init__(self, size=0, name=None,
                 activationFunction=Sigmoid,
                 activationDerivative=None,
                 updateFunction=GenericUpdate,
                 context=None):
        """Initializes the group"""
        if (name is None):
            self.name=GenTemp("Group-")
        else:
            self.name=name
        self.SetActivationFunction(activationFunction)
        self.ActivationDerivative=activationDerivative
        self.SetUpdateFunction(updateFunction)
        self.size        = size
        self.activations = np.zeros((size, 1))
        self.geometry    = self.activations.shape
        self.baselines   = np.zeros((size, 1))
        self.mask        = np.ones((size, 1))
        self.thresholds  = np.zeros((size, 1))
        self.inputs      = np.zeros((size, 1))
                
        # --- Private fields ---------------------
        self.__history   = deque()
        self.__hlength   = 1
        
        self.incomingProjections=[]
        self.outgoingProjections=[]
        self.kwta = False
        self.SetKWTAFunction(None)
        self.SetClamped(False)
        self.SetContext(context)

    def __repr__(self):
        """String representation of a group"""
        return "<%s, %s>" % (self.name, self.activations.size)

    def __str__(self):
        """String representations of a group"""
        return self.__repr__()

    def SetActivations(self, activations, clamped=False):
        """Sets the activation values of the group's neurons"""
        if self.activations.shape == activations.shape:
            self.activations=np.copy(activations)
            if clamped:
                self.SetClamped(True)
        else:
            #print "Nonmatch", self.activations.shape, activations.shape
            raise NeuralException(self, "Mismatching activations")

    def GetActivations(self):
        """Returns the activation values of the neurons"""
        return self.activations

    def SetUpdateFunction(self, func):
        """Sets the update function"""
        self.__updateFunction=func

    def GetUpdateFunction(self):
        """Returns the update function"""
        return self.__updateFunction

    def SetActivationFunction(self, func):
        """Sets the activation function"""
        self.__activationFunction=func

    def GetActivationFunction(self):
        """Returns the activation function"""
        return self.__activationFunction

    def Update(self, record=False):
        """Updates the neurons (by calling the current Update function)"""
        if record:
            self.RecordActivations()

        if not self.__clamped:
            self.__updateFunction(self, self.GetContext())


    def ClearInputs(self):
        """Sets the neuronal inputs to zero"""
        self.inputs = np.zeros(self.inputs.size).reshape(self.inputs.shape)
    
    def CalculateActivations(self):
        """Calculates and updates the activation values"""
        newActivations=self.__activationFunction(self.inputs-self.thresholds)
        self.activations=newActivations*self.mask
        
        # --- If KWTA is allowed, calculate the active units.
        
        if (self.kwta and self.__kwtaFunction != None):
            self.activations=self.__kwtaFunction(self.activations)

    def SetHistoryLength(self, n):
        if n > 0:
            self.__hlength = n
            if n > len(self.__history):
                j = 0
                while j < n - len(self.__history):
                    self.__history.popleft()
                    j += 1

    def GetHistoryLength(self):
        return self.__hlength

    def GetHistory(self):
        return self.__history

    def RecordActivations(self):
        """Records the current activations"""
        if self.__hlength > 0:
            self.__history.append(np.copy(self.activations))
            if len(self.__history) > self.__hlength:
                self.__history.popleft()
        
    def AddIncomingProjection(self, proj):
        """Adds an incoming projection"""
        if proj not in self.incomingProjections:
            self.incomingProjections.append(proj)
        
    def RemoveIncomingProjection(self, proj):
        """Removes an incoming projection"""
        if proj in self.incomingProjections:
            self.incomingProjections.remove(proj)

    def AddOutgoingProjection(self, proj):
        """Adds an outgoing projection"""
        if proj not in self.outgoingProjections:
            self.outgoingProjections.append(proj)
        
    def RemoveOutgoingProjection(self, proj):
        """Removes an outgoing projection"""
        if proj in self.outgoingProjections :
            self.outgoingProjections.remove(proj)

    def SetClamped(self, clamp):
        self.__clamped=clamp

    def GetClamped(self):
        return self.__clamped

    def ConnectTo(self, group):
        """Creates a projection from self to a target Group"""
        return Projection(self, group)

    def SetKWTAFunction(self, func):
        self.__kwtaFunction=func

    def GetKWTAFunction(self):
        return self.__kwtaFunction


def kwta1(array, k):
    """Simple k-WTA function"""
    if k > 0 and k < array.size:
        F = array.flatten()
        L = zip(range(F.size), F)
        L.sort(key=lambda x: x[1])
        for i, j in L[:-k]:
            F[i]=0
        return F.reshape(array.shape)
    else:
        return array

def boltzmann_kwta(array, k, tau=0.01):
    """Boltzmann-like k-WTA function"""
    if k > 0 and k < array.size:
        F = array.flatten()
        i = Boltzmann(F, tau)
        for j in xrange(F.size):
            if j != i:
                F[j]=0
        return F.reshape(array.shape)
    else:
        return array


## ---------------------------------------------------------------- ##
## PROJECTIONS
## ---------------------------------------------------------------- ##

class Projection(NeuralObject, ContextContainer):
    """A basic projection between two groups"""
    def __init__(self, groupFrom=None, groupTo=None,
                 name=None, learningFunction=None,
                 context=None):
        """Initializes a projection"""
        if name is None:
            self.name=GenTemp("Projection-")
        else:
            self.name=name
            
        self.groupFrom        = groupFrom
        self.groupTo          = groupTo
        self.weights          = np.array(0)
        self.mask             = np.array(0)
        #self.SetLearningFunction(learningFunction)
        self.learningFunction = learningFunction
        self.learningRate     = 0
        self.learningEnabled  = False
        
        self.SetContext(context)
                
        if groupFrom is not None and groupTo is not None:
            self.weights = np.zeros((groupTo.size, groupFrom.size))
            self.mask    = np.ones((groupTo.size, groupFrom.size))

        if groupFrom is not None:
            groupFrom.AddOutgoingProjection(self)

        if groupTo is not None:
            groupTo.AddIncomingProjection(self)

    def __repr__(self):
        return "<%s%s:%s-->%s>" % (self.name, self.weights.shape,
                                   self.groupFrom, self.groupTo)
    
    def PropagateThrough(self):
        w   = self.weights * self.mask
        res = np.dot(w, self.groupFrom.activations)
        self.groupTo.inputs+=res

    def Learn(self):
        """Invokes the internal learning function"""
        if self.learningEnabled and self.learningFunction != None:
            self.learningFunction(self, context=self.GetContext())

## ---------------------------------------------------------------- ##
## CIRCUIT
## ---------------------------------------------------------------- ##

class Circuit(NeuralObject, ParametrizedObject):
    """An abstraction for a neural circuit"""

    def __init__(self):
        """Initializes the circuit"""
        NeuralObject.__init__(self)
        ParametrizedObject.__init__(self)
        self.__groups          = []
        self.__input           = []
        self.__output          = []
        self.__recordMode      = PY_NN_RECORD_FINAL
        self.__updateListeners = []
        self.__learnListeners  = []

        
    def GetGroups(self):
        """Returns the groups in a network"""
        return self.__groups

    def AddGroup(self, group):
        """Adds a new group in a circuit"""
        if not group in self.__groups:
            self.__groups.append(group)

    def AddGroups(self, groups):
        """Adds a list of new groups to a circuit"""
        for x in groups:
            self.AddGroup(x)

    def RemoveGroup(self, g):
        """
        Removes a group from the circuit.  Note that this will remove
        all the incoming and outgoing projections as well.
        """
        if g in self.__groups:
            # --- Remove from all the internal projections -----------
            P = self.GetProjections()
            gIn  = [p for p in P if p.groupTo == g]
            for x, p in zip([p.groupFrom for p in gIn], gIn):
                print "In from", x, p
                x.RemoveOutgoingProjection(p)
                            
            gOut = [p for p in P if p.groupFrom == g] 
            for x, p in zip([p.groupTo for p in gOut], gOut):
                print "Out to, ", x, p
                x.RemoveIncomingProjection(p)
            
            
            # --- Remove from all the internal lists of groups -------
            self.__groups.remove(g)
            if g in self.__input:
                self.RemoveInput(g)
            if g in self.__output:
                self.RemoveOutput(g)
            
                

    def SetInput(self, group):
        """Sets a groups as the input layer"""
        if group in self.__groups:
            self.__input.append(group)

    def RemoveInput(self, group):
        """
        Removes a group from the list of inputs. The group is *not*
        removed from the circuit.
        """ 
        if group in self.__input:
            self.__input.remove(group)

    def GetInput(self):
        """
        Returns the list of groups that are being used as input groups
        in the circuit.
        """
        return self.__input
        
    def SetOutput(self, group):
        """
        Sets a group as part of the output layer.
        """
        if group in self.__groups:
            self.__output.append(group)

    def RemoveOutput(self, group):
        """
        Removes a group from the list of the output groups. The group
        is not removed from the circuit.
        """
        if group in self.__output:
            self.__output.remove(group)

    def GetOutput(self):
        return self.__output

    def GetProjections(self, exclusive=True):
        """Returns all the projections within the circuit"""
        p  = []
        for x in self.__groups:
            p.extend(x.incomingProjections)
            p.extend(x.outgoingProjections)
        res   = list(set(p))
        if exclusive:
            res = [x for x in res if x.groupTo in self.__groups and
                   x.groupFrom in self.__groups]
        return res

    def SetInputActivations(self, inputs, clamped=False):
        if len(inputs) == len(self.GetInput()):
            for i, a in zip(self.GetInput(), inputs):
                i.SetActivations(a)
                if clamped:
                    i.SetClamped(clamped)

    def SetRecordMode(self, mode):
        if mode in [PY_NN_RECORD_NEVER, PY_NN_RECORD_FINAL, PY_NN_RECORD_ALWAYS]:
            self.__recordMode = mode
        
    def GetRecordMode(self):
        return self.__recordMode


    def Propagate(self, verbose=False, record=False):
        """Updates all the groups in a network"""
        nodes   = [x for x in self.__groups]
        visited = []
        bag     = deque(self.__input)
        current=None
        while len([x for x in bag if not x in visited]) > 0:
            # Visit the first node.
            current=bag.popleft()
            if verbose:
                print current.name,
            current.Update(record=record)
            visited.append(current)

            # Updates the list of future nodes
            children=[x.groupTo for x in current.outgoingProjections
                      if x.groupTo not in bag]
            
            bag.extend(children)
            bag=list(set(bag))  # Removes duplicates
            bag=deque([x for x in bag if not x in visited])
        if verbose:
            print ""

    ## ---------------------------------------------------------------
    ## UPDATE FUNCTION
    ##
    ## Update function. Common tricks:
    ##
    ## 1. To check for stable patterns only in the output groups/
    ##    layers:  x.Update(nodes=x.GetOuput())
    ##
    ## 2. To propagate and update nodes only once:
    ##    x.Update(max_epochs=1)
    ##   
    def Update(self, error=1e-6,
               min_epochs=1, max_epochs=1e4,
               nodes=None, verbose=True):
        """Recursively updates a network until it's stable"""
        if nodes is None:
            nodes=self.__groups

        A = {x : np.copy(x.activations) for x in nodes}
        c = 0     # Counter (number of epochs where E < error)
        e = 0     # Current epoch
        E = 0.0   # Current error E.
                
        # If the record modality is 'Final' only, we go
        # thorugh the groups one more time to record
        record = False
        if self.__recordMode == PY_NN_RECORD_FINAL:
            for g in self.__groups:
                g.RecordActivations()
        
        # If the record modality is 'Always', we set a flag
        # to True and use it when we propagate activation
        elif self.__recordMode == PY_NN_RECORD_ALWAYS:
            record = True


        while c < min_epochs and e < max_epochs:
            self.Propagate(verbose, record=record)
            e += 1
            E  = 0.0
            for g, a in A.items():
                E += Error(a, g.activations)
                if E <= error:
                    c += 1
                else:
                    A = {x : np.copy(x.activations) for x in nodes}
                    c = 0
            if verbose:
                print "[%d] %s" % (e, E)

        # --- Calls functions to be notified upon update
        
        for func in self.__updateListeners:
            func(self, e)
        
        # --- Return the number of epochs --------
                    
        return e
            
    def Learn(self):
        """Apply learning functions to projections"""
        N = []
        G = self.GetGroups()
        for g in G:
            P = g.incomingProjections + g.outgoingProjections
            P = [p for p in P if p not in N]
            P = [p for p in P if p.groupTo in G and p.groupFrom in G]
            for p in P:
                p.Learn()
                if p not in N:
                    N.append(p)

    def GetGroupByName(self, name):
        """Returns the group identified by name"""
        G = self.GetGroups()
        R = [x for x in G if x.name == name]
        if len(R) == 0:
            # --- Will return None (no exception thrown) 
            return None
        else:
            # --- Will return only the first one
            return R[0]

    def GetProjectionByName(self, name):
        """Returns the projection identified by name"""
        P = self.GetGroups()
        R = [x for x in P if x.name == name]
        if len(R) == 0:
            # --- Will return None (no exception thrown) 
            return None
        else:
            # --- Will return only the first one
            return R[0]


    def GetProjectionsBetweenGroups(self, g1, g2):
        """Returns the projections between two groups"""
        if g1 in self.__groups and g2 in self.__groups:
            P = g1.outgoingProjections
            R = [x for x in P if x.groupTo == g2]
            return R
        else:
            msg = "Group not existing in circuit %s" % self
            raise NeuralException([g1, g2], msg)

    def GetUpdateListeners(self):
        """Returns a list of the current update listeners"""
        return self.__updateListeners

    def AddUpdateListener(self, listener):
        """Adds a new listener function to be involked on updates"""
        if listener not in self.__updateListeners:
            self.__updateListeners.append(listener)

    def RemoveUpdateListener(self, listener):
        """Removes a function from the list of functions to be called on update""" 
        if listener in self.__updateListeners:
            self.__updateListeners.remove(listener)

    def GetGroupDepth(self, group, fromTop=True):
        """
        Calculates depth of a given group as a node in the circuit's tree.
        By default, depth is calculated top-down, considering the input
        nodes as the root. When the parameter "fromTop" is False, however,
        depth is calculated bottom-up, starting from the output layers
        """
        if group in self.GetGroups():
            if fromTop:
                I = self.GetInput()
                if len(I) > 0:
                    V = []        # visited nodes
                    N = [group]   # current nodes
                    d = 0         # minimum estimated depth
                    Searching = True
                    while Searching:
                        if any([x in I for x in N]):
                            Searching = False
                            
                        else:
                            d += 1
                            V.extend(N)
                            S = []
                            for n in N:
                                C = [x.groupFrom for x in n.incomingProjections]
                                C = [c for c in C if c not in V]
                                S.extend(C)
                            N = S
                        
                    return d
                else:
                    raise NeuralException(group, "No inputs specified for circuit")
                
            else:
                O = self.GetOutput()
                if len(O) > 0:
                    V = []        # visited nodes
                    N = [group]   # current nodes
                    d = 0         # minimum estimated depth
                    Searching = True
                    while Searching:
                        if any([x in O for x in N]):
                            Searching = False
                            
                        else:
                            d += 1
                            V.extend(N)
                            S = []
                            for n in N:
                                C = [x.groupTo for x in n.outgoingProjections]
                                C = [c for c in C if c not in V]
                                S.extend(C)
                            N = S
                        
                    return d
                else:
                    raise NeuralException(group, "No outputs specified for circuit")
        else:
            raise NeuralException(group, "Cannot calcate group's depth")
        
        


## ---------------------------------------------------------------- ##
## TEST
## ---------------------------------------------------------------- ##

## DEMO NETWORK
g1 = Group(200)
g2 = Group(100)
g3 = Group(100)
g4 = Group(80)

p1 = Projection(g1, g2)
p1.weights=np.random.random((100,200))/5 - .1
p2 = Projection(g2, g3)
p2.weights=np.random.random((100,100))/5 - .1
p3 = Projection(g3, g4)
p3.weights=np.random.random((80,100))/5 - .1
p4 = Projection(g3, g2)
p4.weights=np.random.random((100,100))/5 -.1
p5 = Projection(g4, g3)
p5.weights=np.random.random((100,80))/5 - .1

pattern1 = Step(np.random.random((200,1)), .5)
g1.activations=np.array(pattern1)
g1.SetClamped(True)

c=Circuit()
c.AddGroups([g1, g2, g3, g4])
c.SetInput(g1)
c.SetOutput(g3)

## A more complex example. Mockup basal ganglia
##
b1 = Group(1000, name="Cortex")
b2 = Group(500, name="Str")   # Striatum
b3 = Group(200, name="Gpi/Snr")   # Gpi/SNr
b4 = Group(300, name="Gpe")   # Gpe
b5 = Group(100, name="Thal")   # Thal
b6 = Group(50, name="STN")     # STN

bp1 = Projection(b1, b2)
bp1.weights=np.random.random((500,1000))/5 - .1

bp2 = Projection(b2, b3)
bp2.weights=np.random.random((200,500))/5 - .1

bp3 = Projection(b2, b4)
bp3.weights=np.random.random((300,500))/5 - .1

bp4 = Projection(b4, b6)
bp4.weights=np.random.random((50,300))/5 - .1

bp5 = Projection(b6, b3)
bp5.weights=np.random.random((200,50))/5 - .1

bp6 = Projection(b3, b5)
bp6.weights=np.random.random((100,200))/5 - .1

pattern2 = Step(np.random.random((1000,1)), .5)
b1.activations=np.array(pattern2)
b1.SetClamped(True)

bg=Circuit()
bg.AddGroups([b1, b2, b3, b4, b5, b6])
bg.SetInput(b1)
bg.SetOutput(b5)

### A 3-layer network for CHL
###
h1=Group(20, name="chl-1")
h2=Group(20, name="chl-2")
h3=Group(20, name="chl-3")

n=Circuit()
n.AddGroups([h1, h2, h3])
n.SetInput(h1)
hp1=Projection(h1, h2)
hp1.weights=np.random.random((20,20))/5 - .1

hp2=Projection(h2, h3)
hp2.weights=np.random.random((20,20))/5 - .1

hp3=Projection(h3, h2)
hp3.weights=np.random.random((20,20))/5 - .1

input1 = Step(np.random.random((20,1)), .5)
input2 = Step(np.random.random((20,1)), .5)

target1 = Step(np.random.random((20,1)), .5)
target2 = Step(np.random.random((20,1)), .5)


h1.activations=input1
h1.SetClamped(True)
n.SetOutput(h3)