## ---------------------------------------------------------------- ##
## NEURAL.PY
## ---------------------------------------------------------------- ##
## Defines the basic classes for simulating neural networks.
## ---------------------------------------------------------------- ##

import math  as m
import numpy as np
import copy
import operator
from collections import deque
from selection   import Boltzmann
from basic       import SPyNEObject, ParametrizedObject, ContextContainer
from naming      import GenTemp

## ---------------------------------------------------------------- ##
## CONSTANTS
## ---------------------------------------------------------------- ##

PY_NN_RECORD_NEVER  = 0     # Never record activations
PY_NN_RECORD_FINAL  = 1     # During updates, record only at the end 
PY_NN_RECORD_ALWAYS = 2     # During updates, record all the intermediates


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
    """
Sum of squares---Classical error function for a vector and its
target value
    """
    return np.sum((a1 - a2) ** 2)


## ---------------------------------------------------------------- ##
## EXCEPTIONS
## ---------------------------------------------------------------- ##
class NeuralException( Exception ):
    """An exception occurring in a Neural object"""
    def __init__(self, source=None, message=None):
        Exception.__init__(self)
        self.source  = source
        self.message = message

## ---------------------------------------------------------------- ##
## BASIC NEURAL OBJECTS
## ---------------------------------------------------------------- ##

class NeuralObject (SPyNEObject):
    """The root of all objects"""
    def __init__( self ):
        self.name = GenTemp("NeuralObject-")
            


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
        self.ActivationDerivative = activationDerivative
        self.SetUpdateFunction(updateFunction)
        self._size        = size
        self._activations = np.zeros((size, 1))
        self._geometry    = copy.copy(self.activations.shape)
        self._baselines   = np.zeros((size, 1))
        self._mask        = np.ones((size, 1))
        self._thresholds  = np.zeros((size, 1))
        self._inputs      = np.zeros((size, 1))
                
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

    # New code prevents broadcasting errors in NumPy.
    # Requires larger set of compatibility checks, but also
    # makes the code much more tolerant of possible imperfections
    # in assigning inputs, activations, masks, thresholds etc.

    def ArrayCompatible(self, array):
        """
Checks whether a given array is compatible with the internal
models.
        """
        if len(array.shape) == 1:
            return self._activations.shape[0] == array.shape[0]
        
        # If the array is a matrix 
        elif len(array.shape) == 2:
            return  set(array.shape) == set(self.activations.shape)
        
        else:
            return False

    @property
    def size(self):
        """Read-only property size (the size of the group)"""
        return self._size
    
    @property
    def geometry(self):
        return self._geometry
    
    @geometry.setter
    def geometry(self, shape):
        """Sets the geometry (IFF geometry is compatible with size)"""
        if len(shape) <= 2 and \
            reduce(operator.mul, shape) == self.size:
            self._geometry = shape
        else:
            raise NeuralException(self, "Incompatible geometry")
    

    @property 
    def activations(self):
        """Returns the activations"""
        return self._activations
    
    @activations.setter
    def activations(self, activations):
        """Sets the array of activations"""
        self.SetActivations(activations, clamped=False)

    def SetActivations(self, activations, clamped=False):
        """Sets the activation values of the group's neurons"""
        if self.ArrayCompatible(activations):
            newvals = np.copy(activations)
            newvals.shape = self.activations.shape  # Reformat activatipns
            self._activations = newvals
            if clamped:
                self.SetClamped(True)
        else:
            #print "Nonmatch", self.activations.shape, activations.shape
            raise NeuralException(self, "Mismatching activations")

    def GetActivations(self):
        """Returns the activation values of the neurons"""
        return self.activations

    @property
    def inputs(self):
        return self._inputs
    
    @inputs.setter
    def inputs(self, inputs):
        if self.ArrayCompatible(inputs):
            newvals = np.copy(inputs)
            newvals.shape = self.inputs.shape  # Reformat inputs
            self._inputs = newvals
        else:   
            raise NeuralException(self, "Mismatching inputs")   
 
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, mask):
        if self.ArrayCompatible(mask):
            newvals = np.copy(mask)
            newvals.shape = self.mask.shape  # Reformat masks
            self._mask = newvals
        else:   
            raise NeuralException(self, "Mismatching mask")   
 

    @property
    def thresholds(self):
        return self._thresholds
    
    @thresholds.setter
    def thresholds(self, thresholds):
        if self.ArrayCompatible(thresholds):
            newvals = np.copy(thresholds)
            newvals.shape = self.thresholds.shape  # Reformat thresholds
            self._thresholds = newvals
        else:   
            raise NeuralException(self, "Mismatching thresholds")   
 
    @property
    def baselines(self):
        return self._baselines
    
    @inputs.setter
    def baselines(self, baselines):
        if self.ArrayCompatible(baselines):
            newvals = np.copy(baselines)
            newvals.shape = self.baselines.shape  # Reformat activatipns
            self._baselines = newvals
        else:   
            raise NeuralException(self, "Mismatching baselines")   
 

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
        self.learningFunction = learningFunction
        self.learningRate     = 0
        self.learningEnabled  = False
        
        self.SetContext(context)
                
        if self.groupFrom is not None and self.groupTo is not None:
            self.weights = np.zeros((groupTo.size, groupFrom.size))
            self.mask    = np.ones((groupTo.size, groupFrom.size))
            #print "Ws = %s and M = %s" % (self.weights.shape, self.mask.shape)
           
        if groupFrom is not None:
            groupFrom.AddOutgoingProjection(self)

        if groupTo is not None:
            groupTo.AddIncomingProjection(self)

    def __repr__(self):
        """A string representing a projection"""
        return "<%s%s:%s --> %s>" % (self.name, self.weights.shape,
                                     self.groupFrom, self.groupTo)
    
    def PropagateThrough(self):
        """
Updates the activation values of the destination group after propagating
the activation of the source group through the weight matri
        """
        w   = self.weights * self.mask
        res = np.dot(w, self.groupFrom.activations)
        self.groupTo.inputs += res

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
        self._groups          = []
        self._input           = []
        self._output          = []
        self._recordMode      = PY_NN_RECORD_FINAL
        self._updateListeners = []
        self._learnListeners  = []

        
    def GetGroups(self):
        """Returns the groups in a network"""
        return self._groups

    def AddGroup(self, group):
        """Adds a new group in a circuit"""
        if not group in self._groups:
            self._groups.append(group)

    def AddGroups(self, groups):
        """Adds a list of new groups to a circuit"""
        for x in groups:
            self.AddGroup(x)

    def RemoveGroup(self, g):
        """
        Removes a group from the circuit.  Note that this will remove
        all the incoming and outgoing projections as well.
        """
        if g in self._groups:
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
            self._groups.remove(g)
            if g in self._input:
                self.RemoveInput(g)
            if g in self._output:
                self.RemoveOutput(g)
            
                

    def SetInput(self, group):
        """Sets a groups as the input layer"""
        if group in self._groups:
            self._input.append(group)

    def RemoveInput(self, group):
        """
        Removes a group from the list of inputs. The group is *not*
        removed from the circuit.
        """ 
        if group in self._input:
            self._input.remove(group)

    def GetInput(self):
        """
        Returns the list of groups that are being used as input groups
        in the circuit.
        """
        return self._input
        
    def SetOutput(self, group):
        """
        Sets a group as part of the output layer.
        """
        if group in self._groups:
            self._output.append(group)

    def RemoveOutput(self, group):
        """
        Removes a group from the list of the output groups. The group
        is not removed from the circuit.
        """
        if group in self._output:
            self._output.remove(group)

    def GetOutput(self):
        return self._output

    def GetProjections(self, exclusive=True):
        """Returns all the projections within the circuit"""
        p  = []
        for x in self._groups:
            p.extend(x.incomingProjections)
            p.extend(x.outgoingProjections)
        res   = list(set(p))
        if exclusive:
            res = [x for x in res if x.groupTo in self._groups and
                   x.groupFrom in self._groups]
        return res

    def SetInputActivations(self, inputs, clamped=False):
        if len(inputs) == len(self.GetInput()):
            for i, a in zip(self.GetInput(), inputs):
                i.SetActivations(a)
                if clamped:
                    i.SetClamped(clamped)

    def SetRecordMode(self, mode):
        if mode in [PY_NN_RECORD_NEVER, PY_NN_RECORD_FINAL, PY_NN_RECORD_ALWAYS]:
            self._recordMode = mode
        
    def GetRecordMode(self):
        return self._recordMode


    def Propagate(self, verbose=False, record=False):
        """Updates all the groups in a network"""
        nodes   = [x for x in self._groups]
        visited = []
        bag     = deque(self._input)
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
            nodes=self._groups

        A = {x : np.copy(x.activations) for x in nodes}
        c = 0     # Counter (number of epochs where E < error)
        e = 0     # Current epoch
        E = 0.0   # Current error E.
                
        # If the record modality is 'Final' only, we go
        # thorugh the groups one more time to record
        record = False
        if self._recordMode == PY_NN_RECORD_FINAL:
            for g in self._groups:
                g.RecordActivations()
        
        # If the record modality is 'Always', we set a flag
        # to True and use it when we propagate activation
        elif self._recordMode == PY_NN_RECORD_ALWAYS:
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
        
        for func in self._updateListeners:
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
        if g1 in self._groups and g2 in self._groups:
            P = g1.outgoingProjections
            R = [x for x in P if x.groupTo == g2]
            return R
        else:
            msg = "Group not existing in circuit %s" % self
            raise NeuralException([g1, g2], msg)

    def GetUpdateListeners(self):
        """Returns a list of the current update listeners"""
        return self._updateListeners

    def AddUpdateListener(self, listener):
        """Adds a new listener function to be involked on updates"""
        if listener not in self._updateListeners:
            self._updateListeners.append(listener)

    def RemoveUpdateListener(self, listener):
        """Removes a function from the list of functions to be called on update""" 
        if listener in self._updateListeners:
            self._updateListeners.remove(listener)

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