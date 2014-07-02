## Variability in Decision Making

import math, random
import selection as sel
import neural
reload(neural)
from   neural       import *
from   numpy        import dot
from   basalganglia import *


## ---------------------------------------------------------------- ##
## MODEL 1 -- Simple
## ---------------------------------------------------------------- ##

## --- The nuclei --------------------------------

sn               = Group(6,  name="SN",         # Striatonigral
                         activationFunction=np.vectorize(lambda x:
                                                          STanh_plus(x, gain=1)))
sp               = Group(6,  name="SP",         # Striatopallidal
                         activationFunction=np.vectorize(lambda x:
                                                          STanh_plus(x, gain=1)))
context          = Group(9,  "Context")    # Context (options) encoding
#action           = group(18, name=GenTemp("Action-"))     # Action encoding
snr              = Group(6,  name="SNr/GPi-",    # SNr/GPi
                         activationFunction=np.vectorize(lambda x:
                                                          STanh_plus(x, gain=1)))
tans             = Group(1,  name="TAN")        # TANs
da               = Group(1,  name="Da",         # Dopamine (SNc/VTA)
                         activationFunction=Linear)

## --- Projections -------------------------------

c2sn             = context.ConnectTo(sn)
c2sp             = context.ConnectTo(sp)
c2tans           = context.ConnectTo(tans)
tans2sn          = tans.ConnectTo(sn)
tans2sp          = tans.ConnectTo(sp)
da2sn            = da.ConnectTo(sn)
da2sp            = da.ConnectTo(sp)
da2tans          = da.ConnectTo(tans)
sn2snr           = sn.ConnectTo(snr)
sp2snr           = sp.ConnectTo(snr)

## new feedback
snr2sp           = snr.ConnectTo(sp)
snr2sn           = snr.ConnectTo(sn)

c2sn.weights     = np.random.random(c2sn.weights.shape)/10.0
c2sp.weights     = np.random.random(c2sp.weights.shape)/10.0
c2tans.weights   = np.zeros(c2tans.weights.shape)             # No context modulation
tans2sn.weights  = np.random.random(tans2sn.weights.shape)/10.0
tans2sp.weights  = np.random.random(tans2sp.weights.shape)/10.0
da2sn.weights    = np.ones(da2sn.weights.shape)/10.0
da2sp.weights    = np.ones(da2sp.weights.shape)/-10.0
da2tans.weights  = np.random.random(da2tans.weights.shape)/10
sn2snr.weights   = np.ones(sn2snr.weights.shape)*np.eye(sn.size)
sp2snr.weights   = np.ones(sp2snr.weights.shape)*np.eye(sp.size)*-1
snr2sn.weights   = np.ones(sn2snr.weights.shape)*np.eye(sn.size)*0.2
snr2sp.weights   = np.ones(sn2snr.weights.shape)*np.eye(sn.size)*0.2


## --- The PVLV system -------------------------------------------- ##

PVe              = Group(1, name="PVe")
PVi              = Group(1, name="PVi")
LVe              = Group(1, name="LVe")
LVi              = Group(1, name="LVi")

c2PVi            = context.ConnectTo(PVi)
c2LVe            = context.ConnectTo(LVe)
c2LVi            = context.ConnectTo(LVi)
PVe2da           = PVe.ConnectTo(da)
LVe2da           = LVe.ConnectTo(da)
PVi2da           = PVi.ConnectTo(da)
LVi2da           = LVi.ConnectTo(da)

c2sn.mask        = np.dot(np.ones((6,1)), np.array([[1,1,1,0,0,0,0,0,0]]))
c2sp.mask        = np.dot(np.ones((6,1)), np.array([[1,1,1,0,0,0,0,0,0]]))

## Weights. Primary reinforceres have a fixed W = [1].

c2PVi.weights    = np.random.random(c2LVe.weights.shape) * 0.01
c2LVe.weights    = np.random.random(c2LVe.weights.shape) * 0.01
c2LVi.weights    = np.random.random(c2LVi.weights.shape) * 0.01
PVe2da.weights   = np.ones((1,1))
LVe2da.weights   = np.ones((1,1))
PVi2da.weights   = np.ones((1,1))*-1
LVi2da.weights   = np.ones((1,1))*-1

## Setting up random activations (for visualization)

context.SetActivations(np.random.random((context.size, 1)))
tans.SetActivations(np.random.random((tans.size, 1)))
da.SetActivations(np.random.random((da.size, 1)))
sn.SetActivations(np.random.random((sn.size, 1)))

## Cortical input now 

input1 = np.zeros(context.activations.shape)
input1[0,0]=1
context.SetActivations(input1)
context.SetClamped(True)

snr.kwta=True
snr.SetKWTAFunction(lambda x: boltzmann_kwta(x, k=1))

## ---------------------------------------------------------------- ##
## PVLV LEARNING SYSTEM
## ---------------------------------------------------------------- ##
## The PVLV algorithm is a reinforcement learning algorithm 
## described by O'Reilly, Frank, Hazy, & Wats (2007).  It has the
## advantages of being biologically plausible and requiring minimal
## computations to control dopamine response.
## ---------------------------------------------------------------- ##


e1      = 0.050    # Learning rate for PVi and LVe systems
e2      = 0.001    # Learning rate for LVi (taken from O'Reilly et al, 2007)
tmax    = 0.650    # Thrrshold above which the PV system is considered active
tmin    = 0.250    # Threshold below which the PV system is considered active

def PViLearningRule(p, context=None):
    """The PVi algorithm learning rule"""
    d  = PVe.activations - PVi.activations
    X  = p.groupFrom.activations
    W  = p.weights
    dW = e1*d*X.T
    #print d, X, W
    if TRACE_PVLV:
        print "PVi delta: %s" % dW
    W += dW

## See Eq. (5), (6), (7)
def LVeLearningRule(p, context=None):
    """The LVe learning rule"""
    PVfilter = PVi.activations[0,0] > tmax or PVe.activations[0,0] > tmax \
        or PVi.activations[0,0] < tmin or PVe.activations[0,0] < tmin
    # Learning occurs only if PVi or PVe are active
    if PVfilter:
        d  = PVe.activations - LVe.activations
        W  = p.weights
        X  = p.groupFrom.activations
        dW = d*e1*X.T
        if TRACE_PVLV:
            print "LVe delta: %s" % dW
        W += dW

def LViLearningRule(p, context=None):
    """The LVi learning rule"""
    PVfilter = PVi.activations[0,0] > tmax or PVe.activations[0,0] > tmax \
        or PVi.activations[0,0] < tmin or PVe.activations[0,0] < tmin

    if PVfilter:
        d  = LVe.activations - LVi.activations
        W  = p.weights
        X  = p.groupFrom.activations
        dW = d*e2*X.T
        if TRACE_PVLV:
            print "LVi delta: %s" % dW
        W += dW


c2PVi.learningFunction = PViLearningRule
c2LVe.learningFunction = LVeLearningRule
c2LVi.learningFunction = LViLearningRule

c2PVi.learningEnabled  = True
c2LVe.learningEnabled  = True
c2LVi.learningEnabled  = True

def LVUpdate(lv):
    """LV Update functions---considers only changes in input"""
    for p in lv.incomingProjections:
        g = p.groupFrom
        W = p.weights
        dX = g.activations = g.GetHistory()[-1]
        dI = np.dot(W, dX)
        lv.inputs+=dI

LVi.SetUpdateFunction(LVUpdate)
LVe.SetUpdateFunction(LVUpdate)

def PVLV_Update(g):
    GenericUpdate(g)
    if TRACE_PVLV:
        print "  %s: input=%s, X=%s" % (g, g.inputs, g.activations)
#LVe.SetUpdateFunction(PVLV_Update)
#LVi.SetUpdateFunction(PVLV_Update)
PVi.SetUpdateFunction(PVLV_Update)
PVe.SetUpdateFunction(PVLV_Update)


## -------------------------------------------------------------------
## Dopamine's input is updated according to the following rule:
##
##              dPV, if PVe > Tmax or PVe > Tmax or PVi < Tmin or PVe < Tmin 
##   I = dLV + 
##              0    otherwise
##
def DopamineUpdate(da):
    da.ClearInputs()
    PVfilter = PVi.activations[0,0] > tmax or PVe.activations[0,0] > tmax \
        or PVi.activations[0,0] < tmin or PVe.activations[0,0] < tmin

    LV = [p for p in da.incomingProjections if p.groupFrom.name.startswith("LV")]
    PV = [p for p in da.incomingProjections if p.groupFrom.name.startswith("PV")]
    for p in LV:
        p.PropagateThrough()
    
    if PVfilter:
        for p in PV:
            p.PropagateThrough()
    da.CalculateActivations()

    if TRACE_PVLV:
        print "  %s: input=%s, X=%s" % (PVe, PVe.inputs, PVe.activations)
    
    if TRACE_DA:
        print "  Da: input=%s, PVfil=%s, X=%s" % (da.inputs, PVfilter, da.activations)
da.SetUpdateFunction(DopamineUpdate)


## ---------------------------------------------------------------- ##
## STRIATAL LEARNING SYSTEM
## ---------------------------------------------------------------- ##
## The striatum learns only when dopamine is present, and its 
## learning rate is proportional to the amount of dopamine released
## (as in Ashby, 2007; and Stocco, Lebiere, & Anderson, 2010).
## Also, dopamine learning is proportional to the activation
## deltas in SN and SP neurons (as in O'Reilly & Frank, 2006).
## ---------------------------------------------------------------- ##

r   = 1.5
da1 = 1.5
da2 = 1.0

def SN_Update(group):
    group.ClearInputs()
    for p in group.incomingProjections:
        if p.groupFrom.name.startswith("Da"):# and PVe.activations[0,0]!=0.50:
            #D1 = dot(p.weights, p.groupFrom.activations)
            #D2 = group.activations*p.groupFrom.activations
            D1 = dot(p.weights, p.groupFrom.activations)
            D2 = group.GetHistory()[-1]*p.groupFrom.activations
            
            if group.name.startswith("SP"):
                #print 'reversing'
                D2*=-1
            dX = D1/2 + D2/2
            dX *= Step(snr.activations)
            if TRACE_UPDATE:
                print "  Group: %s,\n  Dopa D1: %s\n  D2: %s\n  dX: %s" % (group, D1, D2, dX)
            group.inputs += dX
        else:
            p.PropagateThrough()
    #print "Inputs", group.inputs
    group.CalculateActivations()
    if TRACE_UPDATE:
        print "  Group: %s,\n  Inputs=%s\n  X=%s" % (group, group.inputs, group.activations.T)

sn.SetUpdateFunction(SN_Update)
sp.SetUpdateFunction(SN_Update)

def SN_LearningRule(p, context=None, rate=da1):
    """The Striatonigral learning rule"""
    #D1 = da.activations
    #D0 = da.GetHistory()[-1]
    #d   = da1 - da0
    D   = abs(np.max(da.activations))
    Y1  = p.groupTo.activations
    Y0  = p.groupTo.GetHistory()[-1]
    W   = p.weights
    M   = p.mask
    Y   = Y1 - Y0
    X   = np.copy(p.groupFrom.activations)
    dW  = r*D*np.dot(Y, X.T)
    dW *= M
    if TRACE_STRIATUM_LEARNING:
        print "  P=%s,\n  Y=%s,\n  dW=%s" % (p, Y, dW)
    W  += dW
    
    

def SP_LearningRule(p, context=None):
    """The Striatopallidal learning rule"""
    return SN_LearningRule(p, context, rate=da2)

c2sn.learningFunction   = SN_LearningRule
c2sp.learningFunction   = SP_LearningRule

c2sn.learningEnabled    = True
c2sp.learningEnabled    = True



## ---------------------------------------------------------------- ##
## SETTING UP THE CIRCUIT
## ---------------------------------------------------------------- ##

M1 = Circuit()
M1.AddGroups([sn, sp, context, tans, da, snr, \
               PVe, PVi, LVe, LVi])
M1.SetInput(context)
M1.SetOutput(snr)


## ---------------------------------------------------------------- ##
## VISUALIZATION TRICKS
## ---------------------------------------------------------------- ##
context.geometry = (5, 3)
sn.geometry     = (3, 2)
sp.geometry     = (3, 2)

## TRACING AND DEBUGGING

TRACE_UPDATE             = False
TRACE_PVLV               = False
TRACE_PVLV_LEARNING      = False
TRACE_STRIATUM_LEARNING  = False
TRACE_DA                 = True

def TestRun():
    context.SetActivations(input)
    M1.Update()
    snr.SetClamped(True)
    #ofc.SetActivations(-1*np.ones((1,1)))
    M1.Update()
    M1.Learn()
