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

sn               = Group(6,  name=GenTemp("SN-"),         # Striatonigral
                         activationFunction=np.vectorize(lambda x:
                                                          STanh_plus(x, gain=1)))
sp               = Group(6,  name=GenTemp("SP-"),         # Striatopallidal
                         activationFunction=np.vectorize(lambda x:
                                                          STanh_plus(x, gain=1)))
cortex           = Group(15, name=GenTemp("Cortex-"))     # Cortex
snr              = Group(6,  name=GenTemp("SNr/GPi-"),    # SNr/GPi
                         activationFunction=np.vectorize(lambda x:
                                                          STanh_plus(x, gain=1)))
tans             = Group(1,  name=GenTemp("TAN-"))        # TANs
da               = Group(1,  name=GenTemp("Da-"),         # Dopamine (SNc/VTA)
                         #activationFunction=np.vectorize(lambda x: SSigmoid(x, gain=-2)))
                         activationFunction=np.vectorize(math.tanh))
ofc              = Group(1,  name=GenTemp("OFC-"))        # OFC

da.baselines     = np.ones(da.activations.shape)/2

## --- Projections -------------------------------

c2sn             = cortex.ConnectTo(sn)
c2sp             = cortex.ConnectTo(sp)
c2tans           = cortex.ConnectTo(tans)
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

c2sn.weights     = np.random.random(c2sn.weights.shape)/10
c2sp.weights     = np.random.random(c2sp.weights.shape)/10
c2tans.weights   = np.zeros(c2tans.weights.shape)             # No context modulation
tans2sn.weights  = np.random.random(tans2sn.weights.shape)/10
tans2sp.weights  = np.random.random(tans2sp.weights.shape)/10
da2sn.weights    = np.ones(da2sn.weights.shape)/10
da2sp.weights    = np.ones(da2sp.weights.shape)/-10
da2tans.weights  = np.random.random(da2tans.weights.shape)/10
sn2snr.weights   = np.ones(sn2snr.weights.shape)*np.eye(sn.size)
sp2snr.weights   = np.ones(sp2snr.weights.shape)*np.eye(sp.size)*-1
snr2sn.weights   = np.ones(sn2snr.weights.shape)*np.eye(sn.size)*0.2
snr2sp.weights   = np.ones(sn2snr.weights.shape)*np.eye(sn.size)*-0.2


## --- The PVLV system -------------------------------------------- ##

pve              = Group(1, name=GenTemp("PVe-"))
pvi              = Group(1, name=GenTemp("PVi-"))
lve              = Group(1, name=GenTemp("LVe-"))
lvi              = Group(1, name=GenTemp("LVi-"))


ofc2pve          = ofc.ConnectTo(pve)
ofc2pvi          = ofc.ConnectTo(pvi)
ofc2lve          = ofc.ConnectTo(lve)
c2lve            = cortex.ConnectTo(lve)
c2lvi            = cortex.ConnectTo(lvi)
pve2da           = pve.ConnectTo(da)
lve2da           = lve.ConnectTo(da)
pvi2da           = pvi.ConnectTo(da)
lvi2da           = lvi.ConnectTo(da)


## Weights. Primary reinforceres have a fixed W = [1].

ofc2pve.weights  = np.ones((1,1))
ofc2pvi.weights  = np.zeros((1,1))
ofc2lve.weights  = np.ones((1,1))
c2lve.weights    = np.random.random(c2lve.weights.shape) * 0.01
c2lvi.weights    = np.random.random(c2lvi.weights.shape) * 0.01
pve2da.weights   = np.ones((1,1))
lve2da.weights   = np.ones((1,1))
pvi2da.weights   = np.ones((1,1))*-1
lvi2da.weights   = np.ones((1,1))*-1

## Setting up random activations (for visualization)

cortex.SetActivations(np.random.random((cortex.size, 1)))
tans.SetActivations(np.random.random((tans.size, 1)))
da.SetActivations(np.random.random((da.size, 1)))
sn.SetActivations(np.random.random((sn.size, 1)))

## Cortical input now 

input1 = np.zeros(cortex.activations.shape)
input1[0,0]=1
cortex.SetActivations(input1)
cortex.SetClamped(True)

ofc.SetActivations(np.zeros(ofc.activations.shape))
ofc.SetClamped(True)

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


e1      = 0.05     # Learning rate for PV and LVe systems
e2      = 0.001    # Learning rate for LVi
tmax    = 0.51     # Thrshold above which the PV system is considerd active
tmin    = 0.49     # min Theta

def PViLearningRule(p, context=None):
    """The PVi algorithm learning rule"""
    #print "PVi: ", 
    d  = pve.activations - pvi.activations
    X  = p.groupFrom.activations
    W  = p.weights
    print d, X, W
    if TRACE_PVLV:
        print "PVi delta: %s" % (e1*d*X.T)
    W += e1*d*X.T

## See Eq. (5), (6), (7)
def LVeLearningRule(p, context=None):
    """The LVe learning rule"""
    #print "LVe", 
    PVfilter = pvi.activations[0,0] > tmax or pve.activations[0,0] > tmax or
               pvi.activations[0,0] < tmin or pve.activations[0,0] < tmin
    # Learning occurs only if PVi or PVe are active
    if PVfilter:
        d = pve.activations*lve.activations
        W = p.weights
        X = p.groupFrom.activations
        if TRACE_PVLV:
            print "LVe delta: %f" % d*e1*X.T
        W += d*e1*X.T

def LViLearningRule(p, context=None):
    """The LVi learning rule"""
    #print "LVi",
    PVfilter = pvi.activations[0,0] > tmax or pve.activations[0,0] > tmax or
               pvi.activations[0,0] < tmin or pve.activations[0,0] < tmin

    if PVfilter:
        d = lve.activations - lvi.activations
        W = p.weights
        X = p.groupFrom.activations
        if TRACE_PVLV:
            print "LVi delta: %s" % d*e2*X.T
        W += e2*d*X.T


ofc2pvi.learningFunction = PViLearningRule
c2lve.learningFunction   = LVeLearningRule
c2lvi.learningFunction   = LViLearningRule

ofc2pvi.learningEnabled  = True
c2lve.learningEnabled    = True
c2lvi.learningEnabled    = True

## ---------------------------------------------------------------- ##
## STRIATAL LEARNING SYSTEM
## ---------------------------------------------------------------- ##
## The striatum learns only when dopamine is present, and its 
## learning rate is proportional to the amount of dopamine released
## (as in Ashby, 2007; and Stocco, Lebiere, & Anderson, 2010).
## Also, dopamine learning is proportional to the activation
## deltas in SN and SP neurons (as in O'Reilly & Frank, 2006).
## ---------------------------------------------------------------- ##

r = 75

def SN_Update(group):
    group.ClearInputs()
    for p in group.incomingProjections:
        if p.groupFrom.name.startswith("Da") and ofc.activations[0,0]!=0:
            #D1 = dot(p.weights, p.groupFrom.activations)
            #D2 = group.activations*p.groupFrom.activations
            D1 = dot(p.weights, p.groupFrom.activations)
            D2 = group.GetHistory()[-1]*p.groupFrom.activations
            
            if group.name.startswith("SP"):
                #print 'reversing'
                D2*=-1
            dX= D1/2 + D2/2
            if TRACE_UPDATE:
                print "  Group: %s,\n  Dopa D1: %s\n  D2: %s\n  dX: %s" % (group, D1, D2, dX)
            group.activations += dX
        else:
            p.PropagateThrough()
    #print "Inputs", group.inputs
    group.CalculateActivations()
    if TRACE_UPDATE:
        print "  Group: %s,\n  Inputs=%s\n  X=%s" % (group, group.inputs, group.activations.T)

sn.SetUpdateFunction(SN_Update)
sp.SetUpdateFunction(SN_Update)

def SN_LearningRule(p, context=None):
    """The Striatonigral learning rule"""
    da1 = da.activations
    da0 = da.GetHistory()[-1]
    d   = da1 - da0
    Y1  = p.groupTo.activations
    Y0  = p.groupTo.GetHistory()[-1]
    #Y0  = p.groupTo.baselines
    W   = p.weights
    Y   = Y1 - Y0
    X   = p.groupFrom.activations
    dW  = r*d*np.dot(Y, X.T)
    if TRACE_STRIATUM_LEARNING:
        print "  P=%s,\n  Y=%s,\n  dW=%s" % (p, Y, dW)
    W  += dW
    
    

def SP_LearningRule(p, context=None):
    """The Striatopallidal learning rule"""
    return SN_LearningRule(p, context)

c2sn.learningFunction   = SN_LearningRule
c2sp.learningFunction   = SP_LearningRule

c2sn.learningEnabled    = True
c2sp.learningEnabled    = True

def DA_Update(da):
    GenericUpdate(da)
    if TRACE_DA:
        print "  Da: input=%s, X=%s" % (da.inputs, da.activations)
da.SetUpdateFunction(DA_Update)


def PVLV_Update(g):
    GenericUpdate(g)
    if TRACE_PVLV:
        print "  %s: input=%s, X=%s" % (g, g.inputs, g.activations)
lve.SetUpdateFunction(PVLV_Update)
lvi.SetUpdateFunction(PVLV_Update)
pvi.SetUpdateFunction(PVLV_Update)
pve.SetUpdateFunction(PVLV_Update)

## ---------------------------------------------------------------- ##
## SETTING UP THE CIRCUIT
## ---------------------------------------------------------------- ##

M1 = Circuit()
M1.AddGroups([sn, sp, cortex, tans, da, snr, ofc,
               pve, pvi, lve, lvi])
M1.SetInput(cortex)
M1.SetInput(ofc)
M1.SetOutput(snr)


## ---------------------------------------------------------------- ##
## VISUALIZATION TRICKS
## ---------------------------------------------------------------- ##
cortex.geometry = (5, 3)
sn.geometry     = (3, 2)
sp.geometry     = (3, 2)

## TRACING AND DEBUGGING

TRACE_UPDATE             = False
TRACE_PVLV               = False
TRACE_PVLV_LEARNING      = False
TRACE_STRIATUM_LEARNING  = False
TRACE_DA                 = True

def TestRun():
    cortex.SetActivations(input)
    M1.Update()
    snr.SetClamped(True)
    ofc.SetActivations(-1*np.ones((1,1)))
    M1.Update()
    M1.Learn()
