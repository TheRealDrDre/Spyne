## Variability in Decision Making

import math, random
import selection as sel
import neural
reload(neural)
from   neural       import *
from   basalganglia import *


## ---------------------------------------------------------------- ##
## MODEL 1 -- Simple
## ---------------------------------------------------------------- ##

## --- The nuclei --------------------------------

sn               = Group(6,  name=GenTemp("SN-"),         # Striatonigral
                         activationFunction=np.vectorize(lambda x:
                                                          SSigmoid(x, gain=-3)))
sp               = Group(6,  name=GenTemp("SP-"),         # Striatopallidal
                         activationFunction=np.vectorize(lambda x:
                                                          SSigmoid(x, gain=-2)))
cortex           = Group(36, name=GenTemp("Cortex-"))     # Cortex
snr              = Group(6,  name=GenTemp("SNr/GPi-"))    # SNr/GPi
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

#sn.kwta=False
#sn.SetKWTAFunction(lambda x: kwta1(x, k=1))

## ---------------------------------------------------------------- ##
## PVLV LEARNING SYSTEM
## ---------------------------------------------------------------- ##
## The PVLV algorithm is a reinforcement learning algorithm 
## described by O'Reilly, Frank, Hazy, & Wats (2007).  It has the
## advantages of being biologically plausible and requiring minimal
## computations to control dopamine response.
## ---------------------------------------------------------------- ##


e1      = 0.1     # Learning rate for PV and LVe systems
e2      = 0.075   # Learning rate for LVi
theta   = 0.51     # Thrshold above which the PV system is considerd active

def PViLearningRule(p, context=None):
    """The PVi algorithm learning rule"""
    #print "PVi: ", 
    d  = pve.activations - pvi.activations
    X  = p.groupFrom.activations
    W  = p.weights
    #print e1*d*X.T
    W += e1*d*X.T

## See Eq. (5), (6), (7)
def LVeLearningRule(p, context=None):
    """The LVe learning rule"""
    #print "LVe", 
    PVfilter = pvi.activations[0,0] > theta or pve.activations[0,0] > theta
    # Learning occurs only if PVi or PVe are active
    if PVfilter:
        d = pve.activations*lve.activations
        W = p.weights
        X = p.groupFrom.activations
     #   print d*e1*X.T
        W += d*e1*X.T

def LViLearningRule(p, context=None):
    """The LVi learning rule"""
    #print "LVi",
    PVfilter = pvi.activations[0,0] > theta or pve.activations[0,0] > theta
    if PVfilter:
        d = lve.activations - lvi.activations
        W = p.weights
        X = p.groupFrom.activations
     #   print d*e2*X.T
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

r = 0.75


def SN_LearningRule(p, context=None):
    """The Striatonigral learning rule"""
    da1 = da.activations
    da0 = da.GetHistory()[-1]
    #da0 = da.baselines
    d   = da1 - da0
    #print "SN/SP, da1=%s, da0=%s, d=%s, " % (da1, da0, d)
    Y1  = p.groupTo.activations
    Y0  = p.groupTo.GetHistory()[0]
    W   = p.weights
    Y   = Y1 - Y0
    X   = p.groupFrom.activations
    dW  = r*d*np.dot(Y, X.T)
    print "dW=", dW
    W  += dW
    
    

def SP_LearningRule(p, context=None):
    """The Striatopallidal learning rule"""
    return SN_LearningRule(p, context)

c2sn.learningFunction   = SN_LearningRule
c2sp.learningFunction   = SP_LearningRule

c2sn.learningEnabled    = True
c2sp.learningEnabled    = True



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
cortex.geometry = (6, 6)
sn.geometry     = (3, 2)
sp.geometry     = (3, 2)
