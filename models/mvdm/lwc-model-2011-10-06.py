## ---------------------------------------------------------------- ##
## MODEL
## ---------------------------------------------------------------- ##
## Contains a model for the simulations of the ACh-based learning
## system in the striatum.
## ---------------------------------------------------------------- ##

import math, random
import selection as sel
import numpy     as np
from   neural    import Group, Projection, Circuit, GenericUpdate
from   neural    import Linear, Tanh_plus, Step, boltzmann_kwta, SSigmoid, STanh_plus
import neural    as neu
#from   numpy     import dot, vectorize

## ---------------------------------------------------------------- ##
## PVLV LEARNING SYSTEM
## ---------------------------------------------------------------- ##
## The PVLV algorithm is a reinforcement learning algorithm 
## described by O'Reilly, Frank, Hazy, & Wats (2007).  It has the
## advantages of being biologically plausible and requiring minimal
## computations to control dopamine response.
## The originala values for e1 and e2 were:
##
##    e1  = 0.050
##    e2  = 0.001
## ---------------------------------------------------------------- ##

e1      = 0.250    # Learning rate for PVi and LVe systems
e2      = 0.005    # Learning rate for LVi (taken from O'Reilly et al, 2007)
tmax    = 0.650    # Thrrshold above which the PV system is considered active
tmin    = 0.350    # Threshold below which the PV system is considered active

def PVfilter(circuit):
    """Checks whether the PV filter should be on or off"""
    PVi = circuit.GetGroupByName("PVi")

    PVe = circuit.GetGroupByName("PVe")

    PVfilter = PVi.activations[0,0] > tmax or PVe.activations[0,0] > tmax \
        or PVi.activations[0,0] < tmin or PVe.activations[0,0] < tmin
    return PVfilter

def PViLearningRule(p, context=None):
    """The PVi algorithm learning rule"""
    PVi = context.GetGroupByName("PVi")
    PVe = context.GetGroupByName("PVe")
    d   = PVe.activations - PVi.activations
    X   = p.groupFrom.activations
    W   = p.weights
    M   = p.mask
    dW  = e1*d*X.T*M
    if context.GetParameter('TRACE_PVLV'):
        print "PVi delta: %s" % dW
    W  += dW

## See Eq. (5), (6), (7)
def LVeLearningRule(p, context=None):
    """The LVe learning rule"""
    PVe = context.GetGroupByName("PVe")
    LVe = context.GetGroupByName("LVe")
    e1  = context.GetParameter("e1")
    if PVfilter(context):
        # Learning occurs only if PVi or PVe are active
        d  = PVe.activations - LVe.activations
        W  = p.weights
        M  = p.mask
        X  = p.groupFrom.activations
        dW = d*e1*X.T*M
        if context.GetParameter('TRACE_PVLV'):
            print "LVe delta: %s, mask: %s" % (dW, M)
        W += dW

def LViLearningRule(p, context=None):
    """The LVi learning rule"""
    LVi = context.GetGroupByName("PVi")
    LVe = context.GetGroupByName("PVe")
    e2  = context.GetParameter("e2")
    if PVfilter(context):
        d  = LVe.activations - LVi.activations
        W  = p.weights
        M  = p.mask
        X  = p.groupFrom.activations
        dW = d*e2*X.T*M
        if context.GetParameter('TRACE_PVLV'):
            print "LVi delta: %s" % dW
        W += dW

def LVUpdate(lv, context=None):
    """LV Update functions---considers only changes in input"""
    lv.ClearInputs()
    for p in lv.incomingProjections:
        g  = p.groupFrom
        W  = p.weights
        dX = g.activations = g.GetHistory()[-1]
        dI = np.dot(W, dX)
        lv.inputs+=dI
    lv.CalculateActivations()

def PVLV_Update(g, context=None):
    """
    A simple Update for the PVLV system. It just called the
    GenericUpdate function in the neural model, but also prints
    out a trace if the context's TRACE_PVLV parameter is True.
    """
    GenericUpdate(g)
    if context.GetParameter('TRACE_PVLV'):
        print "  %s: input=%s, X=%s" % (g, g.inputs, g.activations)



# -------------------------------------------------------------------

def DopamineUpdate(da, context):
    """
    The update function for dopamine neurons. Dopamine update depends 
    on the PVLV system. In particular Dopamine output D is updated
    according to the following rule:
    
                  | dPV     if PVfilter() is True
       D = dLV + <
                  | 0       if PVfilter() is Fals
    
    where dLV is the delta of the LV subsystem (i.e., LVe - LVi) and
    dPV is the delta of the PV subsystem (i.e., PVe-PVi)
    """

    da.ClearInputs()
    LV = [p for p in da.incomingProjections if p.groupFrom.name.startswith("LV")]
    PV = [p for p in da.incomingProjections if p.groupFrom.name.startswith("PV")]
    for p in LV:
        p.PropagateThrough()
    
    if PVfilter(context):
        for p in PV:
            p.PropagateThrough()
    da.CalculateActivations()

    if context.GetParameter('TRACE_PVLV'):
        PVe = context.GetGroupByName("PVe")
        print "  %s: input=%s, X=%s" % (PVe, PVe.inputs, PVe.activations)
    
    if context.GetParameter('TRACE_DA'):
        print "  Da: input=%s, PVfil=%s, X=%s" % (da.inputs, PVfilter(), da.activations)


## ---------------------------------------------------------------- ##
## STRIATAL LEARNING SYSTEM
## ---------------------------------------------------------------- ##
## The striatum learns only when dopamine is present, and its 
## learning rate is proportional to the amount of dopamine released
## (as in Ashby, 2007; and Stocco, Lebiere, & Anderson, 2010).
## Also, dopamine learning is proportional to the activation
## deltas in SN and SP neurons (as in O'Reilly & Frank, 2006).
## ---------------------------------------------------------------- ##

#r   = 1.5
da1 = 1.5
da2 = 1.0

def SN_Update(group, context):
    """The update rule for SN and SP cells in the striatum"""
    group.ClearInputs()
    snr = context.GetGroupByName("SNr/GPi")
    for p in group.incomingProjections:
        if p.groupFrom.name.startswith("Da"):# and PVe.activations[0,0]!=0.50:
            D1 = np.dot(p.weights, p.groupFrom.activations)
            D2 = group.GetHistory()[-1]*p.groupFrom.activations
            
            if group.name.startswith("SP"):
                D2*=-1
            dX = D1/2 + D2/2
            dX *= Step(snr.activations)
            if context.GetParameter('TRACE_UPDATE'):
                print "  Group: %s,\n  Dopa D1: %s\n  D2: %s\n  dX: %s" % (group, D1, D2, dX)
            group.inputs += dX
        else:
            p.PropagateThrough()
    group.CalculateActivations()
    if context.GetParameter('TRACE_UPDATE'):
        print "  Group: %s,\n  Inputs=%s\n  X=%s" % (group, group.inputs, group.activations.T)


def SN_LearningRule(p, context=None, rate="da1"):
    """
    The learning rule for the striatonigral neurons.
    """
    r   = context.GetParameter(rate)
    da  = context.GetGroupByName("Da")
    D   = abs(np.max(da.activations))
    Y1  = p.groupTo.activations
    Y0  = p.groupTo.GetHistory()[-1]
    W   = p.weights
    M   = p.mask
    Y   = Y1 - Y0
    X   = np.copy(p.groupFrom.activations)
    dW  = r*D*np.dot(Y, X.T)
    dW *= M
    if context.GetParameter("TRACE_STRIATUM_LEARNING"):
        print "  P=%s,\n  Y=%s,\n  dW=%s" % (p, Y, dW)
    W  += dW
    

def SP_LearningRule(p, context=None):
    """
    The learning rule for striatopallidal neurons. The rule is the
    same as for the striatonigral rules, but uses a different
    dopamine parameter, reflecting the use of Da2-type receptors.
    """
    return SN_LearningRule(p, context, rate="da2")

## ---------------------------------------------------------------- ##
## MODEL 1 -- Simple
## ---------------------------------------------------------------- ##
##  SN      = Striatonigral cells;
##  SP      = Striatopallidal cells;
##  Context = Cortical cells;
##  SNR     = SNR/GPi/Thalamus cells
##  TANS    = Striatal ACh interneurons;
##  DA = Dopamine neurons
## ---------------------------------------------------------------- ##
def Model():
    """Creates an instance of the BABE Model"""
    # --- The nuclei --------------------------------
    sn      = Group(6, name="SN")     ;  sp  = Group(6, name="SP")         
    context = Group(9, name="Context");  snr = Group(6, name="SNr/GPi")             
    tans    = Group(3, name="TAN")    ;  da  = Group(1, name="Da")         

    #sn.activationFunction  = np.vectorize(lambda x: STanh_plus(x, gain=1))
    #sp.activationFunction  = np.vectorize(lambda x: STanh_plus(x, gain=1))
    #snr.activationFunction = np.vectorize(lambda x: STanh_plus(x, gain=1))
    #da.activationFunction  = Linear

    #sn.SetActivationFunction(Tanh_plus)
    #sp.SetActivationFunction(Tanh_plus)
    #snr.SetActivationFunction(Tanh_plus)
    da.SetActivationFunction(Linear)
   
    snr.kwta               = True
    snr.SetKWTAFunction(lambda x: boltzmann_kwta(x, k=1, tau=0.1))
    sn.SetUpdateFunction(SN_Update)
    sp.SetUpdateFunction(SN_Update)
    da.SetUpdateFunction(DopamineUpdate)

    ## --- Projections -------------------------------
    c2sn     = context.ConnectTo(sn);   c2sp     = context.ConnectTo(sp)
    tans2sn  = tans.ConnectTo(sn);      tans2sp  = tans.ConnectTo(sp);
    c2tans   = context.ConnectTo(tans); da2tans  = da.ConnectTo(tans);      
    da2sn    = da.ConnectTo(sn);        da2sp    = da.ConnectTo(sp);
    sn2snr   = sn.ConnectTo(snr);       sp2snr   = sp.ConnectTo(snr)

    # --- Thalamic feedback loops to BG
    snr2sp   = snr.ConnectTo(sp);       snr2sn   = snr.ConnectTo(sn)   
    
    c2sn.weights           = np.random.random(c2sn.weights.shape)/10.0
    c2sp.weights           = np.random.random(c2sp.weights.shape)/10.0
    c2sn.mask              = np.dot(np.ones((6,1)),
                                    np.array([[1,1,1,0,0,0,0,0,0]]))
    c2sp.mask              = np.dot(np.ones((6,1)),
                                    np.array([[1,1,1,0,0,0,0,0,0]]))
    c2sn.learningFunction  = SN_LearningRule
    c2sp.learningFunction  = SP_LearningRule

    c2sn.learningEnabled   = True
    c2sp.learningEnabled   = True

    #c2tans.weights         = np.zeros(c2tans.weights.shape)   # No context modulation
    #tans2sn.weights        = np.random.random(tans2sn.weights.shape)/10.0
    #tans2sp.weights        = np.random.random(tans2sp.weights.shape)/10.0
    da2sn.weights          = np.ones(da2sn.weights.shape)/10.0
    da2sp.weights          = np.ones(da2sp.weights.shape)/-10.0
    #da2tans.weights        = np.random.random(da2tans.weights.shape)/10
    sn2snr.weights         = np.ones(sn2snr.weights.shape)*np.eye(sn.size)
    sp2snr.weights         = np.ones(sp2snr.weights.shape)*np.eye(sp.size)*-1
    snr2sn.weights         = np.ones(sn2snr.weights.shape)*np.eye(sn.size)*0.2
    snr2sp.weights         = np.ones(sn2snr.weights.shape)*np.eye(sn.size)*0.2

    ## --- The PVLV system -------------------------------------------- ##

    PVe = Group(1, name="PVe");      PVi = Group(1, name="PVi")
    LVe = Group(1, name="LVe");      LVi = Group(1, name="LVi")

    PVe2da = PVe.ConnectTo(da);      LVe2da = LVe.ConnectTo(da)
    PVi2da = PVi.ConnectTo(da);      LVi2da = LVi.ConnectTo(da)
    c2PVi  = context.ConnectTo(PVi); c2LVe  = context.ConnectTo(LVe)
    c2LVi  = context.ConnectTo(LVi)

    LVi.SetUpdateFunction(LVUpdate)
    LVe.SetUpdateFunction(LVUpdate)
    #LVe.SetUpdateFunction(PVLV_Update)
    #LVi.SetUpdateFunction(PVLV_Update)
    PVi.SetUpdateFunction(PVLV_Update)
    PVe.SetUpdateFunction(PVLV_Update)
    
    ## --- PVLV Projections

    c2PVi.weights    = np.random.random(c2LVe.weights.shape) * 0.01
    c2LVe.weights    = np.random.random(c2LVe.weights.shape) * 0.01
    c2LVi.weights    = np.random.random(c2LVi.weights.shape) * 0.01
    c2PVi.mask       = np.dot(np.ones((1,1)),
                              #np.array([[1,1,1,0,0,0,0,0,0]]))
                              np.array([[0,0,0,1,1,1,0,0,0]]))
    c2LVi.mask       = np.dot(np.ones((1,1)),
                              np.array([[1,1,1,0,0,0,0,0,0]]))
    c2LVe.mask       = np.dot(np.ones((1,1)),
                              np.array([[1,1,1,0,0,0,0,0,0]]))
    PVe2da.weights   = np.ones((1,1))
    LVe2da.weights   = np.ones((1,1))
    PVi2da.weights   = np.ones((1,1))*-1
    LVi2da.weights   = np.ones((1,1))*-1

    c2PVi.learningFunction = PViLearningRule
    c2LVe.learningFunction = LVeLearningRule
    c2LVi.learningFunction = LViLearningRule

    c2PVi.learningEnabled  = True
    c2LVe.learningEnabled  = True
    c2LVi.learningEnabled  = True

    # --- Tricks for cute visualization
    context.geometry = (3, 3)
    sn.geometry      = (3, 2)
    sp.geometry      = (3, 2)

    # --- Setting up the circuit

    M1 = Circuit()
    
    # --- Add and set up the groups
    for x in [sn, sp, context, tans, da, snr,
              PVe, PVi, LVe, LVi]:
        x.SetContext(M1)
        M1.AddGroup(x)

    # --- Set up the circuit as the context in all projections
    for p in [c2sn, c2sp, tans2sn, tans2sp, c2tans, da2tans,
              da2sn, da2sp, sn2snr, sp2snr, PVe2da, PVi2da,
              LVe2da, LVi2da, c2PVi, c2LVe, c2LVi]:
        p.SetContext(M1)

    # --- Ok now: input and output
    M1.SetInput(context)
    M1.SetOutput(snr)

    M1.SetParameter('TRACE_UPDATE', False)
    M1.SetParameter('TRACE_PVLV', False)
    M1.SetParameter('TRACE_PVLV_LEARNING', False)
    M1.SetParameter('TRACE_STRIATUM_LEARNING',  False)
    M1.SetParameter('TRACE_DA', False)
    M1.SetParameter('e1', e1)
    M1.SetParameter('e2', e2)
    M1.SetParameter('da1', da1)
    M1.SetParameter('da2', da2)
    M1.SetParameter('tmax', tmax)
    M1.SetParameter('tmin', tmin)

    return M1


def Model1():
    """
    Model 1 is the basic reinforcement-learning type of model. It
    includes a simplified cortex (the "Context" layer), a rudimentary
    system to keep track of time within a trial, and a reinforcement
    learning subsystem based on the PVLV framework by O'Reilly et al.
    (2007).
    """
    m = Model()
    m.SetParameter("Model", "M1")
    return Model()

def Model1_A():
    """
    Model2 is like Model1, but it does not include a strict KWTA system
    in the output. For exploration only.
    """
    M1_A = Model()
    snr=M1_A.GetGroupByName("SNr/GPi")
    snr.kwta = False
    return M1_A

def Model1_B():
    """
    Model1_B is like Model1, but it includes a strict KWTA system for 
    the SN/SP subsystems as well.
    
    """
    M1_B = Model()
    return M1_B


## ---------------------------------------------------------------- ##
## MODEL 2
## ---------------------------------------------------------------- ##

tgain = 2.5
dat   = 0.5

def TAN_LearningRule(p, context):
    """
    The TAN learning rule. This rule is designed to work for the
    projections from Cortex to TAN only. It modulates the pressure
    that individual TANs exert on SN/SP neurons.
    """
    da = context.GetGroupByName("Da")
    HB = context.GetParameter("HB")
    d  = abs(np.sum(da.activations))
    W  = p.weights
    M  = p.mask
    X  = p.groupFrom.activations
    Y1 = np.maximum(p.groupTo.activations, HB)
    Y0 = p.groupTo.GetHistory()[-1]
    dY = Y1-Y0
    dW = dat*d*np.dot(dY, X.T)*M
    W -= dW
    if context.GetParameter("TRACE_TAN"):
        print "dY=%s\nW=%s\ndW=%s" % (dY, W, dW)

## ---------------------------------------------------------------- ##
## MODEL 2
## ---------------------------------------------------------------- ##

def Model2():
    """
    Model 2 (aka BABE) is a full version of the circuit that includes
    realistic cholinergic interneurons (TANS). In this version, TANs
    have a more canonical monotonically increasing activation function.
    """
    M2      = Model()
    tan     = M2.GetGroupByName("TAN")
    sn      = M2.GetGroupByName("SN")
    sp      = M2.GetGroupByName("SP")
    da      = M2.GetGroupByName("Da")
    context = M2.GetGroupByName("Context")
    c2tan   = M2.GetProjectionsBetweenGroups(context, tan)[0]
    tan2sn  = M2.GetProjectionsBetweenGroups(tan, sn)[0]
    tan2sp  = M2.GetProjectionsBetweenGroups(tan, sp)[0]
    da2tan  = M2.GetProjectionsBetweenGroups(da, tan)[0]
    
    W = np.zeros((6, 3))
    W[0:2,0] = W[2:4,1] = W[4:6,2] = 1.0
    tan2sn.mask = np.copy(W)
    tan2sp.mask = np.copy(W)
    tan2sn.weights = W*-1
    tan2sp.weights = W*-1

    sn2tan  = sn.ConnectTo(tan)
    sp2tan  = sp.ConnectTo(tan)
    sn2tan.weights = W.T/-10
    sp2tan.weights = W.T/-10
    da2tan.weights = np.ones(da2tan.weights.shape)*0.5
    
    
    tan.SetActivationFunction(np.vectorize(lambda x: SSigmoid(x, tgain)))
    tan.thresholds=0.5*np.ones(tan.inputs.shape)
    sn.thresholds = tan.GetActivationFunction()(np.zeros(sn.inputs.shape)-.5)
    sp.thresholds = tan.GetActivationFunction()(np.zeros(sp.inputs.shape)-.5)
    c2tan.weights  = np.random.random(c2tan.weights.shape)/-100.0
    c2tan.learningEnabled = True

    c2tan.learningFunction = TAN_LearningRule
    
    return M2


    
def Model3():
    """
    Model 3 (aka BABE) is a full version of the circuit that includes
    realistic cholinergic interneurons (TANS). TANs are modeled with
    a monotonically decreasing activation function, as previously done
    in Stocco, Lebiere, & Anderson (2010).
    """
    M2      = Model()
    tan     = M2.GetGroupByName("TAN")
    sn      = M2.GetGroupByName("SN")
    sp      = M2.GetGroupByName("SP")
    da      = M2.GetGroupByName("Da")
    context = M2.GetGroupByName("Context")
    snr     = M2.GetGroupByName("SNr/GPi")

    
    
    c2tan   = M2.GetProjectionsBetweenGroups(context, tan)[0]
    tan2sn  = M2.GetProjectionsBetweenGroups(tan, sn)[0]
    tan2sp  = M2.GetProjectionsBetweenGroups(tan, sp)[0]
    da2tan  = M2.GetProjectionsBetweenGroups(da, tan)[0]

    sn.SetActivationFunction(neu.Tanh_plus)
    sp.SetActivationFunction(neu.Tanh_plus)
    snr.SetActivationFunction(neu.Tanh_plus)

    W = np.zeros((6, 3))
    W[0:2,0] = W[2:4,1] = W[4:6,2] = 1.0
    tan2sn.mask = np.copy(W)
    tan2sp.mask = np.copy(W)
    tan2sn.weights = W*-1
    tan2sp.weights = W*-1

    sn2tan  = sn.ConnectTo(tan)
    sp2tan  = sp.ConnectTo(tan)
    sn2tan.weights = W.T/-10
    sp2tan.weights = W.T/-10
    da2tan.weights = np.ones(da2tan.weights.shape)*-0.25
    
    tan.SetActivationFunction(np.vectorize(lambda x: SSigmoid(x, tgain)))
    tan.thresholds=0.5*np.ones(tan.inputs.shape)
    hb = np.average(sn.thresholds)/-tan.size
    HB = np.ones(tan.inputs.shape)*hb
    sn.thresholds = 0.1*np.ones(sn.activations.shape)
    sp.thresholds = 0.1*np.ones(sp.activations.shape)
    #sn.thresholds = -1*tan.GetActivationFunction()(np.ones(sn.inputs.shape)-1)
    #sp.thresholds = -1*tan.GetActivationFunction()(np.ones(sp.inputs.shape)-1)
    #sn.thresholds = -0.1*tan.GetActivationFunction()(np.zeros(sn.inputs.shape))
    #sp.thresholds = -0.1*tan.GetActivationFunction()(np.zeros(sp.inputs.shape))
    #c2tan.weights  = np.random.random(c2tan.weights.shape)
    c2tan.weights  = np.ones(c2tan.weights.shape)*1.5
    c2tan.mask     = np.dot(np.ones(tan.inputs.shape),
                            np.array([[1,1,1,0,0,0,0,0,0]]))
    c2tan.learningEnabled  = True
    c2tan.learningFunction = TAN_LearningRule

    M2.SetParameter("TRACE_TAN", True)
    M2.SetParameter("HB", HB)
    return M2




def gp(m, n1, n2):
    g1 = m.GetGroupByName(n1)
    g2 = m.GetGroupByName(n2)
    return m.GetProjectionsBetweenGroups(g1, g2)[0]
