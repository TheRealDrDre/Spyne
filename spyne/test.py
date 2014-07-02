## ---------------------------------------------------------------- ##
## TEST.PY
## ---------------------------------------------------------------- ##
## This file contains the necessary primitives to have the model
## interact with a task.
## ---------------------------------------------------------------- ##

import math, random, task, copy
import selection as sel
import numpy     as np
#import mvdm      as mvdm
import model     as model
#from task import *

## --- INTERPRETER OBJECTS ---------------------------------------- ##

## OPTION CONTAINER
## A simple object that contains an internal catalogue of admissible
## options.
## 
class OptionContainer:
    """An object that translates options into neural positions"""
    def __init__(self, options=[]):
        self.SetOptions(options)

    def SetOptions(self, options):
        self.__options = options

    def GetOptions(self):
        return self.__options

    def IsValidOptions(self, option):
        return option in self.GetOptions()

    def NumberOfOptions(self):
        return len(self.GetOptions())

## PARAMETER CONTAINER
## A simple object that manages an internal dictionary of parameters.
##
class ParameterContainer:
    def __init__(self):
        self.__params = {}
        
    def GetParameters(self):
        return self.__params

    def GetParameter(self, param):
        if param in self.__params.keys():
            return self.__params[param]
        else:
            return None

    def SetParameter(self, param, value):
        self.__params[param] = value

## LOCALIST INTERPRETER
##
## An object that interprets options in a localist neural
## representation (i.e., each option corresponds to a single
## neuron). It contains primitives for encoding and decoding
## options.
##
class LocalistInterpreter(OptionContainer, ParameterContainer):
    def __init__(self, options):
        ParameterContainer.__init__(self)
        self.__oTranslationTable = None
        self.__nTranslationTable = None
        self.SetOptions(options)
        if self.GetOptions() is not None:
            self.SetupTranslationTables()

    def SetOptionTranslationTable(self, table):
        self.__oTranslationTable = table

    def GetOptionTranslationTable(self):
        return self.__oTranslationTable

    def SetNeuronTranslationTable(self, table):
        self.__nTranslationTable = table

    def GetNeuronTranslationTable(self):
        return self.__nTranslationTable

    def SetupTranslationTables(self):
        O = self.GetOptions()
        O.sort()
        OTT = {}
        NTT = {}
        for i in xrange(len(O)):
            OTT[O[i]] = i
            NTT[i]    = O[i]
        self.SetNeuronTranslationTable(NTT)
        self.SetOptionTranslationTable(OTT)
        
    def EncodeOptions(self, options):
        A = np.zeros((self.NumberOfOptions(),1))
        for o in options:
            if o in self.GetOptions():
                OTT = self.GetOptionTranslationTable()
                A[OTT[o]] = 1
            else:
                raise OffendingOption(o, msg="Option not included in list of options")
        return A
    
    def DecodeOptions(self, array):
        options = []
        NTT = self.GetNeuronTranslationTable()
        for i in xrange(array.size):
            if array[i,0] == 1:
                options.append(NTT[i])
        return options


## CHOICE INTERPRETER
##
## An object that interprets options in an asymmetric neural
## representation (i.e., each pair of options correspond
## a single neuron). It contains primitives for encoding
## and decoding options.
##
class ChoiceInterpreter(OptionContainer, ParameterContainer):
    def __init__(self, options):
        ParameterContainer.__init__(self)
        self.__oTranslationTable = None
        self.__nTranslationTable = None
        self.SetOptions(options)
        if self.GetOptions() is not None:
            self.SetupTranslationTables()

    def SetOptionTranslationTable(self, table):
        self.__oTranslationTable = table

    def GetOptionTranslationTable(self):
        return self.__oTranslationTable

    def SetNeuronTranslationTable(self, table):
        self.__nTranslationTable = table

    def GetNeuronTranslationTable(self):
        return self.__nTranslationTable

    def SetupTranslationTables(self):
        """
        Sets up the translations table, i.e. the dictionaries
        that map a neural pattern into an decision, and a decision cue
        (or set of options) into a neural pattern.
        """
        O = self.GetOptions()
        O.sort()
        OTT = {}
        NTT = {}
        for i in xrange(len(O)):
            OTT[O[i]] = i
            NTT[i]    = O[i]
        self.SetNeuronTranslationTable(NTT)
        self.SetOptionTranslationTable(OTT)
        
    def EncodeOptions(self, options):
        if len(options) != 2:
            raise OffendingOption(options, "Only two options can be allowed")
        elif len([x for x in options if x not in self.GetOptions()]) > 0:
            raise OffendingOption(options, "One of the options was not allowed")
        else:
            A    = np.zeros((self.NumberOfOptions()**2,1))
            OTT  = self.GetOptionTranslationTable()
            r    = OTT[options[0]]
            c    = OTT[options[1]]
            i    = r*self.NumberOfOptions() + c
            A[i] = 1
        return A
    
    def DecodeOptions(self, array):
        options = []
        NTT = self.GetNeuronTranslationTable()
        for i in xrange(array.size):
            if array[i,0] == 1:
                options.append(NTT[i])
        return options


## SYMMETRIC CHOICE INTERPRETER
##
## An object that interprets options in an asymmetric neural
## representation (i.e., each pair of options correspond
## a single neuron). It contains primitives for encoding
## and decoding options.
##
class SymmetricChoiceInterpreter(OptionContainer, ParameterContainer):
    def __init__(self, options):
        ParameterContainer.__init__(self)
        self.__oTranslationTable = None
        self.__nTranslationTable = None
        self.SetOptions(options)
        if self.GetOptions() is not None:
            self.SetupTranslationTables()

    def SetOptionTranslationTable(self, table):
        self.__oTranslationTable = table

    def GetOptionTranslationTable(self):
        return self.__oTranslationTable

    def SetNeuronTranslationTable(self, table):
        self.__nTranslationTable = table

    def GetNeuronTranslationTable(self):
        return self.__nTranslationTable

    def SetupTranslationTables(self):
        O = self.GetOptions()
        O.sort()
        OTT = {}
        NTT = {}
        for i in xrange(len(O)):
            #OTT[O[i]] = i
            NTT[i]    = O[i]
        D=[[x,y] for x in O for y in O]
        D=[sorted(x) for x in D]
        D=[x for x in D if x[0] != x[1]]
        D=["%s%s" % (x[0], x[1]) for x in D]
        D=list(set(D))
        D.sort()
        
        for i in xrange(len(D)):
            OTT[D[i]] = i
        self.SetNeuronTranslationTable(NTT)
        self.SetOptionTranslationTable(OTT)
        
    def EncodeOptions(self, options):
        if len(options) != 2:
            raise OffendingOption(options, "Only two options can be allowed")
        elif len([x for x in options if x not in self.GetOptions()]) > 0:
            raise OffendingOption(options, "One of the options was not allowed")
        else:
            n    = self.NumberOfOptions()
            n    = n*(n-1)/2
            A    = np.zeros((n,1))
            OTT  = self.GetOptionTranslationTable()
            op   = sorted(options)
            key  = "%s%s" % (op[0], op[1])
            i    = OTT[key]
            A[i] = 1
        return A
    
    def DecodeOptions(self, array):
        options = []
        NTT = self.GetNeuronTranslationTable()
        for i in xrange(array.size):
            if array[i,0] == 1:
                options.append(NTT[i])
        return options


## M1 INTERPRETER
##
## The M1 interpreter encodes and decodes options, and interprets
## non-localist representations as distributions of values in a
## Boltzmann selection algorithm
##
class M1Interpreter(SymmetricChoiceInterpreter):
    def __init__(self, options, tau=0.05, boltzmann=False):
        """Initializes a ChoiceInterpreter object"""
        SymmetricChoiceInterpreter.__init__(self, options)
        self.SetParameter('tau', tau)
        self.boltzmann = boltzmann
                 
    def DecodeChoice(self, array):
        """Decodes a neural representation as a Boltzmann choice"""
        t   = self.GetParameter('tau')
        V   = array.flatten()
        i   = sel.Boltzmann(V, t)
        NTT = self.GetNeuronTranslationTable()
        return NTT[i]


class M1Interpreter2(SymmetricChoiceInterpreter):
    """
    Final version of the Model/Task interpreter:
     * Manages only three types of options, i.e. AB, CD, and EF.
     * Encodes each choice option as a localist 3-neuron 
       representation, where each neuron represents one of the
       possible sets of options.
     * Decodes choices from a localist neural representation where
       each neuron represents one of the possible actions.
    """
    def __init__(self, options, tau=0.01, time=3, boltzmann=False):
        """Initializes a ChoiceInterpreter object"""
        SymmetricChoiceInterpreter.__init__(self, options)
        self.boltzmann = boltzmann
        self.SetParameter('tau', tau)
        self.SetParameter('time', time)

    def EncodeOptions(self, options, time=0):
        alternatives = [['A','B'], ['C','D'], ['E', 'F']]
        if options not in alternatives:
            raise OffendingOption(options, "Combinations of options not allowed")
        else:
            t    = self.GetParameter('time')
            i    = alternatives.index(options)
            A    = np.zeros((len(alternatives)*t,1))
            for j in xrange(0,time+1):
                A[(j*t)+i,0] = 1
        return A
    
    def DecodeChoice(self, array):
        """Decodes a neural representation as a choice"""
        NTT = self.GetNeuronTranslationTable()
        if self.boltzmann:
            # Determines the choice based on a Boltzmann dist
            t = self.GetParameter('tau')      # Tau parameter in Boltzmann eq
            V = array.flatten()               # Neural pattern as row vector
            i = random.choice(range(V.size))  # Base case is random
            try:
                i = sel.Boltzmann(V, t)
            except OverflowError:
                if (V>np.zeros(V)).any():
                    i = np.argmax(array)
                #print "Switch to random"
        else:
            i   = np.argmax(array)
        return NTT[i]
      

    def EncodeChoice(self, choice):
        o = copy.copy(self.GetOptions())
        o.sort()
        i = o.index(choice)
        A = np.zeros((len(o), 1))
        A[i] = 1
        return A
        

class Test:
    """
    A Test is a system that couples a task and a model, and has the
    model interact with the task. It is composed of a (neural) model,
    a task object, and an interpreter object that translates the
    task options into neural activation patterns, and decodes the
    model's output into a decision.
    """
    def __init__(self, model=model.Model1()):
        self.model       = model
        self.interpreter = M1Interpreter2(task.VDM_OPTIONS.keys())
        self.interpreter.boltzmann = True
        self.task        = task.MVDM()   # Multi-valued decision-making
                
    def Run(self, n=20):
        """Runs up to N interactions with the task"""
        j = 0
        t = self.task
        m = self.model
        i = self.interpreter
        
        # --- The groups we use
        context = self.model.GetGroupByName('Context')
        snr     = self.model.GetGroupByName('SNr/GPi')
        PVe     = self.model.GetGroupByName('PVe')
        while j < n:
            t.Next()
            
            # --- Set context to current options
            inputs = i.EncodeOptions(t.stimuli, time=0)
            context.SetClamped(False)
            context.SetActivations(inputs)
            context.SetClamped(True)

            # --- Set primary rewards to 0
            PVe.SetClamped(False)
            PVe.SetActivations(0.5*np.ones(PVe.inputs.shape))
            PVe.SetClamped(True)

            # --- Decode choice
            m.Update(verbose=False)
            T = snr.activations == np.zeros(snr.activations.shape)
            choice = i.DecodeChoice(snr.activations)
            if T.all():
                #print "**** %s (random pick) ****" % T.T
                A      = self.interpreter.EncodeChoice(choice)
                snr.SetActivations(A)
                        
            #choice = i.DecodeChoice(snr.activations)
            t.ProcessResponse(choice)

            # --- Time passes
            inputs = i.EncodeOptions(t.stimuli, time=1)
            context.SetClamped(False)
            context.SetActivations(inputs)
            context.SetClamped(True)

            # --- Represent feedback
            PVe.SetClamped(False)
            if t.feedback == 1:
                PVe.SetActivations(0.8*np.ones((1,1)))
            elif t.feedback == 0:
                PVe.SetActivations(0.20*np.ones((1,1)))
            else:
                PVe.SetActivations(0.1*np.ones((1,1)))
            PVe.SetClamped(True)

            # --- Update with current feedback
            snr.SetClamped(True)
            self.model.Update(verbose=False)

            # --- Learn from update
            self.model.Learn()
            snr.SetClamped(False)
            
            # --- Increase counter
            j+=1

## ---------------------------------------------------------------- ##
## MORE COMPLEX TESTS
## ---------------------------------------------------------------- ##
            

class Test1(Test):
    """
    Test1 is a simple test of learning and relearning.
    """
    def __init__(self, model=model.Model1()):
        Test.__init__(self, model)
        self.window   = 10  # The moving window
        self.targets  = {}  # Trials where optimal performance was achieved first.
        self.maxLength = 5000
        
    def doTest(self):
        """
        Performs the test. Runs the model through a set of decision
        in the MVDM task until either (a) Target performance is reached,
        or (b) the number of trials has exceeded the maximum allowed
        (defaults is 5,000)
        """
        while not self.TargetsReached() and self.task.index < self.maxLength:
            self.Run(1)
            self.UpdateTargets()
        
        return self.targets

    def UpdateTargets(self):
        """ Updates the dictionary of trials where the target is reached"""
        bO = self.task.byOption
        w  = self.window
        if len(bO.keys()) == 3:
            if any([len(bO[x]) >= w for x in bO.keys()]):
                for o, H in bO.iteritems(): 
                    rsp = max(o, key=lambda x: self.task.choices[x])  # best response
                    tgt = self.task.choices[rsp]                      # target performance
                    p   = len([x for x in H[-w:] if x.optimal==True])/float(w)
                    if p >= tgt and o not in self.targets.keys():
                        #print "Response %s passed, i=%d [Targets=%s]" % (rsp, self.task.index, self.targets)
                        self.targets[o] = self.task.index
                    
    def TargetsReached(self):
        """
        The target has been reached when all options have been logged
        in the target dictionary.
        """
        return len(self.targets.keys()) == 3


class Test2(Test):
    """
    Test 2 tests the model's capacity for re-learning. It performs the
    MVDM task until learning is stable
    """
    def __init__(self, model=model.Model1()):
        """
        Inits the Test. By default will test Model1 over max 5000
        trials of the MVDM task.
        """
        Test.__init__(self, model)
        self.window    = 10  # The moving window
        self.targets   = {}  # Trials where optimal performance was achieved first.
        self.phases    = {'Normal' : None, 'Reversal':None }
        self.maxLength = 5000
        self.offset    = 0
        
    def doTest(self):
        """Performs the test."""
        self.task.choices     = task.VDM_OPTIONS
        while not self.TargetsReached() and self.task.index < self.maxLength:
            self.Run(1)
            self.UpdateTargets()
        
        # Save the previous stats and reset
        self.phases['Normal'] = self.targets
        self.offset           = self.task.index
        self.task.choices     = task.VDM_OPTIONS_REVERSED
        self.task.index       = 1
        self.task.byOption    = {}
        self.targets          = {}
        while not self.TargetsReached() and self.task.index < self.maxLength:
            self.Run(1)
            self.UpdateTargets()
        
        self.phases['Reversal'] = self.targets
        self.task.Stats(breakdown=False)
        return self.phases

    def UpdateTargets(self):
        """ Updates the dictionary of trials where the target is reached"""
        bO = self.task.byOption
        w  = self.window
        if len(bO.keys()) == 3:
            if any([len(bO[x]) >= w for x in bO.keys()]):
                for o, H in bO.iteritems(): 
                    rsp = max(o, key=lambda x: self.task.choices[x])  # best response
                    tgt = self.task.choices[rsp]                      # target performance
                    p   = len([x for x in H[-w:] if x.optimal==True])/float(w)
                    if p >= tgt and o not in self.targets.keys():
                        #print "Response %s passed, i=%d [Targets=%s]" % (rsp, self.task.index, self.targets)
                        self.targets[o] = self.task.index # - self.offset
                    
    def TargetsReached(self):
        """
        The target has been reached when all options have been logged
        in the target dictionary.
        """
        return len(self.targets.keys()) == 3


class Test3(Test):
    """
    This test tracks the amount of dopamine and the level of Entropy
    in different parts of the circuit.
    """
    def __init__(self, model=model.Model1()):
        Test.__init__(self)
        self.model   = model
        self.nBefore = 600
        self.nAfter  = 600
        self.window  = 10
        self.tracked = ("Da", "SN", "SP", "SNr/GPi")
        self.history = []

    def GetModelPhase(self):
        c=self.model.GetGroupByName("Context")
        return np.sum(c.activations)

    def GetGroupEntropy(self, g, fromInputs=False):
        if fromInputs:
            X = g.GetActivationFunction()(g.inputs)
        else:
            X = g.activations
            
        V = [float(x) for x in X]
        V = [x for x in V if x > 0]
        if len(V) != 0:
            return sel.Entropy(V)
        else:
            return sel.Entropy(np.ones(X.shape))

    def GetMovingAverages(self):
        R = []
        for o in ['AB', 'CD', 'EF']:
            p = 1./6.               # Probability of correct response (initial)
            if o in self.task.byOption.keys():
                #print "[%d]  Found list for %s" % (self.task.index, o)
                H = self.task.byOption[o]  # Previous history for that options
                w = min([self.window, len(H)])
                #print "  Set window of %d" % w
                if w > 0:
                    p = len([x for x in H[-w:] if x.optimal==True])/float(w)
                    #print "  P=%.3f" % p
            R.append(p)
        return R

    def GetProbabilityDistributions(self):
        res =[]
        for o in ['AB', 'CD', 'EF']:
            D=[]
            if o in self.task.byOption.keys():
                H = self.task.byOption[o]  # Previous history for that options
                w = min([self.window, len(H)])
                R = [x.response for x in H[-w:]]
                D = [R.count(x) for x in sorted(self.task.choices.keys())]
                q = float(max(1, w))
                D = [x/q for x in D]
                D.append(sel.Entropy(D))                
            else:
                D=[0]*(len(self.task.choices.keys())+1)

            res += D
        return res
 
    
    def Track(self, circuit, e):
        if self.GetModelPhase() == 1:
            V  = self.GetMovingAverages()
            V += self.GetProbabilityDistributions()
            for name in self.tracked:
                g = self.model.GetGroupByName(name)
                v = None
                if name=="SNr/GPi":
                    v = self.GetGroupEntropy(g, fromInputs=True)
                elif name=="Da":
                    v = float(g.activations)
                else:
                    v = self.GetGroupEntropy(g)
                V.append(v)
            V=[self.task.index]+V
            self.history.append(V)


    def doTest(self):
        """Runs the test"""
        self.model.AddUpdateListener(self.Track)
        self.Run(self.nBefore)
        self.task.choices=task.VDM_OPTIONS_REVERSED
        self.Run(self.nAfter)

        return self.history

class Test4(Test2, Test3):
    """
    Performs a mixture of Text3 and Text4. Runs until a model
    reaches a successul performance on the AB pair. Then, it
    reverses the contingencies and runs for a fixed amount of
    trial. It records probabilities of each action.
    """

    def __init__(self, model=model.Model1()):
        Test2.__init__(self, model)
        Test3.__init__(self, model)
        self.updateTargets = True
        self.minPre = 30

    def doTest(self):
        """Performs the test."""
        self.model.AddUpdateListener(self.Track)
        self.task.choices     = task.VDM_OPTIONS
        while not self.TargetsReached():
            self.Run(1)
            self.UpdateTargets()

        # When the targets are reached, proceed with reversal
        
        self.task.choices  = task.VDM_OPTIONS_REVERSED
        print "Reversal", self.targets
        # self.task.index       = 1
        # self.task.byOption    = {}
        L_ab = len(self.task.byOption['AB']) + 100
        
        self.updateTargets = False
        while len(self.task.byOption['AB']) < L_ab:
                  self.Run(1)
        
        print "Stats---"
        self.task.Stats(breakdown=False)
        # now we want to keep only the last 130 trials with AB
        H  = []
        AB = [x for x in self.task.log if x.stimuli == ['A', 'B']]
        for x in AB[-130:]:
            H.append(self.history[x.index-1])
        #H = self.history[-150:]
        for i, V in enumerate(H):
            V[0]=i+1
        return H

    def UpdateTargets(self):
        if (self.updateTargets):
            Test2.UpdateTargets(self)
                    
    def TargetsReached(self):
        """
        In Test4, the The target has been reached when an entry has
        been logged for the AB pair, and AB has a history of a least
        30 choices.
        """
        res = False
        if 'AB' in self.targets.keys():
            if len(self.task.byOption['AB']) >= self.minPre:
                res = True
        return res


    
