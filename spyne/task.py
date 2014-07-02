## ---------------------------------------------------------------- ##
## TASK
## ---------------------------------------------------------------- ##
## This file contains classes, methods, and specifications for
## running the VDM task suite on neural models.  It also provides
## general function interfaces from tasks to neural models.
## ---------------------------------------------------------------- ##

import math, random, copy
import selection as sel
import numpy     as np
import neural    as neural


## --- VARIABLES AND CONSTANTS ------------------------------------ ##

## Stimuli and probabilities for the VDM task.

VDM_OPTIONS = {'A' : 0.8,
               'B' : 0.2,
               'C' : 0.7,
               'D' : 0.3,
               'E' : 0.6,
               'F' : 0.4}

VDM_OPTIONS_REVERSED = {'A' : 0.2,
                        'B' : 0.8,
                        'C' : 0.3,
                        'D' : 0.7,
                        'E' : 0.4,
                        'F' : 0.6}


## --- ERRORS AND EXCEPTIONS -------------------------------------- ##

class OffendingOption(Exception):
    """
    An exception raised when handling an options that cannot be 
    processed
    """
    option=None
    msg   =None
    def __init__(option=None, msg=None):
        self.option = option
        self.msg    = msg



## ---------------------------------------------------------------- ##
## THE TASK
## ---------------------------------------------------------------- ##

class Decision:
    """An abstract representation of a decision act"""
    def __init__(self, stimuli=[], response=None,
                 probabilities = {}, 
                 feedback=0, optimal=False, index=0):
        self.stimuli       = stimuli
        self.probabilities = probabilities
        self.response      = response
        self.feedback      = feedback
        self.optimal       = optimal
        self.index         = index

    def __repr__(self):
        return "D[%d]<%s, %s, %s>" % (self.index, self.stimuli,
                                      self.response, self.feedback)

    def __str__(self):
        return self.__repr__()

class VDM(neural.ParametrizedObject):
    """The VDM task"""
    def __init__(self, choices=VDM_OPTIONS, n=2):
        """Initializes a multi-option decision-making task"""
        neural.ParametrizedObject.__init__(self)
        self.choices   = VDM_OPTIONS
        self.log       = []
        self.current   = None
        self.stimuli   = None
        self.probs     = []
        self.feedback  = None
        self.optimal   = None
        self.response  = None
        self.responded = False
        self.index     = 0
        self.byOption  = {}
        self.NewChoice()
        self.SetParameter("TRACE_OPTIONS", False)
        self.SetParameter("TRACE_FEEDBACK", False)
        self.SetParameter("TRACE_RESPONSES", False)
        

    def ProcessResponse(self, response, proceed=False):
        """Processes a response"""
        self.response  = response
        self.responded = True

        if self.GetParameter("TRACE_RESPONSES"):
            print "[%04d] Response: %s" % (self.index, response)
        
        if response in self.stimuli:
            v = random.uniform(0,1)
            if v < self.choices[response]:
                self.feedback = 1
            else:
                self.feedback = 0
                
            A = [x for x in self.stimuli if x != response]  # Alternatives
            V = [self.choices[a] for a in A]                # Alternatives' values  

            if self.choices[response] >= max(V):
                self.optimal = True
            else:
                self.optimal = False
        self.LogDecision()
        
        if self.GetParameter("TRACE_FEEDBACK"):
            print "[%04d] Feedback: %s" % (self.index, self.feedback)

        if proceed and response in self.stimuli:
            self.Next()

    def Next(self):
        """Proceeds to the next decision"""
        #self.LogDecision()
        self.NewChoice()
        self.index+=1
        self.feedback  = None
        self.optimal   = None
        self.response  = None
        self.responded = False

        if self.GetParameter("TRACE_OPTIONS"):
            print "[%04d] Options: %s" % (self.index, self.stimuli)

        return self.stimuli

    def LogDecision(self):
        """Logs the current decision"""
        decision = Decision(stimuli=self.stimuli, response=self.response,
                            probabilities = self.probs,
                            feedback=self.feedback, optimal=self.optimal,
                            index=self.index)
        self.log.append(decision)
        
        # Log by option as well:

        key = "%s%s" % (self.stimuli[0], self.stimuli[1])
        if key in self.byOption.keys():
            self.byOption[key].append(decision)
        else:
            self.byOption[key]=[decision]
        

    def NewChoice(self):
        """Generates a new choice"""
        self.stimuli = random.sample(self.choices.keys(), 2)
        self.probs   = {}
        for x in self.stimuli:
            self.probs[x] = self.choices[x]


    def Stats(self, intervals=10, breakdown=True):
        """Prints out statistics on task performance"""
        n  = len(self.log)
        l1 = "%d" % len("%d" % n)
        l2 = "%d" % len("%d" % intervals)
        fs = "[%"+ l2+"d] %" + l1 + "d-%" + l1 + "d Answered %.3f Positive %.3f Correct %.3f"
        if n >= 100:
            m = n/intervals
            b = 0
            e = min(n, b+m)
            for i in xrange(intervals-1):
                S = self.log[b:e]
                f = len([x for x in S if x.feedback == 1])/float(m)
                c = len([x for x in S if x.optimal == True])/float(m)
                a = len([x for x in S if x.feedback != None])/float(m)
                print fs % (i+1, b, e, a, f, c)
                if breakdown:
                    self.Breakdown(S)
                
                b+=m
                e = min(n, b+m)
            # --- Last batch (include up to the end)
            S = self.log[b:n]
            f = len([x for x in S if x.feedback == 1])/float(m)
            c = len([x for x in S if x.optimal == True])/float(m)
            a = len([x for x in S if x.feedback != None])/float(m)
            print fs % (i+2, b, b+len(S), a, f, c)
            if breakdown:
                self.Breakdown(S)

    def Breakdown(self, S):
        """Prints a percentage of each response to each given option"""
        O = []
        for x in S:
            if x.stimuli not in O:
                O.append(x.stimuli)
        O.sort()
        for o in O:
            L = [x.response for x in S if x.stimuli==o]
            R = list(set(L))
            R.sort()
            res = ""
            for r in R:
                res+=("%s=%.2f%% " % (r, 100*L.count(r)/float(len(L))))
            print "   %s: %s" % (o, res)


    def Table(self):
        """Returns results as a table"""
        T = []
        r = None
        for x in self.log:
            options = "%s%s" % (x.stimuli[0], x.stimuli[1])
            rType   = None
            if x.optimal:
                rType = "Optimal"
            elif x.feedback == None:
                rType = "Other"
            else:
                rType = "Non-optimal"
            
            r = [x.index, options, rType, x.optimal, x.feedback]
            T.append(r)
        return T

class MVDM (VDM):
    """The Multi-VDM task"""
    def __init__(self, choices=VDM_OPTIONS):
        VDM.__init__(self, choices=choices, n=2)

    def NewChoice(self):
        """Generates a new choice"""
        alternatives = [['A','B'], ['C','D'], ['E', 'F']]
        self.stimuli = copy.copy(random.choice(alternatives))
                                        
        self.probs   = {}
        for x in self.stimuli:
            self.probs[x] = self.choices[x]


## TRACING AND DEBUGGING

TRACE_OPTIONS   = True
TRACE_RESPONSES = True
TRACE_FEEDBACK  = True

