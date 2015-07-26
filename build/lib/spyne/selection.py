## ---------------------------------------------------------------- ##
## SELECTION.PY
## ---------------------------------------------------------------- ##
## This file provides some primitives for selecting between
## alternative options that are associated with different values.
## ---------------------------------------------------------------- ##

import math, random
from decimal import Decimal

class SelectionException(Exception):
    """An Exception occurring in a Selection function"""
    def __init__(self, value, msg):
        self.value=value
        self.msg=msg

    def __repr__(self):
        return "%s (%s)" % (self.value, self.msg)

    def __str__(self):
        return self.__repr__()

## ENTROPY
##
## Calculates Shannon's H function, defined as:
##
##    H(X) = - sum_x(P(x)*log_2(P(x)))
##
def Entropy(probs):
    """Calculates the entropy of the distribution of states"""
    S = math.fsum(probs)  
    P = [float(x)/S for x in probs]     # Normalize
    P = [x for x in P if x > 0]         # Remove zeros
    H = [x*math.log(x,2) for x in P]    # Calculate x*log(x) for each x.
    return -1*math.fsum(H)              # The return the sum

## This is the simplest selection function, a 'spin wheel'
## algorithm. Given a list V of values v1, v2...vN positioned at
## indexes i1, i2...iN, returns each index i with probability:
##
##   P(i) = V(i)/sum(V)
##
## If V is a Prob Distribution Function (ie, sum(v)==1), then:
##
##   P(i) = V(i)
##
def Wheel(probs, verbose=False):
    S = math.fsum(probs)
    P = [x for x in probs if x < 0.0]
    if len(P) > 0:
        raise SelectionException(P, "Negative values are not allowed")
    if S == 0:
        raise SelectionException(S, "Sum of values must be > 0")
    C = [math.fsum(probs[:i+1]) for i in range(len(probs))]
    v = random.uniform(0,math.fsum(probs))
    if verbose:
        print "CDF = %s; x = %s" % (C, v)
    for i in xrange(len(probs)):
        if C[i] > v:
            return i
    return len(probs)-1

## A Boltzmann selection function.
## Given a list V of values v1, v2...vN positioned at indexes i1,
## i2...iN, returns each index i with probability:
##
##   P(i) = e^(V(i)/T) / sum_i(e^(V)i)/T)
##
## where T is the Boltzmann temperature (Tau).
##
def Boltzmann(values, tau):
    E = [math.exp(x/tau) for x in values]
    s = math.fsum(E)
    P = [x/s for x in E]
    return Wheel(P, verbose=False)

## Simulates N selections from a given PDF
## using the given selection functon (default is 'Wheel')
##
def Lottery(probs, n, function=Wheel, frequencies=False):
    D = {}
    L = len(probs)
    for i in xrange(L):
        D[i]=0
    for i in xrange(n):
        v = function(probs)
        D[v]+=1
    if not frequencies:
        for i in xrange(L):
            D[i]/=float(n)
    return D
        
