from neural import *
import numpy as np
import copy
import random

## ---------------------------------------------------------------- ##
## CHL - CONTRASTIVE HEBBIAN LEARNING
## ---------------------------------------------------------------- ##
## Two CHL methods are given, Synchronous and Asynchronous.   In
## The synchronous methods, the activation values for the hebbian
## and antihebbian phases are calculated, and a single update of the
## projection weights is performed (based on the difference between
## the two phases).  This is the fastest method, but also the most
## biologically implausible.
## In the Asynchronous procedure, the hebbian and anti-hebbian
## phases are calculated separately 
##
def chl_synchronous(c, inputs, targets, rate=0.2):
    """Synchronous Contrastive Hebbian Learning"""
    hebb        = {}
    antihebb    = {}
    projections = c.GetProjections(exclusive=True)
    c_inputs    = c.GetInput()
    c_outputs   = c.GetOutput()
    
    for i in range(len(inputs)):
        c_inputs[i].activations=inputs[i]
        c_inputs[i].SetClamped(True)
        
    c.Update(verbose=False)
    hebb={g : np.copy(g.activations) for g in c.GetGroups()}

    for i in range(len(targets)):
        c_outputs[i].activations=targets[i]
        c_outputs[i].SetClamped(True)

    c.Update(verbose=False)
    antihebb={g : np.copy(g.activations) for g in c.GetGroups()}

    for i in range(len(targets)):
        c_outputs[i].SetClamped(False)

    for p in projections:
        # I think this was a mistake
        #delta = antihebb[p.groupTo] * hebb[p.groupFrom].T - hebb[p.groupTo] * hebb[p.groupFrom].T
        delta = antihebb[p.groupTo] * antihebb[p.groupFrom].T - hebb[p.groupTo] * hebb[p.groupFrom].T
        p.weights += delta


def chl_asynchronous(c, inputs, targets, rate=0.2):
    """Asynchronous Contrastive Hebbian Learning"""
    projections = c.GetProjections(exclusive=True)
    c_inputs = c.GetInput()
    c_outputs = c.GetOutput()
    
    for i in range(len(inputs)):
        c_inputs[i].activations=inputs[i]
        c_inputs[i].SetClamped(True)
        c_outputs[i].activations=targets[i]
        c_outputs[i].SetClamped(True)

    c.Update(verbose=False)
    
    for p in projections:
        delta = p.groupTo.activations*p.groupFrom.activations.T
        p.weights += delta

    for i in range(len(targets)):
        c_outputs[i].SetClamped(False)

    c.Update(verbose=False)
    
    for p in projections:
        delta = p.groupTo.activations*p.groupFrom.activations.T
        p.weights -= delta
    

def chl(c, inputs, targets,
        rate=0.2, error=10e-4, max_epochs=10e4,
        func=chl_synchronous, verbose=False):
    """Performs contrastive Hebbian learning of a specified pattern"""
    c.Update(verbose=False)
    A = [x.activations for x in c.GetOutput()]
    e = sum(map(Error, A, targets))
    i = 0
    if verbose:
        print "[%d] %s" % (i, e)
    while (e > error and i < max_epochs):
        i+=1
        func(c, inputs, targets, rate)
        c.Update(verbose=False)
        A = [x.activations for x in c.GetOutput()]
        e = sum(map(Error, A, targets))
        if verbose:
            print "[%d] %s" % (i, e)
    return i


def chla(c, inputs, targets,
         rate=0.2, error=10e-4, max_epochs=10e4,
         func=chl_synchronous, verbose=False):
    """Performs contrastive Hebbian learning of a specified pattern"""
    e = []
    i = 0
    for ins, tgts in zip(inputs, targets):
        c.SetInputActivations(ins, clamped=True)
        c.Update(verbose=False)
        A = [x.activations for x in c.GetOutput()]
        e.append(sum(map(Error, A, tgts)))
    if verbose:
        print "[%d] %s" % (i, max(e))
    while (max(e) > error and i < max_epochs):
        i += 1
        e  = []
        pattern_set = copy.copy(zip(inputs, targets))
        random.shuffle(pattern_set)
        for ins, tgts in pattern_set:
            func(c, ins, tgts, rate)
            c.Update(verbose=False)
            A = [x.activations for x in c.GetOutput()]
            #e.append(sum(map(Error, A, tgts)))
            # Redo all the tests insetad!
            e.append(test(c, inputs, targets, error))
        if verbose:
            print "[%d] %s" % (i, max(e))
    return i
    
    
def test(c, input_set, target_set, error=10e-4, verbose=False):
    """Tests the correct learning of a series of patterns"""
    e = []
    i = 0
    for ins, tgts in zip(input_set, target_set):
        i += 1
        c.SetInputActivations(ins, clamped=True)
        c.Update(verbose=False)
        A = [x.activations for x in c.GetOutput()]
        current_error = sum(map(Error, A, tgts))
        if verbose:
            if current_error < error:
                print("Pattern %d: Pass" % i)
            else:
                print("Pattern %d: *Failed*" % i)
        e.append(current_error)
    results = np.array(e) < error
    return max(e)
        


## ---------------------------------------------------------------- ##
## BACKPROPAGATION
## ---------------------------------------------------------------- ##

def backprop(c, inputs, targets, rate=0.2):
    pass
