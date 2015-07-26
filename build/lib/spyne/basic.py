## ---------------------------------------------------------------- ##
## BASIC.PY
## ---------------------------------------------------------------- ##
## The basic building blocks of the SPyNE package.
## ---------------------------------------------------------------- ##
## This fiels defines a number of foundation objects for the SPyNE
## package. These objects incluce the SPyNEObject (root of all
## objects), the ParametrizedObject, and the ContextContainer.
## ---------------------------------------------------------------- ##

__all__ = ['SPyNEObject', 'ParametrizedObject', 'ContextContainer']

## Basic implementation of a SPyNE object.
##
## SPyne objects are naturally provided with a number
## of (more or less) useful features.  These include
## a list of notifiers that can be invoked when changes
## to their structure occurs.  Works much like a Java
## delegation model.  Might kill performance but we will
## not know until we try


## --- The SPyNEObject -----------------------------------------------
## The SPyNEObject is the root object for the package. 
## 
class SPyNEObject (object):
    """A basic SPyNEObject"""
    def __init__(self):
        self._notifiables = []

    def AddNotifiable(self, function):
        """Adds a notifieable function"""
        if function is not None:
            self.GetNotifiables().append(function)

    def RemoveNotifiable(self, function):
        """
        Removes a notifiable function F from the list of
        functions---if the list contains F.
        """
        if function in self._notifiables:
            self._notifiables.remove(function)

    def GetNotifiables(self):
        """Returns the list of notifable functions"""
        return self._notifiables

    def Notify(self):
        """Invokes the notifiable functions one at a time"""
        for f in self._notifiables:
            f(self)
        
class ParametrizedObject (object):
    """An object with parameters"""
    def __init__(self):
        self.__parameters = {}

    def HasParameter(self, parameter):
        """Checks whether it contains the given parameter"""
        return parameter in self.__parameter.keys()

    def SetParameters(self, parameters):
        """
        Sets the internal dictionary of paramenters to the desired 
        value
        """
        self.__parameters = parameters

    def GetParameters(self):
        """Returns the entire internal dictionary of parameters"""
        return self.__parameters

    def SetParameter(self, key, value):
        """
        Sets the values of the parameter 'key' to the given value
        """
        self.__parameters[key]=value

    def GetParameter(self, key):
        """Returns the valyes of the given parameter"""
        return self.__parameters[key]

class ContextContainer:
    """An object that contains an execution context"""
    __context = None
    def __init__(self, context=None):
        self.__context = context

    def SetContext(self, context):
        self.__context = context

    def GetContext(self):
        return self.__context
