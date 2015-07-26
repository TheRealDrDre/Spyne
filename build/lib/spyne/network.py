
class Nucleus(Group, ParametrizedObject):
    """A nucleus (a structured set of Groups)"""
    def __init__(self, size=0, structure=(1,1), name=None,
                 activationFunction=Linear,
                 activationDerivative=None,
                 updateFunction=GenericUpdate):
        Group.__init__(self, size=0, name=None,
                       activationFunction=Linear,
                       activationDerivative=None,
                       updateFunction=GenericUpdate)
        if name is None:
            self.name=GenTemp("Nucleus-")
        self.structure=structure


class Network(NeuralObject, ParametrizedObject):
    """A network"""
    def __init__(self, groups={}, projections={}):
        self.__groups={}
        self.__projections={}

    def GetGroups(self):
        return self.__groups

    def AddGroup(self, name, group):
        if not name in self.__groups.keys():
            P = {}
            self.__groups[name]=group
            self.__projections[name]=P
            for g in self.__groups.keys():
                P[g] = None
                if g is not name:
                    self.__projections[g][name]=None

    def GetGroup(self, name):
        if name in self.__groups.keys():
            self.__groups[name]

    def RemoveGroup(self, name):
        if name in self.__groups.keys():
            self.__groups.pop(name)

    def GetProjections(self):
        return self.__projections
        
