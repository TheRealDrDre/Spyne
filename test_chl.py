from spyne.neural import *
import numpy as np

g1 = Group(5)
g2 = Group(15)
g3 = Group(2)

g1g2 = g1.ConnectTo(g2)
g2g3 = g2.ConnectTo(g3)
g3g2 = g3.ConnectTo(g2)

g1g2.weights = np.random.random((15, 5)) / 100
g2g3.weights = np.random.random((2, 15)) / 100
g3g2.weights = np.random.random((15, 2)) / 100

c = Circuit()
c.AddGroups([g1, g2, g3])
c.SetInput(g1)
c.SetOutput(g3)

pattern = np.array([1, 0, 1, 0, 1])
pattern.shape  = (5,1)

target = np.array([1, 0])
target.shape  = (2,1)