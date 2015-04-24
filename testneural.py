# TestNeural.py: A simple test of Spyne's Neural object capabilities

from spyne.neural import *

## ---------------------------------------------------------------- ##
## TEST
## ---------------------------------------------------------------- ##

## DEMO NETWORK
g1 = Group(200)
g2 = Group(100)
g3 = Group(100)
g4 = Group(80)

p1 = Projection(g1, g2)
p1.weights=np.random.random((100,200))/5 - .1
p2 = Projection(g2, g3)
p2.weights=np.random.random((100,100))/5 - .1
p3 = Projection(g3, g4)
p3.weights=np.random.random((80,100))/5 - .1
p4 = Projection(g3, g2)
p4.weights=np.random.random((100,100))/5 -.1
p5 = Projection(g4, g3)
p5.weights=np.random.random((100,80))/5 - .1

pattern1 = Step(np.random.random((200,1)), .5)
g1.activations=np.array(pattern1)
g1.SetClamped(True)

c=Circuit()
c.AddGroups([g1, g2, g3, g4])
c.SetInput(g1)
c.SetOutput(g3)

## A more complex example. Mockup basal ganglia
##
b1 = Group(1000, name="Cortex")
b2 = Group(500, name="Str")   # Striatum
b3 = Group(200, name="Gpi/Snr")   # Gpi/SNr
b4 = Group(300, name="Gpe")   # Gpe
b5 = Group(100, name="Thal")   # Thal
b6 = Group(50, name="STN")     # STN

bp1 = Projection(b1, b2)
bp1.weights=np.random.random((500,1000))/5 - .1

bp2 = Projection(b2, b3)
bp2.weights=np.random.random((200,500))/5 - .1

bp3 = Projection(b2, b4)
bp3.weights=np.random.random((300,500))/5 - .1

bp4 = Projection(b4, b6)
bp4.weights=np.random.random((50,300))/5 - .1

bp5 = Projection(b6, b3)
bp5.weights=np.random.random((200,50))/5 - .1

bp6 = Projection(b3, b5)
bp6.weights=np.random.random((100,200))/5 - .1

pattern2 = Step(np.random.random((1000,1)), .5)
b1.activations=np.array(pattern2)
b1.SetClamped(True)

bg=Circuit()
bg.AddGroups([b1, b2, b3, b4, b5, b6])
bg.SetInput(b1)
bg.SetOutput(b5)

### A 3-layer network for CHL
###
h1=Group(20, name="chl-1")
h2=Group(20, name="chl-2")
h3=Group(20, name="chl-3")

n=Circuit()
n.AddGroups([h1, h2, h3])
n.SetInput(h1)
hp1=Projection(h1, h2)
hp1.weights=np.random.random((20,20))/5 - .1

hp2=Projection(h2, h3)
hp2.weights=np.random.random((20,20))/5 - .1

hp3=Projection(h3, h2)
hp3.weights=np.random.random((20,20))/5 - .1

input1 = Step(np.random.random((20,1)), .5)
input2 = Step(np.random.random((20,1)), .5)

target1 = Step(np.random.random((20,1)), .5)
target2 = Step(np.random.random((20,1)), .5)


h1.activations=input1
h1.SetClamped(True)
n.SetOutput(h3)
