# Test.py: A simple test of Spyne's capabilities

#import spyne.gui.visualize
import spyne.gui.gui
import spyne.demo.basalganglia

model = spyne.demo.basalganglia.CreateBasalGanglia()
m = spyne.demo.basalganglia.CreateSimple()

spyne.gui.gui.Show(m)
