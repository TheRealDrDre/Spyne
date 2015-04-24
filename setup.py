## A setup file for the SPyNE package.
## 

from distutils.core import setup
setup(name='spyne',
      version='0.1',
      author="Andrea Stocco",
      author_email="stocco@uw.edu",
      packages=['spyne', 'spyne.gui', 'spyne.demo'],
      )