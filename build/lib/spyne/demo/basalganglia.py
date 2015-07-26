# ================================================================== #
# BASALGANGLIA
# ================================================================== #
# A mockup version of the basal ganglia circuit for demo purposes
# ================================================================== #

import numpy as np
from ..neural import *

def CreateBasalGanglia(regions=5, n_cortex=100, n_tans=10, n_str=20,
                       n_gpe=20, n_snrgpi=10, n_thal=10):
    """
Creates a mockup basal ganglia circuit, using the structure proposed in
Stocco, Lebiere, & Anderson (2010) as a template.
    """
    bg       = Circuit()
    tans     = Group(n_tans * regions ** 2,   name=GenTemp("TANS-"))
    sp       = Group(n_str * regions ** 2,    name=GenTemp("SP-"))
    sn       = Group(n_str * regions ** 2,    name=GenTemp("SN-"))
    snr_gpi  = Group(n_snrgpi * regions ** 2, name=GenTemp("SN/GPI-"))
    gpe      = Group(n_gpe * regions ** 2,    name=GenTemp("GPE-"))
    thal     = Group(n_thal * regions ** 2,   name=GenTemp("THAL-"))
    
    bg.AddGroups([tans, sn, sp, snr_gpi, gpe, thal])
    bg.SetInput(tans)
    bg.SetInput(sn)
    bg.SetInput(sp)
    bg.SetOutput(thal)

    p=tans.ConnectTo(sn)
    p.weights=np.random.random((sn.size, tans.size))
    
    p=tans.ConnectTo(sp)
    p.weights=np.random.random((sp.size, tans.size))

    p=sn.ConnectTo(snr_gpi)
    p.weights=np.random.random((snr_gpi.size, sn.size))

    p=sp.ConnectTo(gpe)
    p.weights=np.random.random((gpe.size, sp.size))

    p=gpe.ConnectTo(snr_gpi)
    p.weights=np.random.random((snr_gpi.size, gpe.size))
    
    return bg

def CreateSimple():
    c = Circuit()
    g1 = Group(10)
    g2 = Group(8)
    p = Projection(g1, g2)
    p = Projection(g2, g1)
    c.AddGroups([g1, g2])
    c.SetInput(g1)
    c.SetOutput(g2)
    return c

def CreateSimple(regions=1, n_cortex=100, n_str=20,
                 n_gpe=20, n_snrgpi=10, n_thal=10):
    bg       = Circuit()
    sp       = Group(n_str * regions ** 2,    name=GenTemp("SP-"))
    sn       = Group(n_str * regions ** 2,    name=GenTemp("SN-"))
    snr_gpi  = Group(n_snrgpi * regions ** 2, name=GenTemp("SN/GPI-"))
    gpe      = Group(n_gpe * regions ** 2,    name=GenTemp("GPE-"))
    thal     = Group(n_thal * regions ** 2,   name=GenTemp("THAL-"))
    
    #sp.geometry = (5, 4)
    #sn.geometry = (5, 4)
    #gpe.geometry = (5, 4)
    #thal.geometry = (5, 2)
    #snr_gpi.geometry = (5, 2) 
    
    bg.AddGroups([sn, sp, snr_gpi, gpe, thal])
    bg.SetInput(sn)
    bg.SetInput(sp)
    bg.SetOutput(thal)

    p1 = sn.ConnectTo(snr_gpi)
    p1.weights=np.random.random((snr_gpi.size, sn.size))

    p2 = sp.ConnectTo(gpe)
    p2.weights=np.random.random((gpe.size, sp.size))

    p3 = gpe.ConnectTo(snr_gpi)
    p3.weights=np.random.random((snr_gpi.size, gpe.size))
    
    p4 = snr_gpi.ConnectTo(thal)
    p4.weights = np.random.random((snr_gpi.size, thal.size))

    return bg