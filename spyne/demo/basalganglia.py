# ================================================================== #
# BASALGANGLIA
# ================================================================== #
# A mockup version of the basal ganglia circuit for demo purposes
# ================================================================== #

import numpy as np
from ..neural import *

def CreateBasalGanglia(regions=5, n_cortex=100, n_tans=10, n_str=20,
                       n_gpe=20, n_snrgpi=10, n_thal=10):
    bg       = Circuit()
    tans     = Group(n_tans*regions**2,   name=GenTemp("TANS-"))
    sp       = Group(n_str*regions**2,    name=GenTemp("SP-"))
    sn       = Group(n_str*regions**2,    name=GenTemp("SN-"))
    snr_gpi  = Group(n_snrgpi*regions**2, name=GenTemp("SN/GPI-"))
    gpe      = Group(n_gpe*regions**2,    name=GenTemp("GPE-"))
    thal     = Group(n_thal*regions**2,   name=GenTemp("THAL-"))
    
    bg.AddGroups([tans, sn, sp, snr_gpi, gpe, thal])
    bg.SetInput(tans)
    bg.SetInput(sn)
    bg.SetInput(sp)

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
