## ---------------------------------------------------------------- ##
## NAMING MECHANISM
## ---------------------------------------------------------------- ##
## Something akin to what GenTemp is for Lisp.
## ---------------------------------------------------------------- ##

# --- Gentemp dictionary ---------------------------------------------

#__all__ = ['GenTemp']

_GT_NAMES_ = {}

def GenTemp(root="obj-"):
    """Generates a new unique name based on the given root"""
    global _GT_NAMES_
    if root in _GT_NAMES_.keys():
        _GT_NAMES_[root]+=1
    else:
        _GT_NAMES_[root]=0
    return "%s%d" % (root, _GT_NAMES_[root])

def gentemp(root="obj-"):
    """Generates a new unique name based on the given root (alias for GenTemp)"""
    return GenTemp(root)
