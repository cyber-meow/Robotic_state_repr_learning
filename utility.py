
"""
Utility functions
"""

def setAllArgs(obj, argdict):
    for k in argdict.keys():
        if hasattr(obj, k):
            setattr(obj, k, argdict[k])
        else:
            print("Warning: parameter name {} not found!".format(k))

