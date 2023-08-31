
import numpy as np
# from matplotlib import pyplot as plt


def StandardDragCurve(Re):
    """
    Standard Drag Curve following expressions of Clift, Grace and
    Weber (1978). Bubbles, Drops, and Particles. Dover Publications.

    See Table 5.2.
    """
    
    if Re == 0:
        Cd = 1E3

    elif Re < 0.01: # (A)

        Cd = 3./16 + 24./Re

    elif Re <= 20: # (B)
        w = np.log10(Re)
        auxB = 1.+0.1315*Re**(0.82-0.05*w)
        Cd =24./Re*auxB
    elif Re <= 260: # (C)
        w = np.log10(Re)
        auxC = 1.+0.1935*Re**0.6305
        Cd = 24./Re*auxC

    elif Re <= 1500: # (D)
        w = np.log10(Re)
        auxD = 1.6435 - 1.1242*w + 0.1558*w*w
        Cd = 10**auxD

    elif Re <= 1.2E4: # (E)
        w = np.log10(Re)
        auxE = -2.4571 + 2.5558*w - 0.9295*w*w + 0.1049*w*w*w
        Cd = 10**auxE

    elif Re <= 4.4E4: # (F)
        w = np.log10(Re)
        auxF = -1.9181 + 0.6370*w - 0.0636*w*w
        Cd = 10**auxF

    elif Re <= 3.38E5: # (G)
        w = np.log10(Re)
        auxG = -4.3390 + 1.5809*w - 0.1546*w*w
        Cd = 10**auxG

    elif Re <= 4E5: # (H)
        w = np.log10(Re)
        Cd = 29.78-5.3*w

    elif Re <= 1E6:  # (I)
        w = np.log10(Re)
        Cd = 0.1*w-0.49

    elif Re <= 1E6: # (J)

        Cd = 0.19-8E4 /Re
  
    else: # over limits, extraploation needed
        Cd = np.nan
    return Cd
