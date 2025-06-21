from bosonicplus.base import State
from thewalrus.symplectic import squeezing, beam_splitter, rotation, xxpp_to_xpxp, expand, expand_vector
import numpy as np
from bosonicplus.conversions import dB_to_r
from bosonicplus.states.nongauss import prepare_sqz_cat_coherent

hbar = 2
# Two mode General photon subtraction circuit with twice the squeezing in one mode
def GPS_Andersen(r, theta, phi, n, alpha = 0, inf = 1e-8):
    """
    Args: 
        r : squeezing
        theta : beam splitter transmittivity angle
        phi: rotation before the beamspltter
        alpha: disp in mode 0
    """
    state = State(2)
    S1 = expand(squeezing(r, 0), 0, 2)
    S2 = expand(squeezing(r, np.pi), 1, 2)
    R1 = expand(rotation(phi), 0,2)
    BS = beam_splitter(theta,0)
    R2 = expand(rotation(-phi/2 + np.pi/2),1,2)
    Stot = xxpp_to_xpxp(R2 @ BS @ R1 @ S2 @ S1)
    state.apply_symplectic(Stot)
    
    disp = expand_vector(alpha, 0, 2)
    

    state.apply_displacement(disp)
    

    state.post_select_fock_coherent(0, n, inf = inf)
    return state


def get_input_sq_cat(num, r_dB, parity, which, eta, fast = False):
    """Get squeezed cat for input to breeding circuit
    """
    
    r_cat = dB_to_r(r_dB)
    
    if which == 'rect':
        alpha = np.sqrt(num)*np.sqrt(np.pi/2)*np.exp(r_cat) #Rectangular
    elif which == 'square':
        alpha = np.sqrt(num)*np.sqrt(np.pi)/2*np.exp(r_cat) #Square
    elif which == 'hex':
        alpha = np.sqrt(num)*np.sqrt(np.pi)*np.exp(r_cat) #Rectangular
        
    #alpha = np.sqrt(num)*np.sqrt(np.pi * np.sqrt(3)/2)*np.exp(r_cat) #Hexagonal
    #alpha = np.sqrt(num)*np.sqrt(np.pi/hbar * 2/np.sqrt(3))*np.exp(r_cat) #Hexagonal 
        
    sq_cat = prepare_sqz_cat_coherent(r_cat, alpha, parity, fast)

    if which =='hex':
        sq_cat.apply_symplectic(squeezing(np.log(3**(1/4)), 2*np.pi/4))
  
    #Apply loss to squeezed cat
    #---------------------------
    if eta != 1:
        sq_cat.apply_loss(eta, 0)
    
    return sq_cat