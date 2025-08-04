# Copyright © 2025 Technical University of Denmark

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from lcg_plus.base import State
from lcg_plus.operations.symplectic import squeezing, rotation, beam_splitter, expand_displacement_vector
import numpy as np
from lcg_plus.conversions import dB_to_r
from lcg_plus.states.nongauss import prepare_sqz_cat_coherent


# Two mode General photon subtraction circuit
def GPS_circuit(r, theta, phi, n, alpha = 0, inf = 1e-8):
    """
    Args: 
        r : squeezing
        theta : beam splitter transmittivity angle
        phi: rotation before the beamspltter
        alpha: disp in mode 0
    """
    state = State(2)
    state.apply_symplectic_fast(squeezing(r,0), [0])
    state.apply_symplectic_fast(squeezing(r,np.pi), [1])
    state.apply_symplectic_fast(rotation(phi), [0])
    state.apply_symplectic(beam_splitter(theta,0))
    state.apply_symplectic_fast(rotation(-phi/2+np.pi/2), [1])
    disp = np.sqrt(2*state.hbar) * np.array([alpha.real,alpha.imag])
    state.apply_displacement(expand_displacement_vector(disp, [0], 2))

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

    #Scale the means by 1/sqrt(eta)
  
    #Apply loss to squeezed cat
    #---------------------------
    if eta != 1:
        sq_cat.apply_loss(eta, 0)
    
    return sq_cat