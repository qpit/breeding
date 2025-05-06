import numpy as np
from copy import copy, deepcopy
from thewalrus.symplectic import squeezing, beam_splitter, rotation, xxpp_to_xpxp, expand, expand_vector
from mpmath import mp



def sim_breeding_circuit(input_state, num, phis, results, out = False, MP = False, bs_thetas = None, rot =False):
    """Simulate the breeding of num copies of the input_state
    """
    multistate = deepcopy(input_state)
        
    if out:
        print('Input data shape', multistate.means.shape, multistate.covs.shape, multistate.weights.shape)
        print('norm', multistate.norm)
    
    for i in range(1,num):
        multistate.add_state(input_state, MP=MP)
        
        if out:
            print('norm', multistate.norm)
        #if rot == True:
            #R = xxpp_to_xpxp(expand(rotation(np.pi),1,2)) #Rotate the 2nd mode by 180 degrees
            #multistate.apply_symplectic(R)
        if out:
            print('newstate shape', multistate.means.shape, multistate.covs.shape, multistate.weights.shape)

        #Apply custom beamsplitters if given
        if bs_thetas:
            S = beam_splitter(bs_thetas[i-1][0],bs_thetas[i-1][1])
            if out:
                print(f'Applying BS({bs_thetas[i-1][0]}{bs_thetas[i-1][1]}) on modes {i-1},{i}')
        #Apply 
        else:
            #Apply beam splitter
            S = beam_splitter(np.arccos(np.sqrt(1/(1+i))),0)
            if out:
                print(f'Applying BS({np.arccos(np.sqrt(1/(1+i)))} on modes {i-1},{i}')

        multistate.apply_symplectic(xxpp_to_xpxp(S))
        
        if out:
            print('num_k', multistate.num_k, 'num_weights', multistate.num_weights, 'norm', multistate.norm)
       
        multistate.post_select_homodyne(0, phis[i-1], results[i-1], MP)
        
          
        if out:
            print(f'Measure p={results[i-1]} on mode {i-1} with conditional prob {multistate.norm}')
    
        #Reduce
        multistate.reduce_equal_means(MP)
        
        if out:
            print('new no. of weights: ', multistate.num_weights)
            print('norm', multistate.norm)
            
    return multistate, multistate.norm

def sample_breeding_circuit(input_state, num, out = False, MP = False, bs_thetas = None):
    """Simulate the breeding of num copies of the input_state
    """
    samples = np.zeros(num-1)
    rejects = np.zeros(num-1)
    multistate = deepcopy(input_state)
        
    for i in range(1,num):
        multistate.add_state(input_state, MP=MP)

        #Apply custom beamsplitters if given
        if bs_thetas:
            S = beam_splitter(bs_thetas[i-1][0],bs_thetas[i-1][1])
            if out:
                print(f'Applying BS({bs_thetas[i-1][0]}{bs_thetas[i-1][1]}) on modes {i-1},{i}')
        #Apply inverse beamsplitter cascade
        else:
            #Apply beam splitter
            S = beam_splitter(np.arccos(np.sqrt(1/(1+i))),0)
            if out:
                print(f'Applying BS({np.arccos(np.sqrt(1/(1+i)))} on modes {i-1},{i}')

        #Rotate mode 0 by 90 degrees to sample prepare p-quadrature sampling
        R = expand(rotation(-np.pi/2),[0], 2)

        multistate.apply_symplectic(xxpp_to_xpxp(R@S))

        sample, reject = multistate.sample_dyne([0], shots = 1, MP=MP)
        sample = sample[0,0]
       

        
        multistate.post_select_homodyne(0, 0, sample, MP)
        samples[i-1] = sample
        rejects[i-1] = len(reject) #store the number of rejections
        
        if out:
            print(f'Measure p={sample} on mode {i-1} with conditional prob {multistate.norm}')
    
        #Reduce
        multistate.reduce_equal_means(MP)
        
        if out:
            print('new no. of weights: ', multistate.num_weights)
            print('norm', multistate.norm)
            
    return multistate, multistate.norm, samples, rejects


def multi_breed_state(input_state, num, out = False, MP = False, bs_thetas = None, rot =False):
    """Return the multimode output state after the beamsplitter cascade before homodyning
    """

    multistate = copy(input_state)
        
    if out:
        print('Input data shape', multistate.means.shape, multistate.covs.shape, multistate.weights.shape)
    
    for i in range(1,num):
        multistate.add_state(input_state, MP=MP)
        if rot == True:
            R = xxpp_to_xpxp(expand(rotation(np.pi),1,2)) #Rotate the 2nd mode by 180 degrees
            multistate.apply_symplectic(R)
        if out:
            print('newstate shape', multistate.means.shape, multistate.covs.shape, multistate.weights.shape)

        if bs_thetas:
            S = beam_splitter(bs_thetas[i-1][0],bs_thetas[i-1][1])
            if out:
                print(f'Applying BS({bs_thetas[i-1][0]}{bs_thetas[i-1][1]}) on modes {i-1},{i}')
        else:
            #Apply beam splitter
            S = beam_splitter(np.arccos(np.sqrt(1/(1+i))),0)
            if out:
                print(f'Applying BS({np.arccos(np.sqrt(1/(1+i)))} on modes {i-1},{i}')

        multistate.apply_symplectic(xxpp_to_xpxp(S))
            
        
    return multistate