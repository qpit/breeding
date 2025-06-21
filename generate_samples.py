from breeding import sim_breeding_circuit, sample_breeding_circuit
from bosonicplus.effective_sqz import effective_sqz
from bosonicplus.conversions import Delta_to_dB

from catprep import get_input_sq_cat
import numpy as np


#Settings 1
ns = np.arange(2,14)
etas = np.array([0.96, 0.95])
r_dBs = np.array([-12,-15])
parity = 0
shots = 100000

times = np.zeros((len(ns),len(etas)))


def perform_sampling(*args, shots = 1):

    #Define input cat state
    cat = get_input_sq_cat(*args)
    num = args[0]
    which = args[3]
    n = 0

    Dp = np.zeros(shots)
    Dx = np.zeros(shots)
    samples = np.zeros((shots,num-1))

    while n < shots:
        out, norm, vals, rej = sample_breeding_circuit(cat, num)
        samples[n,:] = vals
        
        Dp[n] = effective_sqz(out, which[0]+'p')
        Dx[n] = effective_sqz(out, which[0]+'x')
        n+=1
    
    return samples, Dp, Dx

for i, num in enumerate(ns):
    for j, eta in enumerate(etas):
        for k, r_dB in enumerate(r_dBs):
            args = num, r_dB, parity, 'square', eta
            samples, Dp, Dx = perform_sampling(*args, shots = shots)
            np.save(f'samples_num={num}_r={r_dB}_eta={eta}.npy', samples)
            np.save(f'Dp_num={num}_r={r_dB}_eta={eta}.npy', Dp)
            np.save(f'Dx_num={num}_r={r_dB}_eta={eta}.npy', Dx)


    
