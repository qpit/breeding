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

from breeding import sim_breeding_circuit, sample_breeding_circuit
from lcg_plus.effective_sqz import effective_sqz
from lcg_plus.conversions import Delta_to_dB

from catprep import get_input_sq_cat
import numpy as np
from time import time

t0 = time()

ns = np.arange(8,14)
etas = np.array([1, 0.99, 0.98, 0.97, 0.96, 0.95])
r_dBs = np.array([-12,-15])
parity = 0
shots = 100

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

tf = time()
print(tf-t0)
    
