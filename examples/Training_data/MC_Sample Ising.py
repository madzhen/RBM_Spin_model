import numpy as np
import matplotlib.pyplot as plt

Spin_number = 6

#i.e : transfer  63 to [ 1,1,1,1,1,1]     transfer 1 to [-1,-1,-1,-1,-1,1]
def Return_SpinConfiguration_array(Decimal_num, N_number_of_spin_sites=Spin_number):
    spin_array = np.asarray(
        [1 if int(x) else -1 for x in
         "".join([str((Decimal_num >> y) & 1) for y in range(N_number_of_spin_sites - 1, -1, -1)])])
    return spin_array

#i.e : transfer   [ 1,1,1,1,1,1]   to 63,   transfer [0,0,0,0,0,1] to 1
def Return_SpinarrayToDec(spin_array):
    result = 0
    for i in range(Spin_number):
        result = result * 2 + (1 if spin_array[i] == 1 else 0)
    return result


def Hamiltonian(sigma):
    H = 0
    list_sigma = Return_SpinConfiguration_array(sigma)
    for i in range(Spin_number - 1):
        H += list_sigma[i] * list_sigma[i + 1]
    return -1 * H


def Z_s(Spin_number):

    Z = 0
    for i in range(2 ** Spin_number):
        Z += np.exp(-1 * Hamiltonian(i))
    return Z

Z_ss=Z_s(Spin_number)
def P_s_density(sigma):
    return np.exp(-1 * Hamiltonian(sigma)) / Z_ss



#######################################################################################


def check_move(p_prime, p, s_prime, s):
    if p / p_prime >= 1:
        return np.copy(p), np.copy(s)
    else:
        rand = np.random.uniform(0, 1)
        if p / p_prime + rand >= 1:
            return np.copy(p), np.copy(s)
        else:
            return np.copy(p_prime), np.copy(s_prime)


#  transfer [-1,-1,-1,1,1,1] to [0,0,0,1,1,1]
def transfer_Mi1_to_zero(s_array):
    return np.asarray([  (1 if i==1 else 0) for i in s_array   ])


def Metroplis_algorithm_walkers(Nsteps,threshold, m, walkers):
    a_total = 0
    samples = []

    sigma_begin = np.random.randint(0, 2**Spin_number)
    s_prime = np.asarray([Return_SpinConfiguration_array(sigma_begin) for w in range(walkers)])
    p_prime = np.asarray([P_s_density(Return_SpinarrayToDec(s_prime[w])) for w in range(walkers)])

    s = np.zeros(shape=s_prime.shape)
    p = np.zeros(shape=p_prime.shape)

    for i in range(Nsteps):
        if(i%10000==0):
            print(i)
        for w in range(walkers):

            s[w] =np.copy( s_prime[w])
            (s[w])[np.random.randint(0,Spin_number)]*=(-1)

            p[w] = P_s_density(Return_SpinarrayToDec(s[w]))

            a = min(1, p[w] / p_prime[w])
            a_total += a

            p_prime[w], s_prime[w] = check_move(p_prime[w], p[w], s_prime[w], s[w])

            if ((i % m == 0) and (i>=threshold )):
                samples.append(transfer_Mi1_to_zero(np.copy(s_prime[w])))
                #samples.append(Return_SpinarrayToDec(np.copy(s_prime[w])))
    return np.asarray(samples), a_total / Nsteps / walkers * 100.0





samples_final,acceptance=Metroplis_algorithm_walkers(210000,10000,2,1)

np.save('Ising_demo.npy',samples_final)

#np.savetxt('Ising_MC_Value.csv', samples_final, delimiter = ',')  
