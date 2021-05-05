'''import statements'''
import numpy as np
import scipy.io

'''
This script computes a binding score for a given sequence or set of sequences

> Inputs: DNA sequence in letter format
> Outputs: Sequence binding scores

To-Do:
==> automate MD sampling
==> Potts Hamiltonian is giving unexpectedly small energies
'''

class oracle():
    def __init__(self,params):
        '''
        initialize the oracle
        :param params:
        '''
        self.params = params


    def score(self, queries):
        '''
        assign correct scores to selected sequences
        :param queries: sequences to be scored
        :return: computed scores
        '''
        #return self.PottsEnergy(queries)
        return self.toyEnergy(queries)

    
    def toyEnergy(self,queries):
        '''
        return the energy of a toy model for the given set of queries
        :param queries:
        :return:
        '''
        np.random.seed(int(queries.shape[-1] * self.params['random seed'])) # this ensures we get the same energy function for the same initial conditions
        hamiltonian = np.random.randn(queries.shape[-1],queries.shape[-1]) # energy function
        energies = np.zeros(len(queries))

        for i in range(len(queries)):
            energies[i] = queries[i] @ hamiltonian @ queries[i].transpose() # compute energy for each sample

        return energies


    def initializeDataset(self,inputSize, numSamples):
        '''
        generate an initial toy dataset with a given number of samples
        :param numSamples:
        :return:
        '''
        data = {}
        np.random.seed(int(numSamples * inputSize * self.params['random seed']))
        data['sequences'] = np.random.randint(0,2,size=(numSamples, inputSize)) # samples are a binary set
        data['scores'] = self.toyEnergy(data['sequences'])

        np.save('datasets/' + self.params['dataset'], data)


    def PottsEnergy(self, queries):
        '''
        test oracle - learn one of our 40-mer Potts Hamiltonians
        :param queries: sequences to be scored
        :return:
        '''

        # DNA Potts model
        ''''''
        coupling_dict = scipy.io.loadmat('40_level_scored.mat')

        N = coupling_dict['h'].shape[1] # length of DNA chain

        assert N == len(queries[0]), "Hamiltonian and proposed sequences are different sizes!"

        h = coupling_dict['h']
        J = coupling_dict['J']

        energies = []
        ''''''
        for sequence in queries:
            energy = 0

            # potts hamiltonian
            '''
            l = 0
            for i in range(N - 1):

                for j in range(i + 1, N):
                    energy += J[sequence[i], sequence[j], l]  # site-specific couplings
                    l += 1

            for i in range(len(sequence)):
                energy += h[sequence[i], i]  # site-specific base energy
            '''
            # simpler hamiltonian
            energy = np.sum(sequence)

            energies.append(energy)

        return np.asarray(energies)