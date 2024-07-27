# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:59:22 2024

@author: Kojo Nketia
"""

import time
import numpy as np
import scipy.stats as stats

from matplotlib import pyplot as plt
from math import sqrt, floor, exp, log


class Host:
    '''
	This is the host class that contains the dynamics of 
    worms and microfilaria in humans
	'''
    def __init__(self):
        '''
		Initialise state variables for humans
		'''
        self.b = 0  # risk of infection
        self.t = 0  # time of host
        self.initialise()
        self.age = 0  # age of host
        self.pRisk = 0  # p-value for the risk of infection (updates b)
        self.uComp = 0  # random parameter used to calculate compliance with MDA
        self.uCompN = 0 # random parameter used to calculate compliance with bed nets
        self.bedNet = 0 # host used bednet
        np.random.seed(int(time.time()))


    def react(self):
        '''
		Update microfilariae and worm dynamics over time
		'''
        bNReduction = 1 - (1 - params['sN']) * self.bedNet
        self.I = self.immuneRK4Step(self.W, self.I)
        # male worms update
        maleBirths = self.poisson_dist(0.5 * bNReduction * params['xi'] * self.biteRate() * params['L3'] * exp(-1 * params['theta'] * self.I) * self.b * params['dt'])
        maleDeaths = self.poisson_dist(params['mu'] * (1 + self.aWol) * self.WM * params['dt'])
        self.WM += maleBirths - maleDeaths

        # female worm update
        femaleBirths = self.poisson_dist(0.5 * bNReduction * params['xi'] * self.biteRate() * params['L3'] * exp(-1 * params['theta'] * self.I) * self.b * params['dt'])
        femaleDeaths = self.poisson_dist(params['mu'] * (1 + self.aWol) * self.WF * params['dt'])
        self.WF += femaleBirths - femaleDeaths

        #Mf update
        self.M += params['dt'] * (self.repRate() * (1 - self.aWol) - params['gamma'] * self.M)

        # total worm count
        self.W = self.WM + self.WF

        # time-step
        self.t += params['dt']
        self.age += params['dt']

        # ensure all positive state variables remain positive
        if self.W < 0: self.W = 0
        if self.WM < 0: self.WM = 0
        if self.WF < 0: self.WF = 0
        if self.I < 0: self.I = 0
        if self.M < 0: self.M = 0

        # simulate an event where host dies and is replaced by a new host
        if self.uniform_dist() < (1 - exp(-1 * params['tau'] * params['dt'])) or self.age > 1200: # if age over 100
            self.initialise()
            self.age = 0 # birth event so age is 0

    def evolve(self, tot_t):
        while self.t < tot_t:
            self.react()

    def initialise(self):
        self.W = 0  # number of worms
        self.WM = 0 # number of male worms
        self.WF = 0 # number of female worms
        self.I = 0  # immune response (assumed to be deterministic)
        self.M = 0  # mf produced. - integer
        self.bedNet = 0 # time of host
        self.aWol = 0   # host is treated with Doxycycline

    def mfConc(self): 
        '''
        returns concentration of the mf as opposed to absolute number
        '''
        return self.M  # 0.005

    def biteRate(self):
        if self.age < 108:  # less than 9 * 12 = 108
            return self.age / 108
        else:
            return 1

    def repRate(self):
        if params['nu'] == 0:
            if self.WM > 0:
                return self.WF
            else:
                return 0
        else:
            return params['alpha'] * np.min([self.WF, (1/params['nu']) * self.WM])

    def updateRisk(self, init = False):
        if init:
            if self.age < 240:   # age under 20
                self.b = stats.gamma.ppf(self.pRisk, params['shapeRisk'], scale=params['riskMu1']/params['shapeRisk'])
            elif self.age < 348: # age between 20 and 29
                self.b = stats.gamma.ppf(self.pRisk, params['shapeRisk'], scale=params['riskMu2']/params['shapeRisk'])
            else:  # age 30+
                self.b = stats.gamma.ppf(self.pRisk, params['shapeRisk'], scale=params['riskMu3']/params['shapeRisk'])
        else:
            if self.age < 12: # age at 0
                self.b = stats.gamma.ppf(self.pRisk, params['shapeRisk'], scale=params['riskMu1']/params['shapeRisk'])
            elif self.age >= 240 and self.age < 252:  # age 25
                self.b = stats.gamma.ppf(self.pRisk, params['shapeRisk'], scale=params['riskMu2']/params['shapeRisk'])
            elif self.age >= 360 and self.age < 372: # age 30 +
                self.b = stats.gamma.ppf(self.pRisk, params['shapeRisk'], scale=params['riskMu3']/params['shapeRisk'])

    def poisson_dist(self, rate):
        return np.random.poisson(rate)

    def uniform_dist(self):
        return np.random.uniform(0, 1)

    def immuneRK4Step(self, W, I):
        k1 = params['dt'] * (W - params['z'] * I)
        k2 = params['dt'] * (W - params['z'] * (I + 0.5 * k1))
        k3 = params['dt'] * (W - params['z'] * (I + 0.5 * k2))
        k4 = params['dt'] * (W - params['z'] * (I + k3))
        return I + 0.1666667 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)



class Model:
    '''
    This is the model class.

    The methods includes
        react()
        evolve(total time (in months))
        evovleAndSaves(total time, filename)
        save(filename) - save outputs over time
        saveOngoing(filename)
        outputs() - returns the risk of infection
        L3() 
        MDAEvent() - run MDA
        bedNetEvent()
        aWolEvent(intervention (bool))
    '''

    def __init__(self, size):
        '''
        Takes in the population size and defined parameter class.
        That is --> Model(size, Paramaters()),
        provided the 'Parameters' is the parameter class name.
        '''
        self.sU, self.sB, self.sN = 0, 0, 0

        self.n = size
        self.a = params['a']
        self.b = params['b']
        np.random.seed(int(time.time()))

        self.host_pop = [Host() for i in range(self.n)]

        self.maleWorms = []
        self.antigen = []
        self.femaleWorms = []
        self.mfCounts = []
        self.L3counts = []
        self.ages = []
        self.immunity = []

        for i in range(self.n):
            self.host_pop[i].b = self.gamma_dist(self.a, self.b) # this is unneccessary now that b is updated via update risk method. Keeping for legacy.
            self.host_pop[i].age = self.expTrunc(params['tau'], 100) * 12  # age_dist(generator) * 365.0
            self.setUB(self.sU, self.sB, self.sN)
            self.host_pop[i].pRisk = self.sB
            self.host_pop[i].updateRisk(True)  # update risk with initialisation True
            self.host_pop[i].uComp = self.sU
            self.host_pop[i].uCompN = self.sN
            self.host_pop[i].bedNet = 0
            #print(self.sU, self.sB, self.sN)

    def react(self):
        for i in range(self.n):
            self.host_pop[i].react()

    def evolve(self, tot_t):
        for i in range(self.n):
            self.host_pop[i].evolve(tot_t * 12)


    def evolveAndSaves(self, tot_t, filename):
        self.t = 0
        self.icount = 0
        self.maxMDAt = 1200
        self.maxoldMDAt = 0 # used in triple drug treatment
        self.aWolInt = False    # undergoing aWol intervention
        self.bedNetInt = 0  # undergoing bednet intervention

        if params['aWol'] == 0:
            self.maxMDAt = 1200 + params['nMDA'] * params['mdaFreq']
            if params['IDAControl'] == 1: # if switching to IDA after five treatment rounds.
                self.maxoldMDAt = 1200 + (5 * params['mdaFreq'])
            else:
                self.madoldMDAt = 2 * self.maxMDAt
        else:
            self.maxMDAt = 1200 + 36  # if aWol intervention then effects only the last two years

        #print("mosquito species: ", params['mosquitoSpecies'], "\n")
        # current L3 = 5.0
        params['L3'] = 5.0



        # open file and save a blank line to make sure file is empty before appending to it
        mf = open(filename, "w")
        mf.write(" ")
        mf.close()
        print("0----------100\n-", end = '')

        while self.t < tot_t * 12:  # for 100 years update annually, then update monthly when recording and intervention is occurring.
            if self.t < 960:  # 1200
                params['dt'] = 12
            else:
                params['dt'] = 1

            for i in range(self.n):
                self.host_pop[i].react()

            # update time t
            self.t = self.host_pop[0].t

            params['L3'] = self.L3()

            if int(self.t) % 2 == 0 and self.t < floor(self.t) + params['dt']:
                #print("t = ", self.t)
                self.saveOngoing()

            if int(self.t) % int(tot_t * 12 / 10) == 0 and self.t < floor(self.t) + params['dt']:  # every 10% of time run.
                print("-", end = '')

            if self.t >= 1200 and self.t < 1200 + params['dt']:  # events that occur at start of treatment after 100 years.
                print("bednet event at ", self.t)
                self.bedNetEvent()
                self.bedNetInt = 1

            if int(self.t) % params['mdaFreq'] == 0 and self.t < floor(self.t) + params['dt']:  # things that need to occur annually
                #if self.t > self.maxoldMDAt:
                    #params['mfPropMDA'] = 0
                    #params['wPropMDA'] = 0
                if self.t > 1200 and self.t <= self.maxMDAt:  # if after one hundred years and less than 125 years.
                    if params['aWol'] == 0:
                        self.MDAEvent()
                    else:
                        self.aWolEvent(True)
                        self.aWolInt = True  # switch so know that aWol intervention is occurring and to not update hosts that may not have been treated initially.
                    self.setBR(True)  # intervention true.
                    self.setVH(True)  #
                    self.setMu(True)
                else:
                    self.setBR(False)  # intervention false.
                    self.setVH(False)
                    self.setMu(False)
                    if params['aWol'] == 1:
                        self.aWolEvent(False)
                        self.aWolInt = False

                for i in range(self.n):
                    self.host_pop[i].updateRisk()  # update risk annually as don't need to check this as frequently as reactions.
            self.icount += 1
        print("\n")

    def save(self, filename):
        with open(filename, 'w') as mf:
            for i in range(self.n):
                mf.write(f"{self.host_pop[i].W} {self.host_pop[i].WM} {self.host_pop[i].WF} {self.host_pop[i].M}\n")

    def saveOngoing(self):
        self.maleWorms.append([self.host_pop[i].WM for i in range(self.n)])             # male worms
        self.femaleWorms.append([self.host_pop[i].WF for i in range(self.n)])           # female worms
        self.mfCounts.append([self.host_pop[i].M for i in range(self.n)])               # mf counts
        self.L3counts.append([params['L3'] for i in range(self.n)])                     # L3 larvae density
        self.ages.append([self.host_pop[i].age for i in range(self.n)])                 # ages
        self.immunity.append([self.host_pop[i].I for i in range(self.n)])               # inmmunity
        self.antigen.append([(self.host_pop[i].WM + self.host_pop[i].WF) for i in range(self.n)])       # antigen

    def mfPrevalence(self):
        self.mfCounts = np.array(self.mfCounts)
        mfPrev = [(self.mfCounts > 1)[i].sum()/self.n * 100 for i in range(self.mfCounts.shape[0])]

        return mfPrev[200:]

    def antigenPrevalence(self):
        self.antigen = np.array(self.antigen)
        antigenPrev = [(self.antigen > 0)[i].sum()/self.n * 100 for i in range(self.antigen.shape[0])]

        return antigenPrev[200:]

    def antigenPrevalence(self):
        self.ages = np.array(self.ages)
        age_range = [0, 8, 14, 19, 23, 29, 34, 45, 53, 70, 100]

        for i in range(10):
            n = self.ages.shape[0]
            ageGroup = self.ages[self.ages < age_range[i+1]]
            ageGroup = ageGroup[ageGroup > age_range[i]]
        agePrev = [(self.antigen > 0)[i].sum()/self.n * 100 for i in range(n)]

        return agePrev[200:]


    def displayOutput(self):
        plt.plot(np.arange(len(self.mfPrevalence())), self.mfPrevalence(), color = 'darkgray')
        plt.show()

    def outputs(self):
        for i in range(self.n):
            print(self.host_pop[i].b, end=" ")
            print("/n")

    def L3(self):
        mf = 0
        bTot = 0
        for i in range(self.n):
            # mf += self.param.kappas1 * math.pow(1 - math.exp(-self.param.r1 * (self.host_pop[i].mfConc() * self.host_pop[i].b) / self.param.kappas1), 2.0)
            mf += self.host_pop[i].b * self.L3Uptake(self.host_pop[i].M)
            bTot += self.host_pop[i].b

        mf = mf / bTot  # (double) n;
        return mf * (1 + self.bedNetInt * params['covN'] * (params['sN'] - 1)) * params['lbda'] * params['g'] / (params['sig'] + params['lbda'] * params['psi1'])

    def MDAEvent(self):
        for i in range(self.n):
            if self.normal_dist(self.host_pop[i].uComp, 1) < 0:
                self.host_pop[i].M = params['mfPropMDA'] * self.host_pop[i].M
                self.host_pop[i].WM = floor(params['wPropMDA'] * self.host_pop[i].WM)
                self.host_pop[i].WF = floor(params['wPropMDA'] * self.host_pop[i].WF)

    def bedNetEvent(self):
        params['sig'] = params['sig'] + params['lbda'] * params['dN'] * params['covN']
        for i in range(self.n):
            if self.normal_dist(self.host_pop[i].uCompN, 1) < 0:
                self.host_pop[i].bedNet = 1  # using bed-net
            else:
                self.host_pop[i].bedNet = 0  # not using bed-net

    def aWolEvent(self, intervention):
        for i in range(self.n):
            if intervention and not self.aWolInt:
                if self.normal_dist(self.host_pop[i].uComp, 1) < 0:
                    self.host_pop[i].M = 0
                    self.host_pop[i].WM = int(0.1 * self.host_pop[i].WM)
                    self.host_pop[i].WF = int(0.1 * self.host_pop[i].WF)
                    self.host_pop[i].aWol = 1
            elif not intervention:
                self.host_pop[i].aWol = 0

    def setBR(self, intervention):
        if intervention:
            params['lbda'] = params['lbdaR'] * params['lbda_original']
            params['xi'] = params['lbda'] * params['v_to_h'] * params['psi1'] * params['psi2'] * params['s2']
        else:
            params['lbda'] = params['lbda_original']
            params['xi'] = params['lbda'] * params['v_to_h'] * params['psi1'] * params['psi2'] * params['s2']

    def setVH(self, intervention):
        if intervention:
            params['v_to_h'] = params['v_to_hR'] * params['v_to_h_original']
            params['xi'] = params['lbda'] * params['v_to_h'] * params['psi1'] * params['psi2'] * params['s2']
        else:
            params['v_to_h'] = params['v_to_h_original']
            params['xi'] = params['lbda'] * params['v_to_h'] * params['psi1'] * params['psi2'] * params['s2']

    def setMu(self, intervention):
        if intervention:
            params['sig'] = params['sigR']
        else:
            params['sig'] = params['sig_original']

    def L3Uptake(self, mf):
        if params['mosquitoSpecies'] == 0:
            return params['kappas1'] * (1 - exp(-params['r1'] * mf / params['kappas1']))**2
        else:
            return params['kappas1'] * (1 - exp(-params['r1'] * mf / params['kappas1']))

    def gamma_dist(self, a, b):
        return np.random.gamma(a, b)

    def uniform_dist(self):
        return np.random.uniform(0, 1)

    def expTrunc(self, lambd, trunc):
        return (-1 / lambd) * log(1 - self.uniform_dist() * (1 - exp(-lambd * trunc)))

    def normal_dist(self, mu, sigma):
        return mu + np.random.normal(0, sigma)

    def setUB(self, u, b, n):
        if params['sigComp'] == 0:
            self.sU = params['u0Comp']
            self.sB = self.uniform_dist()
            self.sN = self.normal_dist(params['u0CompN'], params['sigCompN'])
        else:
            v = np.array([0, params['u0Comp'], params['u0CompN']])
            m =  np.array([[1, params['sigComp'] * params['rhoBU'], params['sigComp'] * params['rhoCN'] * params['sigCompN']],
                            [params['sigComp'] * params['rhoBU'], params['sigComp'] * params['sigComp'], 0],
                            [params['sigComp'] * params['rhoCN'] * params['sigCompN'], 0, params['sigCompN'] * params['sigCompN']]])

            result = np.random.multivariate_normal(v, m)

            b0 = result[0]
            self.sU = result[1]
            self.sN = result[2]
            self.sB = stats.norm.cdf(b0, 0, 1)

        return self.sU, self.sN, self.sB

    def reset_parameters(self):
        parameters = {
            'riskMu1': 1, # mean of risk for age group less than 16
            'riskMu2': 1, # mean of risk for age group less than 17-29
            'riskMu3': 1, # mean of risk for age group less than 30+
            'shapeRisk': 0.065, # shape parameter for bite-risk distribution (0.1/0.065)
            'mu': 0.0104, # death rate of worms
            'theta': 0, # 0.001 # immune system response parameter. 0.112
            'gamma': 0.1, # mf death rate
            'alpha': 0.58, # mf birth rate per fertile worm per 20 uL of blood.
            'lbda': 10, # number of bites per month.
            'v_to_h': 10, # vector to host ratio (39.,60.,120.)
            'kappas1': 4.395, # vector uptake and development anophelene
            'r1': 0.055, # vector uptake and development anophelene
            'tau': 0.00167, # death rate of population
            'z': 0, # waning inmmunity
            'nu': 0, # poly-monogamy parameter
            'L3': 0, # larvae density.
            'g': 0.37, # Proportion of mosquitoes which pick up infection when biting an infected host
            'sig': 5, # death rate of mosquitos
            'psi1': 0.414, # Proportion of L3 leaving mosquito per bite
            'psi2': 0.32, # Proportion of L3 leaving mosquito that enter host
            'dt': 1, # time spacing (months)
            'lbdaR': 1, # use of bed-net leading to reduction in bite rate
            'v_to_hR': 1, # use of residual-spraying leading to reduction in v_to_h
            'nMDA': 7, # number of rounds of MDA
            'mdaFreq': 12, # frequency of MDA (months)
            'covMDA': 0.65, # coverage of MDA
            's2': 0.00275, # probability of L3 developing into adult worm.
            'mfPropMDA': 0.01, # proportion of mf removed for a single MDA round.
            'wPropMDA': 0.35, # proportion of worms permanently sterilised for a single MDA round. (0.55)
            'sysComp': 0.999, # proportion of systematic non-compliance 0- none 1- all.
            'mosquitoSpecies': 0,   # 0 - Anopheles facilitation squared, 1 - Culex limitation linear.
            'rhoBU': 1, # correlation between bite risk and systematic non-compliance.
            'aWol': 0, # using doxycycline in intervention 0- not used, 1- is used.
            'sigR': 5.0, # new mortality rate of mosquitoes during vector intervention.
            'covN': 0.47, # coverage of bed nets.
            'sysCompN': 0.99, # systematic non-compliance of bed nets. set to near one.
            'rhoCN': 0, # correlation between receiving chemotherapy and use of bed nets.
            'IDAControl': 0 # if 1 then programme switches to IDA after five rounds of standard MDA defined with chi and tau.
        }

        # calculate other parameters for params
        parameters['lbda_original'] = parameters['lbda']
        parameters['v_to_h_original'] = parameters['v_to_h']
        parameters['sig_original'] = parameters['sig']
        parameters['xi'] = parameters['lbda'] * parameters['v_to_h'] * parameters['psi1'] * parameters['psi2'] * parameters['s2'] # constant bite rate
        parameters['a'] = parameters['shapeRisk'] # shape parameter (can vary)
        parameters['b'] = 1 / parameters['a'] # scale parameter determined so mean is 1

        # bed net parameters
        parameters['sN'] = 0.03 # probability of mosquito successfully biting given use of bed nets.
        parameters['dN'] = 0.41 # prob of death due to bed net.

        # non-compliance parameters for MDA
        parameters['sigComp'] = sqrt(parameters['sysComp'] / (1 + parameters['sysComp']))
        parameters['u0Comp'] = -1 * stats.norm.ppf(parameters['covMDA'], 1.0) * sqrt(1 + parameters['sigComp'] * parameters['sigComp'])

        # non-complaince parameters for bed-nets
        parameters['sigCompN'] = sqrt(parameters['sysCompN'] / (1 + parameters['sysCompN']))
        parameters['u0CompN'] = -1 * stats.norm.ppf(parameters['covN'], 1.0) * sqrt(1 + parameters['sigCompN'] * parameters['sigCompN'])

        for par in parameters.keys():
            params[par] = parameters[par]

params = {
    'riskMu1': 1, # mean of risk for age group less than 16
    'riskMu2': 1, # mean of risk for age group less than 17-29
    'riskMu3': 1, # mean of risk for age group less than 30+
    'shapeRisk': 0.16, # shape parameter for bite-risk distribution (0.1/0.065)
    'mu': 0.0104, # death rate of worms
    'theta': 0, # 0.001 # immune system response parameter. 0.112
    'gamma': 0.1, # mf death rate
    'alpha': 0.58, # mf birth rate per fertile worm per 20 uL of blood.
    'lbda': 10, # number of bites per month.
    'v_to_h': 80, # vector to host ratio (39.,60.,120.)
    'kappas1': 4.395, # vector uptake and development anophelene
    'r1': 0.055, # vector uptake and development anophelene
    'tau': 0.00167, # death rate of population
    'z': 0, # waning inmmunity
    'nu': 0, # poly-monogamy parameter
    'L3': 0, # larvae density.
    'g': 0.37, # Proportion of mosquitoes which pick up infection when biting an infected host
    'sig': 5, # death rate of mosquitos
    'psi1': 0.414, # Proportion of L3 leaving mosquito per bite
    'psi2': 0.32, # Proportion of L3 leaving mosquito that enter host
    'dt': 1, # time spacing (months)
    'lbdaR': 1, # use of bed-net leading to reduction in bite rate
    'v_to_hR': 1, # use of residual-spraying leading to reduction in v_to_h
    'nMDA': 7, # number of rounds of MDA
    'mdaFreq': 12, # frequency of MDA (months)
    'covMDA': 0.65, # coverage of MDA
    's2': 0.00275, # probability of L3 developing into adult worm.
    'mfPropMDA': 0.01, # proportion of mf removed for a single MDA round.
    'wPropMDA': 0.35, # proportion of worms permanently sterilised for a single MDA round. (0.55)
    'sysComp': 0.999, # proportion of systematic non-compliance 0- none 1- all.
    'mosquitoSpecies': 0,   # 0 - Anopheles facilitation squared, 1 - Culex limitation linear.
    'rhoBU': 0, # correlation between bite risk and systematic non-compliance.
    'aWol': 0, # using doxycycline in intervention 0- not used, 1- is used.
    'sigR': 5.0, # new mortality rate of mosquitoes during vector intervention.
    'covN': 0.47, # coverage of bed nets.
    'sysCompN': 0.99, # systematic non-compliance of bed nets. set to near one.
    'rhoCN': 0, # correlation between receiving chemotherapy and use of bed nets.
    'IDAControl': 0 # if 1 then programme switches to IDA after five rounds of standard MDA defined with chi and tau.
}

# calculate other parameters for params
params['lbda_original'] = params['lbda']
params['v_to_h_original'] = params['v_to_h']
params['sig_original'] = params['sig']
params['xi'] = params['lbda'] * params['v_to_h'] * params['psi1'] * params['psi2'] * params['s2'] # constant bite rate
params['a'] = params['shapeRisk'] # shape parameter (can vary)
params['b'] = 1 / params['a'] # scale parameter determined so mean is 1

# bed net parameters
params['sN'] = 0.03 # probability of mosquito successfully biting given use of bed nets.
params['dN'] = 0.41 # prob of death due to bed net.

# non-compliance parameters for MDA
params['sigComp'] = sqrt(params['sysComp'] / (1 + params['sysComp']))
params['u0Comp'] = -1 * stats.norm.ppf(params['covMDA'], 1.0) * sqrt(1 + params['sigComp'] * params['sigComp'])

# non-complaince parameters for bed-nets
params['sigCompN'] = sqrt(params['sysCompN'] / (1 + params['sysCompN']))
params['u0CompN'] = -1 * stats.norm.ppf(params['covN'], 1.0) * sqrt(1 + params['sigCompN'] * params['sigCompN'])
