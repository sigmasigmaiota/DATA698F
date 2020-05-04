# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:38:34 2020

@author: steph
"""

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
import numpy as np
import pandas as pd

# For a jupyter notebook add the following line:
#%matplotlib inline

# The below is needed for both notebooks and scripts
import matplotlib.pyplot as plt

######################################################################################################
######################################################################################################

#                         Generate DataSet 

######################################################################################################
######################################################################################################

#nsfg = pd.read_csv('C:/Users/steph/Documents/DATA698/Data/NSFG.csv')
#for i, row in nsfg.iterrows():
#    prob = row['pHIVdiagPCT']/100
#    nsfg.loc[i,'HIV'] = np.random.choice(a=[0, 1],
#        size=1,
#        p=[1-prob,prob])
#    print('Prob: ',prob,' HIV: ',nsfg.loc[i,'HIV'])
#for i, row in nsfg.iterrows():
#    prob = row['NEagesexPrEPpct']/100
#    if (row['HIV'] == 0) & (row['ACTIVE'] == 1):
#        nsfg.loc[i,'PrEP'] = np.random.choice(a=[0, 1],
#                size=1,
#                p=[1-prob,prob])
#        print('Prob: ',prob,' PrEP: ',nsfg.loc[i,'PrEP'])
#        
#nsfg.to_csv('C:/Users/steph/Documents/DATA698/Data/Baseline2/NE_ABM_Baseline2.csv',index=False)

######################################################################################################
######################################################################################################

nsfg = pd.read_csv('C:/Users/steph/Documents/DATA698/Data/Baseline2/NE_ABM_Baseline2.csv')

# The data collector stores three categories of data: 
# model-level variables, agent-level variables, and tables 
# (which are a catch-all for everything else). Model- and 
# agent-level variables are added to the data collector along with 
# a function for collecting them. Model-level collection functions 
# take a model object as an input, while agent-level collection 
# functions take an agent object as an input. Both then return a 
# value computed from the model or each agent at their current state. 
# When the data collector’s collect method is called, with a model 
# object as its argument, it applies each model-level collection function 
# to the model, and stores the results in a dictionary, associating the 
# current value with the current step of the model. Similarly, the method 
# applies each agent-level collection function to each agent currently 
#in the schedule, associating the resulting value with the step of the 
# model, and the agent’s unique_id.

def compute_HIVpct(model):
    agent_HIVstats = [agent.HIVstat for agent in model.schedule.agents]
    x = sorted(agent_HIVstats)
    N = model.num_agents
    B = sum( xi for i,xi in enumerate(x) ) / (N)
    return (B)

def compute_numHIV(model):
    agent_HIVstats = [agent.HIVstat for agent in model.schedule.agents]
    x = sorted(agent_HIVstats)
    B = sum( xi for i,xi in enumerate(x) )
    #return (B/N)
    return (B)

def compute_PrEPpct(model):
    agent_PrEPstats = [agent.prep for agent in model.schedule.agents]
    x = sorted(agent_PrEPstats)
    N = model.num_agents
    B = sum( xi for i,xi in enumerate(x) ) / (N)
    return (B)


class PrEPAgent(Agent):
    """ An agent with fixed initial HIVstat."""
    def __init__(self, unique_id, model):# ppct, model):################################################## ppct added here
        super().__init__(unique_id, model)

        randomagent = nsfg.sample(n=1,replace=False) 
        self.HIVstat = randomagent.HIV.sum()

        self.sex = randomagent.MALE.sum()
        # orientation; should be gay, straight, or bisexual, here as dichotomous
        self.orient = randomagent.SAMESEXANY.sum()
        # taking PrEP and adherent at least 80% of the time
        self.prep = randomagent.PrEP.sum()
        self.agecat = randomagent.AGECAT.sum()
        self.raceeth = randomagent.RACEETH.sum()
        self.active = randomagent.ACTIVE.sum()
        self.condom = randomagent.CONDOM.sum()
        self.prep_prob = randomagent.NEagesexPrEPpct.sum()/100

# But there’s an even simpler way, using the grid’s built-in get_neighborhood method, 
# which returns all the neighbors of a given cell. This method can get two types of 
# cell neighborhoods: Moore (including diagonals), and Von Neumann (only up/down/left/right). 
# It also needs an argument as to whether to include the center cell itself as one of the neighbors.

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        # random order to the order of the transactions
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

# Now we just need to have the agents do what we intend for them to do: check their HIVstat, 
# and if they have the virus, give one unit of it away to another random agent. 
# To allow the agent to choose another agent at random, we use the model.random 
# random-number generator. This works just like Python’s random module, but with 
# a fixed seed set when the model is instantiated, that can be used to replicate a specific model run later.

# To pick an agent at random, we need a list of all agents. Notice that there isn’t 
# such a list explicitly in the model. The scheduler, however, does have an internal 
# list of all the agents it is scheduled to activate.        
        
    def risk_hiv(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            othpprob = other.prep_prob*2
            slfpprob = self.prep_prob*2
            if other.HIVstat == 0 and other.prep == 0:
                other.prep = np.random.choice(a=[0, 1], size=1, p=[1-othpprob,othpprob]).sum()
            if self.HIVstat==0 and self.prep == 0:
                self.prep = np.random.choice(a=[0, 1], size=1, p=[1-slfpprob,slfpprob]).sum()
            ocp = other.condom
            scp = self.condom
            othcond = np.random.choice(a=[0, 1], size=1, p=[1-ocp,ocp]).sum()
            slfcond = np.random.choice(a=[0, 1], size=1, p=[1-scp,scp]).sum()
            # scaling factor
            HETsex = 0
            if self.sex != other.sex:
                HETsex = np.random.choice(a=[0, 1], size=1, p=[0.875,0.125]).sum()
            # same-sex HIV positive male agent insertive to negative male partner
            if self.sex == other.sex and self.orient == 1 and self.HIVstat == 1 and other.HIVstat == 0 and other.prep == 0 and self.sex == 1 and slfcond == 0 and othcond == 0:
                other.HIVstat = 1
                print('Agent male to other male: HIV transmitted.')
                #self.HIVstat -= 1
            # same-sex HIV negative male receptive agent with HIV positive male partner    
            if self.sex == other.sex and self.orient == 1 and self.HIVstat == 0 and other.HIVstat == 1 and self.prep == 0  and self.sex == 1  and slfcond == 0 and othcond == 0:
                self.HIVstat = 1
                print('Other male to male agent: HIV transmitted.')
            if HETsex == 1:
                if self.sex != other.sex and self.orient == 0 and self.HIVstat == 0 and other.HIVstat == 1 and self.prep == 0 and self.sex == 0 and othcond == 0:
                    self.HIVstat = 1
                    print('Other male to female agent: HIV transmitted.')
                # straight, male vaginal other insertive
                if self.sex != other.sex and self.orient == 0 and self.HIVstat == 1 and other.HIVstat == 0 and other.prep == 0 and self.sex == 1 and slfcond == 0:
                    other.HIVstat = 1
                    print('Agent male to other female: HIV transmitted.')


# https://wrrv.com/how-many-sexual-partners-does-the-average-new-yorker-have/ average number of lifetime sexual partners
# https://onlinedoctor.superdrug.com/whats-your-number/?utm_source=affiliatewindow&utm_campaign=Skimlinks&utm_medium=affiliate&utm_term=30283X879131X823ab646c75a214e4447e00ecbf8f3d0&utm_content=0&awc=2026_1504631652_17154af64531ab1f32c80f1d919685ff
# men: 7
# women: 6.4
# https://www.cdc.gov/nchs/nsfg/key_statistics/s.htm#sexualfemales
# 2011-2015: Males 10.7% reported HIV risk-related behavior in the last 12M
# 2011-2015: Females 8.3% reported HIV risk-related behavior in the last 12M
# 2011-2015: Average for both is 9.5% in the last 12M
# If the average is used, with each encounter, 1 in 10  will result in HIV-risk behavior and transmission
                
    def step(self):
        self.move()
        #HIVrisk = np.random.choice(np.arange(0, 2), p=[1-0.095, 0.095])
        #if (HIVrisk==1) & (self.HIVstat > 0):
        if self.active == 1:
            self.risk_hiv()

            
# Let’s add a simple spatial element to our model by 
# putting our agents on a grid and make them walk around at random. 
# Instead of giving their unit of hiv to any random agent, 
# they’ll give it to an agent on the same cell.

# Mesa has two main types of grids: SingleGrid and MultiGrid. 
# SingleGrid enforces at most one agent per cell; MultiGrid 
# allows multiple agents to be in the same cell. Since we want 
# agents to be able to share a cell, we use MultiGrid.            

class PrEPModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):#, ppct): ############################### ppct added here
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True

# We instantiate a grid with width and height parameters, 
# and a boolean as to whether the grid is toroidal. 
# Let’s make width and height model parameters, in addition 
# to the number of agents, and have the grid always be toroidal. 
# We can place agents on a grid with the grid’s place_agent method, 
# which takes an agent and an (x, y) tuple of the coordinates to place the agent.        
        
        # Create agents
        for i in range(self.num_agents):
            a = PrEPAgent(i, self)# ppct, self) ##################################### ppct added here
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            

        self.datacollector = DataCollector(
            #model_reporters={"Gini": compute_gini, "HIV": compute_numHIV},
            model_reporters={"HIV": compute_numHIV, "HIVpct": compute_HIVpct, "PrEPpct": compute_PrEPpct},
            agent_reporters={"HIVstat": "HIVstat", 
                             "PrEPRx":"prep", 
                             "RaceEth":"raceeth",
                             "Male":"sex",
                             "Active":"active",
                             "AgeCat":"agecat",
                             "SexMinority":"orient",
                             "Condom":"condom"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        
########################################################################################## batchrunner
# set to emulate a constant rate of seroconversion in NYC
# resource https://www1.nyc.gov/site/doh/data/data-sets/hiv-aids-annual-surveillance-statistics.page
# 2000: 36261, 39694, 41586, 41886, 42542, 43398, 44368, 45371, 46445, 47412, 
# 2010: 48504, 49756, 51048, 52179, 53679, 55003, 56091, 56908, 
# 2018: 57559
# list above is from 2000 - 2018
        
        
# population of NYC for year 2018
# 8174988 2010
# 0.00593321971848766 proportionally in 2010
# 8398748 2018 https://www.census.gov/quickfacts/newyorkcitynewyork
# 0.006853283370330911 proportionally in 2018

# Since the above proportion of risk behavior is set for the last 12M, 
# keep the units per year. We'll set the max_steps to 52.

######################################################### different approach
# according to the 2018 surveillance report .0125% new diagnoses of HIV, delivered as 12.5 per 100,000 people
# https://www.health.ny.gov/diseases/aids/general/statistics/annual/2018/2018_annual_surveillance_report.pdf
# https://www.health.ny.gov/diseases/aids/general/statistics/

numpl = 5000
hw = 75
stps = 100

model1 = PrEPModel(numpl,hw,hw)
for i in range(stps):
    model1.step()
    
HIVprev = model1.datacollector.get_model_vars_dataframe()


############################################################ plot HIV total percentage growth


plt.figure(figsize = [10,5])
m1, b1 = np.polyfit(HIVprev.index, HIVprev['HIVpct'], 1)
yfit1 = m1*HIVprev.index + b1
diffSqSum1 = ((HIVprev['HIVpct'] - yfit1)**2).sum()
standErrEst1 = np.sqrt(diffSqSum1/len(HIVprev['HIVpct'].index))
plt.scatter(HIVprev.index, HIVprev['HIVpct'], marker = 'o', s = (1000*HIVprev['HIVpct']), alpha = .5, c = HIVprev['HIVpct'])
p1 = plt.plot(HIVprev.index, yfit1, color='black')
plt.fill_between(HIVprev.index, yfit1-standErrEst1, yfit1 + standErrEst1, color='gray', alpha=0.2)
plt.title('HIV Total Percentage\nABM with '+str(stps)+' Steps, '+str(numpl)+' Agents, & '+str(hw*hw)+' Squares')
plt.xlabel('Iterations as Months')
plt.ylabel('HIV Prevalence')
plt.savefig('C:/Users/steph/Documents/DATA698/Visualizations/NE_HIVpctTotalABM_'+str(stps)+'stps_'+str(numpl)+'ppl_'+str(hw)+'hw.png', quality = 90, format = 'png', dpi=600)
plt.show()

print('Slope is '+str(m1))
print('24M percent change is '+str(HIVprev.HIVpct[24]) +' - '+ str(HIVprev.HIVpct[0]) + 
          ' = ' + str(HIVprev.HIVpct[24]- HIVprev.HIVpct[0]))

############################################################ plot PrEP total precentage growth

plt.figure(figsize = [10,5])
m1, b1 = np.polyfit(HIVprev.index, HIVprev['PrEPpct'], 1)
yfit1 = m1*HIVprev.index + b1
diffSqSum1 = ((HIVprev['PrEPpct'] - yfit1)**2).sum()
standErrEst1 = np.sqrt(diffSqSum1/len(HIVprev['PrEPpct'].index))
plt.scatter(HIVprev.index, HIVprev['PrEPpct'], marker = 'o', s = (1000*HIVprev['PrEPpct']), alpha = .5, c = HIVprev['PrEPpct'])
p1 = plt.plot(HIVprev.index, yfit1, color='black')
plt.fill_between(HIVprev.index, yfit1-standErrEst1, yfit1 + standErrEst1, color='gray', alpha=0.2)
plt.title('PrEP Total Percentage\nABM with '+str(stps)+' Steps, '+str(numpl)+' Agents, & '+str(hw*hw)+' Squares')
plt.xlabel('Iterations as Months')
plt.ylabel('PrEP Use')
plt.savefig('C:/Users/steph/Documents/DATA698/Visualizations/NE_PrEPpctTotalABM_'+str(stps)+'stps_'+str(numpl)+'ppl_'+str(hw)+'hw.png', quality = 90, format = 'png', dpi=600)
plt.show()

print('Slope is '+str(m1))
print('24M percent change is '+str(HIVprev.PrEPpct[24]) +' - '+ str(HIVprev.PrEPpct[0]) + 
          ' = ' + str(HIVprev.PrEPpct[24]- HIVprev.PrEPpct[0]))

###############################################################  plot HIV growth as difference in pct between iterations

HIVgrowth = list(HIVprev['HIVpct'])

growth = [t - HIVgrowth[i-1] for i, t in enumerate(HIVgrowth)][1:]

plt.figure(figsize = [10,5])
plt.plot(growth)

m2, b2 = np.polyfit(np.arange(1,len(growth)+1,1), growth, 1)
yfit2 = m2*np.arange(1,len(growth)+1,1) + b2
p2 = plt.plot(np.arange(1,len(growth)+1,1), yfit2, color='black')

plt.title('HIV Prevalence Growth\nABM with '+str(stps)+' Steps, '+str(numpl)+' Agents, & '+str(hw*hw)+' Squares\nHIV Growth in Percent between Iterations')
plt.xlabel('Iterations as Months')
plt.ylabel('HIV Growth')
plt.savefig('C:/Users/steph/Documents/DATA698/Visualizations/NE_HIVGrowthABM_'+str(stps)+'stps_'+str(numpl)+'ppl_'+str(hw)+'hw.png', quality = 90, format = 'png', dpi=600)
plt.show()

print('Slope is '+str(m2))


############################################################## plot PrEP growth as difference in pct between iterations


PrEPgrowth = list(HIVprev['PrEPpct'])

growth = [t - PrEPgrowth[i-1] for i, t in enumerate(PrEPgrowth)][1:]

plt.figure(figsize = [10,5])
plt.plot(growth)

m2, b2 = np.polyfit(np.arange(1,len(growth)+1,1), growth, 1)
yfit2 = m2*np.arange(1,len(growth)+1,1) + b2
p2 = plt.plot(np.arange(1,len(growth)+1,1), yfit2, color='black')

plt.title('PrEP Use Growth\nABM with '+str(stps)+' Steps, '+str(numpl)+' Agents, & '+str(hw*hw)+' Squares\nHIV Growth in Percent between Iterations')
plt.xlabel('Iterations as Months')
plt.ylabel('PrEP Growth')
plt.savefig('C:/Users/steph/Documents/DATA698/Visualizations/NE_PrEPGrowthABM_'+str(stps)+'stps_'+str(numpl)+'ppl_'+str(hw)+'hw.png', quality = 90, format = 'png', dpi=600)
plt.show()

print('Slope is '+str(m2))

agent_hiv = model1.datacollector.get_agent_vars_dataframe()

agent_hiv.to_csv('C:/Users/steph/Documents/DATA698/Data/NE_ABM_'+str(stps)+'stps_'+str(numpl)+'ppl_'+str(hw)+'hw.csv',index=False)

################################################################## race and ethnicity of the sample

plt.figure(figsize = [10,10])
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['White', 'Black/African American', 'Hispanic/Latino', 'Other']
sizes = agent_hiv.RaceEth.value_counts()
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Demographics, Sampled from NE\nABM with '+str(stps)+' Steps, '+str(numpl)+' Agents, & '+str(hw*hw)+' Squares')
plt.savefig('C:/Users/steph/Documents/DATA698/Visualizations/NE_PieChart_'+str(stps)+'stps_'+str(numpl)+'ppl_'+str(hw)+'hw.png', quality = 90, format = 'png', dpi=600)
plt.show()









#agent_hiv.HIVstat.value_counts()
#agent_hiv.PrEPRx.value_counts()
#import pandas as pd
print(pd.crosstab(agent_hiv.Male, agent_hiv.HIVstat, rownames=['Male'], colnames=['HIVstat']))
print(pd.crosstab(nsfg.MALE, nsfg.HIV, rownames=['MALE'], colnames=['HIV']))





























#####################################################################################################################



############################################### BATCH RUNNER BELOW ##################################################



#####################################################################################################################
# Let's use this to calibrate the rate


fixed_params = {
    "width": 50,
    "height": 50
}

setrange = range(1000, 1400, 50)

# set the number of agents in the region with min, max and step size
variable_params = {"N": setrange}

# The variables parameters will be invoke along with the fixed parameters allowing for either or both to be honored.
batch_run = BatchRunner(
    PrEPModel,
    variable_params,
    fixed_params,
    # previously set to 5
    iterations=20,
    ############################################# previously set to 100, represents 52 encounters each year.
    ############################################# 9.5 percent of these encounters will be HIV-risky
    max_steps=3,
    model_reporters={"HIV": compute_numHIV, "HIVpct": compute_HIVpct, "PrEPpct": compute_PrEPpct},
    agent_reporters={"HIVstat":"HIVstat", "PrEPRx":"prep", 
                             "RaceEth":"raceeth",
                             "Male":"male",
                             "Active":"active",
                             "AgeCat":"agecat",
                             "SexMinority":"samesexany"}
)

batch_run.run_all()

########################################################################################## visualization

run_data = batch_run.get_model_vars_dataframe()
run_data.head()

x = run_data.N
y = run_data.HIVpct

################################ add error region
# Compute the confidence interval

stdN = run_data.groupby(['N']).agg(lambda x: np.std(x))['HIVpct'].reset_index()
x_df = pd.DataFrame(x)
stdN = x_df.merge(stdN, left_on='N', right_on='N')

meanN = run_data.groupby(['N']).agg(lambda x: np.mean(x))['HIVpct'].reset_index()
meanN = x_df.merge(meanN, left_on='N', right_on='N')

############################### finish error calculations

plt.figure(figsize = [10,5])

m, b = np.polyfit(x, y, 1)

yfit = m*x + b

diffSqSum = ((y - yfit)**2).sum()

standErrEst = np.sqrt(diffSqSum/len(y.index))

plt.scatter(x, y, marker = 'o', s = (10000*y), alpha = .5, c = y)
p1 = plt.plot(x, yfit, color='black')

plt.fill_between(x, yfit-standErrEst, yfit + standErrEst, color='gray', alpha=0.2)

plt.xlabel('Number of Agents')
plt.ylabel('Percent of Agents with HIV')
plt.title('Agent-Based Model, '+str(batch_run.max_steps)+' Encounters', fontsize = 16)

plt.savefig('C:/Users/steph/Documents/DATA698/ABM_'+str(batch_run.max_steps)+' Calibration.png', quality = 90, format = 'png', dpi=600)
plt.show()

run_data.to_csv('C:/Users/steph/Documents/DATA698/Data/ABM_'+str(batch_run.max_steps)+'_Calibration.csv',index=False)

    
    
    

    