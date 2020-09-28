import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os
import seaborn 
import multiprocessing
import time

from hegemons import run



def run_batch_flow(batch_size, US_strat, Iran_strat, default_strat="low", coop_multiplier=1, 
                conflict_multiplier=1, coop_gain=0.01, war_cost=0.5, war_steal=0.5, power_scale=0.1):
    data = []
    flow = []
    for i in range(batch_size):
        res = (run(end_at_tick=1000, US_strat=US_strat, Iran_strat=Iran_strat, default_strat=default_strat, coop_multiplier=coop_multiplier, 
                conflict_multiplier=conflict_multiplier, coop_gain=coop_gain, war_cost=war_cost, war_steal=war_steal, power_scale=power_scale).get_model_run(0,1000))
        data.append(res[0])
        flow.append(res[1])

    return (data, flow)
def experiment_1_flow(args):
    batch_size = 100
    with multiprocessing.Pool(processes=4) as pool:
        result = pool.starmap(run_batch_flow, [(batch_size, "low", "low")+args, (batch_size, "low", "high")+args, (batch_size, "high", "high")+args, (batch_size, "high", "low")+args])
        result = np.asarray(result)
        means = pool.map(calculate_mean, result[:,0,:])
        
    res = np.asarray(means)
    #return res

    lowlow_flows_avg = sum(np.asarray(result[0,1,:])) / batch_size
    US_gains = lowlow_flows_avg[0,:]
    US_losses = lowlow_flows_avg[1,:]
    width =0.3
    fig,ax = plt.subplots()
    ax.bar(np.arange(len(US_gains)), US_gains, width=width)
    ax.bar(np.arange(len(US_losses))+ width, US_losses, width=width)
    ax.set_xticks([0,1,2,3,4,5,6,7,8])
    ax.set_xticklabels(["US", "Iran", "UK", "France", "Germany", "China", "Russia", "Israel", "Saudi"])
    ax.legend(["Gains","Losses"])
    plt.show()
    Iran_gains = lowlow_flows_avg[2,:]
    Iran_losses = lowlow_flows_avg[3,:]
    width =0.3
    fig,ax = plt.subplots()
    ax.bar(np.arange(len(Iran_gains)), Iran_gains, width=width)
    ax.bar(np.arange(len(Iran_losses))+ width, Iran_losses, width=width)
    ax.set_xticks([0,1,2,3,4,5,6,7,8])
    ax.set_xticklabels(["US", "Iran", "UK", "France", "Germany", "China", "Russia", "Israel", "Saudi"])
    ax.legend(["Gains","Losses"])
    plt.show()
    res = np.asarray(means)
    
    width = 0.3
    plt.bar(np.arange(4), res[:,0], width=width)
    plt.bar(np.arange(4) + width, res[:,1], width=width)
    plt.show()
    return res

def run_batch(batch_size, US_strat, Iran_strat, default_strat="low", coop_multiplier=1, 
                conflict_multiplier=1, coop_gain=0.01, war_cost=0.5, war_steal=0.5, power_scale=0.1):
    data = []
    
    for i in range(batch_size):
        data.append(run(end_at_tick=1000, US_strat=US_strat, Iran_strat=Iran_strat, default_strat=default_strat, coop_multiplier=coop_multiplier, 
                conflict_multiplier=conflict_multiplier, coop_gain=coop_gain, war_cost=war_cost, war_steal=war_steal, power_scale=power_scale).get_model_run(0,1000)[0])

    return data

def calculate_mean(data):
    U_sum = 0
    I_sum = 0
    for i in range(len(data)):
        U_sum = U_sum + data[i][0,-1]
        I_sum = I_sum + data[i][1,-1]
    return U_sum/len(data), I_sum/len(data)

def experiment_1(args):
    batch_size = 100
    with multiprocessing.Pool(processes=4) as pool:
        result = pool.starmap(run_batch, [(batch_size, "low", "low")+args, (batch_size, "low", "high")+args, (batch_size, "high", "high")+args, (batch_size, "high", "low")+args])
        means = pool.map(calculate_mean, result)
    print(np.shape(result))
    res = np.asarray(means)
    #return res
    
    print(means)
    res = np.asarray(means)
    """
    print(res)
    width =0.3
    plt.bar(np.arange(4), res[:,0], width=width)
    plt.bar(np.arange(4) + width, res[:,1], width=width)
    plt.show()"""
    return res

def experiment_1_sensitivity():
    args = ["low", 1, 1, 0.01, 0.5, 0.5, 0.1]
    values = np.arange(0.001,1,0.05)
    data = []
    for i in values:
        args[5] = i
        tuple_args = tuple(args)
        data.append(experiment_1(tuple_args))
    US_1 = []
    US_2 = []
    US_3 = []
    Iran_1 = []
    Iran_2 = []
    Iran_3 = []
    for d in data:
        US_1.append(d[1,0]/d[0,0])
        US_2.append(d[2,0]/d[0,0])
        US_3.append(d[3,0]/d[0,0])
        Iran_1.append(d[1,1]/d[0,1])
        Iran_2.append(d[2,1]/d[0,1])
        Iran_3.append(d[3,1]/d[0,1])
    plt.plot(US_1)
    plt.plot(US_2)
    plt.plot(US_3)
    plt.xlabel("Power_Scale")
    plt.ylabel("Relative wealth over Baseline Strategy")
    plt.show()
    plt.plot(Iran_1)
    plt.plot(Iran_2)
    plt.plot(Iran_3)
    plt.xlabel("Power_Scale")
    plt.ylabel("Relative wealth over Baseline Strategy")
    plt.show()
    print(US_1)
    print(US_2)
    print(US_3)
 

def get_model_run(path):
    return np.loadtxt(path, delimiter=',')

if __name__ == "__main__":
    experiment_1_flow(("low", 1, 1, 0.01, 0.5, 0.5, 0.1))
    #experiment_1_sensitivity()