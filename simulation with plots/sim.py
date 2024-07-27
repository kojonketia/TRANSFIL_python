# importing model library
from Transfil import *

# importing other libraries
import sys, time
import seaborn as sns
import numpy as np, scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib.animation as animation


def main():
    # generator.seed(time(0))
    duration = 0        # model runtime
    tot_y = 0           # number of years (in months)
    n = 0               # total population
    filename = "test3.txt"
    runs = 1            # number of simulation runs

    if len(sys.argv) == 4:
        print("Four arguments imported.")
        tot_y = float(sys.argv[1])
        n = int(sys.argv[2])
        runs =  int(sys.argv[3])

    elif len(sys.argv) == 3:
        print("three arguments imported.")
        tot_y = float(sys.argv[1])
        n = int(sys.argv[2])

    else:
        tot_y = 120         # 20 years after start of intervention
        n = 1000           # default number of people

    u, b, pnet = 0, 0, 0
    u, b, pnet = Model(n).setUB(u, b, pnet)
    print("random u, b, and n:", u, b, pnet)
    print("random gamma example:", Model(n).gamma_dist(1.0, 1.0))
    print("random poisson example:", Host().poisson_dist(100.0))
    print("random gaussian example:", Model(n).normal_dist(1.0, 0.1))
    print("random gamma example inverse method:", stats.gamma.ppf(Host().uniform_dist(), 1.0, 1.0))
    start = time.time()

    print("Initialising arrays...")
    print(' ')
    print("Running...")

    mfPrevalence = []           # saves mf prevalences at each time point
    antigenPrevalence = []      # saves antigen prevalence at each time point

    # simulation begins for a given number of runs
    for i in range(runs):
        print(i+1)
        m1 = Model(n)
        m1.evolveAndSaves(tot_y, filename)
        mfPrevalence.append(m1.mfPrevalence())
        antigenPrevalence.append(m1.antigenPrevalence())
        m1.reset_parameters()

    age_list = m1.ages

    duration = (time.time() - start)        # returns simulatiton duration
    print("Finished simulation in", duration/60, "mins")

    #az.style.use("arviz-grayscale")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # plots all mf prevalences
    for i in range(runs):
        plt.plot(np.arange(len(mfPrevalence[i])), mfPrevalence[i], color = 'darkgray', lw = 0.5)

    plt.plot(np.arange(len(mfPrevalence[0])), np.median(np.array(mfPrevalence), axis = 0), color = 'blue', label = "median")
    plt.scatter(np.arange(len(mfPrevalence[0])), np.median(np.array(mfPrevalence), axis = 0), color = 'blue', s = 10)
    plt.axhline(y = 1, color = 'red', linestyle = 'dotted', xmin = 0, xmax = 120)
    plt.ylabel("mf prevalence (%)")
    plt.xticks([i for i in range(0,len(mfPrevalence[0])+1,30)], [str(int(i/6)) for i in range(0,len(mfPrevalence[0])+1,30)]);
    plt.grid(linewidth = 0.2)
    plt.xlim(right = 120)
    plt.legend()
    plt.show()

    # plot all antigen prevalences
    plt.subplot(1,2,1)
    plt.plot(np.array(antigenPrevalence).T, color = 'darkgray', lw = 0.5)
    plt.plot(np.arange(len(antigenPrevalence[0])), np.median(np.array(antigenPrevalence), axis = 0), color = 'blue', label = "median")
    plt.scatter(np.arange(len(antigenPrevalence[0])), np.median(np.array(antigenPrevalence), axis = 0), color = 'blue', s = 10)
    plt.axhline(y = 2, color = 'red', linestyle = 'dotted')
    plt.ylabel("antigen prevalence (%)")
    plt.xlabel("time starts since intervention (%)")
    plt.xticks([i for i in range(0,len(antigenPrevalence[0])+1,30)], [str(int(i/6)) for i in range(0,len(antigenPrevalence[0])+1,30)]);
    plt.grid(linewidth = 0.2)
    plt.xlim(right = 120)
    plt.legend()

    plt.style.use("ggplot")

    plt.subplot(1,2,2)
    colors = ['plum', 'g', 'orange']
    data = [33.1, 4.0, 2.2]
    antigenPrevalence = np.array(antigenPrevalence)
    sns.boxplot(antigenPrevalence[:,[0,42,72]], palette = colors)
    sns.scatterplot(data, label = "data", color = "black")
    plt.xticks([0,1,2],["2000", "2007", "2012"])
    plt.xlabel("Year")
    plt.ylabel("antigen prevalence{%}")
    plt.title("antigen prevalence validation")
    plt.show()

    print(antigenPrevalence.shape)

    #np.savetxt('antigen100.txt', np.array(antigenPrevalence))
    #print("write to file...")

if __name__ == "__main__":
    main()
