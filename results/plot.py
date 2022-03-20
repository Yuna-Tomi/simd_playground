#!/usr/bin/env python
import matplotlib.pyplot as plt

def plt_setup():
    plt.rcParams["font.size"] = 15
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["xtick.major.size"] = 5.0
    plt.rcParams["ytick.major.size"] = 5.0
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["legend.edgecolor"] = "black"

X = range(3, 20, 2)
NAIVE2 = "naive2"
SIMD1 = "simd1"
SIMD2 = "simd2"
SIMD3 = "simd3"
naive2 = []
simd1 = []
simd2 = []
simd3 = []
with open("raw", "r") as f:
    for line in f.readlines():
        tokens = line.split(' ')
        tokens = [s for s in tokens if len(s) > 0]
        try:
            i = tokens.index("bench:")
        except:
            continue

        test = tokens[1]
        ms = round(int(tokens[i+1].replace(',', '')), -3)/1000
        if "naive2" in test: 
            naive2.append(ms)
        elif "simd1" in test:
            simd1.append(ms)
        elif "simd2" in test:
            simd2.append(ms)
        elif "simd3" in test:
            simd3.append(ms)
        else:
            pass

naive2.sort()
simd1.sort()
simd2.sort()
simd3.sort()

plt_setup()
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.set_xlim(2, 20)
ax.set_ylim(0, 200000)
ax.set_xlabel("Kernel Size", size=30)
ax.set_ylabel("Average Execution Time (ms)", size=30)

ax.plot(X, naive2, label=NAIVE2)
ax.plot(X, simd1, label=SIMD1)
ax.plot(X, simd2, label=SIMD2)
ax.plot(X, simd3, label=SIMD3)
ax.legend()

fig.savefig("bench.jpeg")