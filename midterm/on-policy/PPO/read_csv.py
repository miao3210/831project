import csv 
import numpy as np 
import matplotlib.pyplot as plt 

path = './logs-cpu/monitor.csv'
data = []
with open(path) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar=',')
    t = 0
    for row in spamreader:
        if t < 2:
            t += 1
        else:
            data.append(np.array([np.float32(k) for k in row[0].split(',')]))
print(data[:100])
data = np.array(data)
idn = range(0,data.shape[0],100)
plt.plot(data[:,1].cumsum()[idn], data[idn,0])
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.xlabel('timesteps')
plt.ylabel('final rewards of per episode')
plt.title('LunarLander-v2: reward of on-policy PPO agent')
plt.savefig('ppo_curve_tmp.png')

plt.figure()
interval = 100
idn = range(interval, data.shape[0], interval)
idn = [k-1 for k in idn]
print(data.shape)
data = data[:data.shape[0]//interval * interval, :]
print(data.shape)
timesteps = data[:, 1].cumsum()[idn]
data = data.reshape((data.shape[0]//interval, interval, 3))
datamean = data[:,:,0].mean(axis=1)
datastd = data[:,:,0].std(axis=1)
upper = datamean+datastd/2
lower = datamean-datastd/2
plt.fill_between(timesteps, upper, lower, alpha=0.3)
plt.plot(timesteps, datamean)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.xlabel('timesteps')
plt.ylabel('final rewards of per episode')
plt.title('LunarLander-v2: reward of on-policy PPO agent')
plt.savefig('ppo_curve_bar.png')

quit()

reward = []
eplen = []
with open(path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pass
        #print(row['r'], row['l'])
