import csv 
import numpy as np 
import matplotlib.pyplot as plt 


pathdqn = './logs-DQN-seed666/monitor.csv'

#path = './logs-MQN/monitor.csv'
path = './logs-MQN-max/monitor.csv'
path2 = './logs-MQN2-min/monitor.csv'
path3 = './logs-MQN3-min/monitor.csv'
path4 = './logs-MQN4-min/monitor.csv'
pathr = '../random_agent.csv'


def read_data(path):
    data = []
    with open(path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar=',')
        t = 0
        for row in spamreader:
            if t < 2:
                t += 1
            else:
                t += 1
                data.append(np.array([np.float32(k) for k in row[0].split(',')]))
    #print(data[:100])
    return np.array(data)
'''
databaseline = []
with open(pathbaseline) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar=',')
    t = 0
    for row in spamreader:
        if t < 2:
            t += 1
        else:
            databaseline.append(np.array([np.float32(k) for k in row[0].split(',')]))
print(databaseline[:100])

data = np.array(data)
databaseline = np.array(databaseline)

idn = range(0,databaseline.shape[0],100)
plt.plot(databaseline[:,1].cumsum()[idn], databaseline[idn,0], label='PPO')

plt.legend()
plt.xlim(0, 8e6)

plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.xlabel('timesteps')
plt.ylabel('final rewards of per episode')
plt.title('LunarLander-v2: reward of modified PPO agent')
plt.savefig('ppo2_curve_tmp.png')

'''

def sample(data, label, interval=1):
    idn = range(0, data.shape[0], interval)
    plt.plot(data[:,1].cumsum()[idn], data[idn,0], label=label)


def fill(data, label, interval=200, xscale=1):
    idn = range(interval-1, data.shape[0], interval)
    #idn = [k-1 for k in idn]
    print(len(idn))
    print(data.shape)
    data = data[:data.shape[0]//interval * interval, :]
    print(data.shape)
    timesteps = data[:, 1].cumsum()[idn]
    data = data.reshape((data.shape[0]//interval, interval, 3))
    datamean = data[:,:,0].mean(axis=1)
    datastd = data[:,:,0].std(axis=1)
    upper = datamean+datastd/2
    lower = datamean-datastd/2
    plt.fill_between(timesteps*xscale, upper, lower, alpha=0.3, label=label)
    plt.plot(timesteps*xscale, datamean)

plt.figure()
#fill(data=databaseline, label='PPO')
#fill(data=data, label='modified PPO')
data = read_data(pathdqn)
fill(data=data, label='DQN')
data = read_data(path)
fill(data=data, label='Aggressive MQN') # normal ensemble with max _predict, mqn
data = read_data(path2)
fill(data=data, label='Conservative MQN') # normal ensemble with min _predict, mqn2
data = read_data(path3)
fill(data=data, label='Cyclic MQN-8') # cyclic ensemble with min, m=8, mqn3
data = read_data(path4)
fill(data=data, label='Cyclic MNQ-2') # cyclic ensemble with min, m=2, mqn4
data = read_data(pathr)
fill(data=data, label='random agent', xscale=1) # cyclic ensemble with min, m=2

plt.legend(loc='lower right')

#plt.plot(np.ones(250)*50000, range(-250, 0))

plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.xlabel('timesteps')
plt.ylabel('final rewards of per episode')
plt.xlim(0, 6e6)
plt.title('LunarLander-v2: reward of modified DQN agent')
plt.savefig('QN_curve_bar.png')

quit()

reward = []
eplen = []
with open(path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pass
        #print(row['r'], row['l'])
