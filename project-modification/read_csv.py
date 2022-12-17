import csv 
import numpy as np 
import matplotlib.pyplot as plt 

#path = './logs/monitor.csv'
pathbaseline = '/home/miao/16831/midterm-onpolicy/PPO/logs-basePPO/monitor.csv'
pathbaseline2 = '/home/miao/16831/midterm-onpolicy/PPO/logs-PPO-again/monitor.csv'

pathppo2final = './logs-PPO2-seed234/monitor.csv'
pathxi4 = './logs-PPO2-xi4.0/monitor.csv'
path = './logs/monitor.csv'

path3 = './logs-PPO3/monitor.csv'

pathr = '../random_agent.csv'

pathppo2 = [
    './logs',
    #'./logs-PPO2-record',
    #'./logs-ppo2-reproduce',=
    './logs-PPO2-seed234',
    #'./logs-PPO2-seed5',
    #'./logs-PPO2-xi5.0-seed5',
    #'./logs-PPO2-xi10.0-seed5', #999',
    ##'./logs-PPO2-xi4.0',
    #'./logs-PPO3',
    #'./logs-PPO2-record',
    #'./logs-ppo2-reproduce',
]
pathppo2csv = [p + '/monitor.csv' for p in pathppo2]

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


def fill(data, label, interval=100):
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
    plt.fill_between(timesteps, upper, lower, alpha=0.3, label=label.replace('xi', '$\psi$'))
    plt.plot(timesteps, datamean)

plt.figure()
#for m in pathppo2csv:
#    print(m)
#    data = read_data(m)
#    fill(data=data, label=m.split('/monitor')[0])
#fill(data=databaseline, label='PPO')
#fill(data=data, label='modified PPO')

#data = read_data(path)
#fill(data=data, label='modified PPO')
data = read_data(path3)
fill(data=data, label='modified PPO xi=-1')

data = read_data(pathppo2final)
fill(data=data, label='modified PPO xi=1')

data = read_data(pathxi4)
fill(data=data, label='modified PPO xi=4')

data = read_data(pathbaseline)
fill(data=data, label='PPO')
#data = read_data(pathbaseline2)
#fill(data=data, label='PPO again')

data = read_data(pathr)
fill(data=data, label='random agent') # cyclic ensemble with min, m=2

plt.legend(loc='lower right')


plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.xlabel('timesteps')
plt.ylabel('final rewards of per episode')
plt.xlim(0, 2e6)
plt.title('LunarLander-v2: reward of modified PPO agent')
plt.savefig('ppo2_curve_bar.png')

quit()

reward = []
eplen = []
with open(path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pass
        #print(row['r'], row['l'])
