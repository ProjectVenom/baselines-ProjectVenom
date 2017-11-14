import matplotlib.pyplot as plt
import statistics
log_file = open('./logs.txt.2pdisc', 'r')
lines = log_file.readlines()
x = []
y = []
average = []
medians = []
t = 100.0
last_t = []
for line in lines:
    data = line.split(' ')
    #if True:
    if (float(data[1]) >= -3 and float(data[1]) <= 500):
        x.append(int(data[0][:-1]))
        y.append(float(data[1]))
        if len(last_t) < t:
            last_t.append(float(data[1]))
            average.append(sum(last_t)/len(last_t))
            medians.append(statistics.median(last_t))
        else:
            last_t.pop(0)
            last_t.append(float(data[1]))
            average.append(sum(last_t)/t)
            medians.append(statistics.median(last_t))

fig, ax = plt.subplots()
print(sum(last_t)/t)
print(medians[-1])
ax.plot(x, y)
ax.plot(x, medians)
#ax.plot(x,average)
ax.set(xlabel='episodes', ylabel='reward',
       title='')
ax.grid()
plt.show()
