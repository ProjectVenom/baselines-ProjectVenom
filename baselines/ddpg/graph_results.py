import matplotlib.pyplot as plt
log_file = open('./logs.txt', 'r')
lines = log_file.readlines()
x = []
y = []
average = []
t = 100.0
last_t = []
for line in lines:
    data = line.split(' ')
    #if True:
    if (float(data[1]) >= -1 and float(data[1]) <= 500):
        x.append(int(data[0][:-1]))
        y.append(float(data[1]))

        if len(last_t) < t:
            last_t.append(float(data[1]))
            average.append(0)
        else:
            last_t.pop(0)
            last_t.append(float(data[1]))
            average.append(sum(last_t)/t)

fig, ax = plt.subplots()
print(sum(last_t)/t)
ax.plot(x, y)
ax.plot(x,average)
ax.set(xlabel='time', ylabel='reward',
       title='DDPG with 3 CNN Layers on 144p RGB')
ax.grid()
plt.show()
