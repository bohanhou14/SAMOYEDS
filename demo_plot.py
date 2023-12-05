import matplotlib.pyplot as plt

labels = ['female', 'male']
data = [443, 50]
plt.pie(data, labels=labels)
plt.savefig("fig/500-hesitant-gender")