import pickle
import matplotlib.pyplot as plt
import statistics
import numpy as np

save_file = "model_checkpoint/score_history"

with open(save_file, 'rb') as f:
    data = np.array(pickle.load(f))

def plot_data(data):
    num_epoch = len(data)
    plt.title("Game score history")
    plt.plot(range(1,num_epoch+1), data)
    plt.xlabel("Game")
    plt.ylabel("Game Score")
    plt.legend(loc='best')
    plt.show()

#plot_data(data)
a = data[data>0] - 1
b = data[data>1] - 2

#print "Mean:", statistics.mean(data)
#print "Median:", statistics.median(data)
#print "Mode:", statistics.mode(data)
#print "Stdev:", statistics.stdev(data)
