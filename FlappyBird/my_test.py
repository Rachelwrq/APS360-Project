import pickle
import matplotlib.pyplot as plt

#model_game_time = [100, 49, 102, 31, 50, 61, 31, 50, 100, 100, 31, 100, 31, 91, 65, 31, 31, 31, 31, 50, 48, 69, 31, 47, 100, 100, 52, 45, 47, 97, 46, 31, 102, 94, 59, 50, 92, 83, 38, 95, 87, 37, 92, 78, 98, 100, 54, 104, 109, 35, 93, 31, 48, 105, 54, 31, 100, 50, 139, 31, 40, 31, 88, 97, 31, 31, 66, 31, 70, 95, 50, 45, 31, 50, 100, 45, 48, 67, 100, 50, 41, 31, 31, 31, 31, 31, 49, 50, 31, 31, 31, 31, 48, 31, 69, 31, 31, 50, 31, 31, 67, 31, 49, 31]
#baseline_game_time = [31, 100, 100, 100, 40, 31, 100, 100, 100, 100, 31, 100, 100, 100, 31, 31, 100, 31, 100, 100, 31, 31, 31, 100, 100, 100, 100, 100, 31, 100, 100, 100, 100, 31, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 31, 31, 100, 31, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
#model_game_time = model_game_time[:60]
#baseline_game_time = baseline_game_time[:60]
#randome_game_time = randome_game_time[:60]

with open ('model_checkpoint/loss_history', 'rb') as fp:
	loss_history = pickle.load(fp)

num_epoch = len(loss_history)
plt.plot(range(1,num_epoch+1), loss_history)
plt.xlabel("Number of games played")
plt.ylabel("Training loss)")
plt.legend(loc='best')
plt.show()

