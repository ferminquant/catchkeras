import json
import numpy as np
import time
from keras.model import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

class TTT(object)
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.state = np.zeros((9))-1
	
	def observe(self):
		return self.state
	
	def act(self,action,opponent):
		self.state[action] = 1
		if not self.is_over();
			valid_action = False
			q = opponent.predict(self.state)[0]
			while not valid_action;
				action2 = np.argmax{q}
				if self.state[action2] == -1:
					valid_action = True
				else:
					del q[action2]
			
		else:
			#aqui gano
	
	def _is_over(self):
		if self.state[0] == self.state[1] and self.state[0] == self.state[2]:
			return True
		elif self.state[3] == self.state[4] and self.state[3] == self.state[5]:
			return True
		elif self.state[6] == self.state[7] and self.state[6] == self.state[8]:
			return True
		elif self.state[0] == self.state[3] and self.state[0] == self.state[6]:
			return True
		elif self.state[1] == self.state[4] and self.state[1] == self.state[7]:
			return True
		elif self.state[2] == self.state[5] and self.state[2] == self.state[8]:
			return True
		elif self.state[0] == self.state[4] and self.state[0] == self.state[8]:
			return True
		elif self.state[2] == self.state[4] and self.state[2] == self.state[6]:
			return True
		return False

class ExpRep(object):
	def __init__(self, max_memory=100, discount=0.9):
		self.max_memory = max_memory
		self.memory = list()
		self.discount = discount

if __name__ == "__main__":
	epsilon = 0.1
	num_actions = 9
	epoch = 10
	max_memory = 500
	batch_size = 50
	hidden_size = 90
	
	model = Sequential()
	model.add(Dense(hidden_size, input_shape=(num_actions,), activation='relu'))
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dense(num_actions))
	model.compile(sgd(lr=.2), "mse")
	
	# to continue from a previous model 
	# model.load_weight("TTTModel.h5")
	
	#load opponent
	opponent = Sequential()
	opponent.add(Dense(hidden_size, input_shape=(num_actions,), activation='relu'))
	opponent.add(Dense(hidden_size, activation='relu'))
	opponent.add(Dense(num_actions))
	opponent.compile(sgd(lr=.2), "mse")
	#opponent.load_weight("TTTModel_v0.h5")
	
	env = TTT()
	exp_rep = ExpRep(max_memory=max_memory)
	
	for e in range(epoch):
		loss = 0.
		env.reset()
		game_over = False
		input_t = env.observe()
		
		while not game_over:
			input_tm1 = input_t
			valid_action = False
			if np.random.rand() <= epsilon:
				while not valid_action:
					action = np.random.randint(0, num_actions, size=1)
					if input_tm1[action] == -1:
						valid_action = True
			else:
				q = model.predict(input_tm1)[0]
				while not valid_action:
					action = np.argmax(q)
					if input_tm1[action] == -1:
						valid_action = True
					else:
						del q[action]
			
			input_t, reward, game_over = env.act(action, opponent)
			if reward == 1:
				win_cnt += 1
			else:
				if reward == 0:
					tie_cnt += 1
				else:
					lose_cnt += 1
	
	#save model
	model.save_weights("TTTModel.h5", overwrite=True)
	with open("TTTModel.json", "w") as outfile:
		json.dump(model.to_json(), outfile)