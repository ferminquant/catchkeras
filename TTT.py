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
		not_a_draw, game_over = self.is_over()
		if not_a_draw;
			#I won
			return self.observe(), 1, game_over
		elif game_over:
			#it was a draw
			return self.observe(), -1, game_over
		else:
			#I did not win, and it is the opponent's turn
			valid_action = False
			q = opponent.predict(self.state)[0]
			while not valid_action;
				action2 = np.argmax{q}
				if self.state[action2] == -1:
					valid_action = True
				else:
					del q[action2]
			self.state[action] = 0
			not_a_draw, game_over = self.is_over()
			if not_a_draw;
				#I lost
				return self.observe(), -2, game_over
			elif game_over:
				#it was a draw
				return self.observe(), -1, False
			else:
				#the game continues
				return self.observe(), 0, False
		
	
	def _is_over(self):
		if self.state[0] == self.state[1] and self.state[0] == self.state[2]:
			not_a_draw = True
		elif self.state[3] == self.state[4] and self.state[3] == self.state[5]:
			not_a_draw =  True
		elif self.state[6] == self.state[7] and self.state[6] == self.state[8]:
			not_a_draw =  True
		elif self.state[0] == self.state[3] and self.state[0] == self.state[6]:
			not_a_draw =  True
		elif self.state[1] == self.state[4] and self.state[1] == self.state[7]:
			not_a_draw =  True
		elif self.state[2] == self.state[5] and self.state[2] == self.state[8]:
			not_a_draw =  True
		elif self.state[0] == self.state[4] and self.state[0] == self.state[8]:
			not_a_draw =  True
		elif self.state[2] == self.state[4] and self.state[2] == self.state[6]:
			not_a_draw =  True
		not_a_draw =  False
		if not_a_draw:
			game_over = True
		else:
			game_over = -1 in self.state
		return not_a_draw, game_over

class ExpRep(object):
	def __init__(self, max_memory=100, discount=0.9):
		self.max_memory = max_memory
		self.memory = list()
		self.discount = discount
	
	def remember(self, states, game_over):
		self.memory.append([states,game_over])
		if len(self.memory) > self.max_memory:
			del self.memory[0]
			
	def get_batch(self, model, batch_size=10):
		len_memory = len(self.memory)
		num_actions = model.output_shape[-1]
		env_dim = self.memory[0][0][0].shape[1]
		inputs = np.zeros((min(len_memory, batch_size), env_dim))
		targets = np.zeros((inputs.shape[0], num_actions))
		for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
			state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
			game_over = self.memory[idx][1]
			inputs[i:i+1] = state_t
			targets[i] = model.predict(state_t)[0]
			Q_sa = np.max(model.predict(state_tp1)[0])
			if game_over:
				targets[i, action_t] = reward_t
			else:
				targets[i, action_t] = reward_t + self.discount * Q_sa
		return inputs, targets

if __name__ == "__main__":
	epsilon = 0.1
	num_actions = 9
	epoch = 5
	max_memory = 500
	batch_size = 50
	hidden_size = 90
	
	model = Sequential()
	model.add(Dense(hidden_size, input_shape=(num_actions,), activation='relu'))
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dense(num_actions))
	model.compile(sgd(lr=.2), "mse")
	
	#save v0 model, run only first time ever
	model.save_weights("TTTModel_v0.h5", overwrite=True)
	with open("TTTModel_v0.json", "w") as outfile:
		json.dump(model.to_json(), outfile)
	
	# to continue from a previous model 
	# model.load_weight("TTTModel.h5")
	
	#load opponent
	opponent = Sequential()
	opponent.add(Dense(hidden_size, input_shape=(num_actions,), activation='relu'))
	opponent.add(Dense(hidden_size, activation='relu'))
	opponent.add(Dense(num_actions))
	opponent.compile(sgd(lr=.2), "mse")
	opponent.load_weight("TTTModel_v0.h5")
	
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
			
			#make a move
			input_t, reward, game_over = env.act(action, opponent)
			if reward == 1:
				win_cnt += 1
			elif reward == -1:
				draw_cnt += 1
			elif reward == -2:
				lose_cnt += 1
				
			#remember
			exp_rep.remember([input_tm1, action, reward, input_t], game_over)
			
			#get batch of previous experiences for training
			inputs, targets = exp_rep.get_batch(model, batch_size=batch_size)
			
			#train neural network on previous experiences
			loss += model.train_on_batch(inputs, targets)
			
			print("{:02d}/{:02d} |{:02d}|{:02d}|{:02d}| {:.4f}".format(e,epoch,win_cnt,draw_cnt,lose_cnt,loss))
	
	#save model
	model.save_weights("TTTModel.h5", overwrite=True)
	with open("TTTModel.json", "w") as outfile:
		json.dump(model.to_json(), outfile)