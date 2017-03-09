import json
import numpy as np
import os.path
import copy
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

class TTT(object):
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.state = np.zeros((9))-1
	
	def observe(self):
		return copy.deepcopy(self.state[np.newaxis])
	
	def act(self,action,opponent):
		win_value = 1
		draw_value = -1
		lose_value = -2
		self.state[action] = 1
		not_a_draw, game_over = self._is_over()
		if not_a_draw:
			#I won
			return self.observe(), win_value, game_over
		elif game_over:
			#it was a draw
			return self.observe(), draw_value, game_over
		else:
			#I did not win, and it is the opponent's turn
			valid_action = False
			backstate = self.observe()
			for j in range(len(backstate)):
				if backstate[0][j] == 0:
					backstate[0][j] = 1
				elif backstate[0][j] == 1:
					backstate[0][j] = 0
			q = opponent.predict(backstate)[0]
			ql = list()
			for i in range(0,len(q)):
				ql.append([i,q[i]])
			ql = sorted(ql,reverse=True,key=lambda tuz: tuz[1])
			tmp_idx = 0;
			while not valid_action:
				action2 = ql[tmp_idx][0]
				if self.state[action2] == -1:
					valid_action = True
				else:
					tmp_idx = tmp_idx + 1
			self.state[action2] = 0
			not_a_draw, game_over = self._is_over()
			if not_a_draw:
				#I lost
				return self.observe(), lose_value, game_over
			elif game_over:
				#it was a draw
				return self.observe(), draw_value, game_over
			else:
				#the game continues
				return self.observe(), 0, game_over
		
	
	def _is_over(self):
		if self.state[0] == self.state[1] and self.state[0] == self.state[2] and self.state[0] != -1:
			not_a_draw = True
		elif self.state[3] == self.state[4] and self.state[3] == self.state[5] and self.state[3] != -1:
			not_a_draw =  True
		elif self.state[6] == self.state[7] and self.state[6] == self.state[8] and self.state[6] != -1:
			not_a_draw =  True
		elif self.state[0] == self.state[3] and self.state[0] == self.state[6] and self.state[0] != -1:
			not_a_draw =  True
		elif self.state[1] == self.state[4] and self.state[1] == self.state[7] and self.state[1] != -1:
			not_a_draw =  True
		elif self.state[2] == self.state[5] and self.state[2] == self.state[8] and self.state[2] != -1:
			not_a_draw =  True
		elif self.state[0] == self.state[4] and self.state[0] == self.state[8] and self.state[0] != -1:
			not_a_draw =  True
		elif self.state[2] == self.state[4] and self.state[2] == self.state[6] and self.state[2] != -1:
			not_a_draw =  True
		else:
			not_a_draw =  False
		if not_a_draw:
			game_over = True
		else:
			game_over = not(-1 in self.state)
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
			state_t, action_t, reward_t, state_tp1 = copy.deepcopy(self.memory[idx][0])
			game_over = self.memory[idx][1]
			inputs[i:i+1] = copy.deepcopy(state_t)
			targets[i] = model.predict(state_t)[0]
			#Q_sa = np.max(model.predict(state_tp1)[0])
			
			q = model.predict(state_tp1)[0]
			ql = list()
			for j in range(0,len(q)):
				if state_tp1[0][j] == -1:
					ql.append([j,q[j]])
			ql = sorted(ql,reverse=True,key=lambda tuz: tuz[1])
			#print(state_t)
			#print(state_tp1)
			#print("Action: {} | Reward: {} | GO: {}".format(action_t, reward_t, game_over))
			#print(ql)
			tmp = self.memory
			if len(ql) == 0 and not game_over:
				tmp = self.memory

			#Q[s,a] = Q[s,a] + alpha(reward + discount(maxa'Q[s',a']) - Q[s,a])			
			if game_over:
				Q_sa = 0.0
			else:
				Q_sa = q[ql[0][0]]
			targets[i, action_t] = targets[i, action_t] +  0.5*(reward_t + self.discount*(Q_sa) - targets[i, action_t])
		return inputs, targets
	
	def get_memory(self):
		return self.memory

if __name__ == "__main__":
	epsilon = 0.1
	num_actions = 9
	epoch = 1000
	max_memory = 500
	batch_size = 50
	hidden_size = 100
		
	opponent_version = 0
	found_last_version = False
	first_run_ever = False
	while not found_last_version:
		opponent_name = "TTTModel_v{}".format(opponent_version)
		if os.path.isfile("{}.h5".format(opponent_name)):
			opponent_version = opponent_version + 1
		else:
			found_last_version= True
			if opponent_version <= 1:
				if opponent_version == 0:
					opponent_version = 0
					first_run_ever = True
				else:
					opponent_version = opponent_version - 1
			else:
				#in case the last version is not complete
				opponent_version = opponent_version - 2
			
	model_version = opponent_version + 1
	
	model = Sequential()
	model.add(Dense(hidden_size, input_shape=(num_actions,), activation='relu'))
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dense(num_actions))
	model.compile(sgd(lr=.2), "mse")
	
	if first_run_ever:
		#save v0 model, run only first time ever, it is suppossed to be a new untrained model
		model.save_weights("TTTModel_v0.h5", overwrite=True)
		with open("TTTModel_v0.json", "w") as outfile:
			json.dump(model.to_json(), outfile)
	
	#opponent model
	opponent = Sequential()
	opponent.add(Dense(hidden_size, input_shape=(num_actions,), activation='relu'))
	opponent.add(Dense(hidden_size, activation='relu'))
	opponent.add(Dense(hidden_size, activation='relu'))
	opponent.add(Dense(num_actions))
	opponent.compile(sgd(lr=.2), "mse")
	
	#self improvement loop
	win_rate = 0.0
	cnt = 1
	while True:#model_version <= 2:
		epsilon = 1.0-(win_rate/2)
		if win_rate >= 0.9:
			opponent_version = opponent_version + 1
			model_version = opponent_version + 1
	
		#load models
		model_name = "TTTModel_v{}".format(model_version)
		opponent_name = "TTTModel_v{}".format(opponent_version)
		
		if os.path.isfile("{}.h5".format(model_name)):
			model.load_weights("{}.h5".format(model_name))
		else:
			model.load_weights("{}.h5".format(opponent_name))
		
		opponent.load_weights("{}.h5".format(opponent_name))
		
		env = TTT()
		exp_rep = ExpRep(max_memory=max_memory)
		win_cnt = 0
		draw_cnt = 0
		lose_cnt = 0
		loss_avg = list()
		
		for e in range(epoch):
			loss = 0.
			env.reset()
			game_over = False
			input_t = env.observe()
			
			while not game_over:
				input_tm1 = copy.deepcopy(input_t)
				valid_action = False
				if np.random.rand() <= epsilon:
					while not valid_action:
						action = np.random.randint(0, num_actions, size=1)
						if input_tm1[0][action] == -1:
							valid_action = True
				else:
					q = model.predict(input_tm1)[0]
					ql = list()
					for i in range(0,len(q)):
						if input_tm1[0][i] == -1:
							ql.append([i,q[i]])
					ql = sorted(ql,reverse=True,key=lambda tuz: tuz[1])
					action = ql[0][0]
					valid_action = True
					
				#make a move
				input_t, reward, game_over = env.act(action, opponent)
				if reward == 1:
					win_cnt += 1
				elif reward == -1:
					draw_cnt += 1
				elif reward == -2:
					lose_cnt += 1
					
				#remember
				tmp_cnt_1 = list(input_tm1[0]).count(-1)
				tmp_cnt_2 = list(input_t[0]).count(-1)
				if abs(tmp_cnt_2-tmp_cnt_1)>2:
					tmp2 = 0
				exp_rep.remember([input_tm1, action, reward, input_t], game_over)
				
				#get batch of previous experiences for training
				inputs, targets = exp_rep.get_batch(model, batch_size=batch_size)
				
				#train neural network on previous experiences
				loss += model.train_on_batch(inputs, targets)
			
			loss_avg.append(loss)
			if (e+1) % epoch == 0:
				print("{:03d} | {:03d}|{:03d}|{:03d} | {:.20f} | {:03d}vs{:03d}".format(cnt,win_cnt,draw_cnt,lose_cnt,sum(loss_avg)/epoch,opponent_version,model_version))
				
			#tuz = env.observe()[0]
			#print("{:02d}|{:02d}|{:02d}".format(int(tuz[0]),int(tuz[1]),int(tuz[2])))
			#print("{:02d}|{:02d}|{:02d}".format(int(tuz[3]),int(tuz[4]),int(tuz[5])))
			#print("{:02d}|{:02d}|{:02d}".format(int(tuz[6]),int(tuz[7]),int(tuz[8])))
			#print("")
				
		cnt = cnt + 1
		win_rate = 0.0 + float(win_cnt)/float(epoch)
		#save model
		model.save_weights("{}.h5".format(model_name), overwrite=True)
		with open("{}.json".format(model_name), "w") as outfile:
			json.dump(model.to_json(), outfile)