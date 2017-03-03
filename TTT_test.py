import json
import numpy as np
import time
import os.path
from keras.models import model_from_json
from TTT import TTT

def print_board(tuz):
	print("{:02d}|{:02d}|{:02d}".format(int(tuz[0]),int(tuz[1]),int(tuz[2])))
	print("{:02d}|{:02d}|{:02d}".format(int(tuz[3]),int(tuz[4]),int(tuz[5])))
	print("{:02d}|{:02d}|{:02d}".format(int(tuz[6]),int(tuz[7]),int(tuz[8])))
	print("")

def my_action():
	return raw_input("Please enter your move: ")

def model_action(state):
	q = model.predict(state)[0]
	ql = list()
	for i in range(0,len(q)):
		ql.append([i,q[i]])
	ql = sorted(ql,reverse=True,key=lambda tuz: tuz[1])
	tmp_idx = 0;
	valid_action = False
	while not valid_action:
		action = ql[tmp_idx][0]
		if state[0][action] == -1:
			valid_action = True
		else:
			tmp_idx = tmp_idx + 1
	return action
	
if __name__ == "__main__":
	epsilon = 0.1
	num_actions = 9
	epoch = 1000
	max_memory = 500
	batch_size = 50
	hidden_size = 900
		
	model_version = raw_input("Please enter model version: ")
	model_name = "TTTModel_v{}".format(model_version)
	
	with open("{}.json".format(model_name), "r") as jfile:
		model = model_from_json(json.load(jfile))
	model.load_weights("{}.h5".format(model_name))
	model.compile("sgd", "mse")

	state = np.zeros((9))-1
	state = state[np.newaxis]
	
	while -1 in state:
	
		if True:
			#model goes first
			state[0][model_action(state)] = 1
			print_board(state[0])
			
			state[0][my_action()] = 0
			print_board(state[0])
		else:
			#player goes first
			state[0][my_action()] = 0
			print_board(state[0])
			
			state[0][model_action(state)] = 1
			print_board(state[0])
			