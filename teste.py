from evostra.models import FeedForwardNetwork
import pickle
from environment import make_get_reward, env_info

import argparse

parser = argparse.ArgumentParser(description='Evolution Strategies. ')
parser.add_argument('--env',default="Humanoid-v2")

args = parser.parse_args()

observationSpace, actionSpace = env_info(args.env)


# A feed forward neural network with input size of 5, two hidden layers of size 4 and output of size 3
model = FeedForwardNetwork(layer_sizes=[observationSpace, 64, 32,32,32,16, actionSpace])

with open(args.env +".pkl",'rb') as fp:
    model.set_weights(pickle.load(fp))

get_reward=make_get_reward(args.env, model,args.env)
while True:    
    print(get_reward(model.get_weights(),True))
