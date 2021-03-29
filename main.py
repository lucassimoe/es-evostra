from evostra import EvolutionStrategy
from evostra.models import FeedForwardNetwork
import pickle
from environment import make_get_reward, env_info


import argparse

parser = argparse.ArgumentParser(description='Evolution Strategies. ')
parser.add_argument('--env',default="Humanoid-v2")
parser.add_argument('--render',type=bool,default=False)

args = parser.parse_args()

observationSpace, actionSpace = env_info(args.env)

# A feed forward neural network with input size of 5, two hidden layers of size 4 and output of size 3
model = FeedForwardNetwork(layer_sizes=[observationSpace, 32, 16, actionSpace])


get_reward = make_get_reward(args.env, model,args.render)
# if your task is computationally expensive, you can use num_threads > 1 to use multiple processes;
# if you set num_threads=-1, it will use number of cores available on the machine; Here we use 1 process as the
#  task is not computationally expensive and using more processes would decrease the performance due to the IPC overhead.
es = EvolutionStrategy(model.get_weights(), get_reward, population_size=20, sigma=0.1, learning_rate=0.03, decay=0.995, num_threads=1)
es.run(1000, print_step=100)
with open(args.env +".pkl", 'wb') as fp:
    pickle.dump(es.get_weights(), fp)
#while True:
#   print(get_reward(es.get_weights(),True))
    