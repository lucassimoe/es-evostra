import numpy as np
import gym
from gym.spaces.discrete import Discrete

from evostra import EvolutionStrategy
from evostra.models import FeedForwardNetwork
import pickle


import numpy
import array
import random
from deap import creator, base, tools, algorithms


import argparse

parser = argparse.ArgumentParser(description='Evolution Strategies. ')
parser.add_argument('--env',default="Humanoid-v2")
parser.add_argument('--render',type=bool,default=False)

args = parser.parse_args()

def env_info(env_name):
    env = gym.make(env_name)

    observation_space = env.observation_space.shape[0]
    if type(env.action_space) == Discrete:
        action_space = env.action_space.n
    else:
        action_space = env.action_space.shape[0]
    env.close()
    return observation_space, action_space

observationSpace, actionSpace = env_info(args.env)

# A feed forward neural network with input size of 5, two hidden layers of size 4 and output of size 3
model = FeedForwardNetwork(layer_sizes=[observationSpace, 32,16, actionSpace])
numWeights = model.get_weights()

IND_SIZE = numWeights #tamanho do individuo, sendo a quantidade de pesos (376*32*16*17)
MIN_VALUE = 4
MAX_VALUE = 5
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3
max_step = 1000

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", array.array, typecode="d")

# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
    IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)


def evalOneMax(individual):
    return sum(individual),

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", benchmarks.sphere)

toolbox.decorate("mate", checkStrategy(MAX_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MAX_STRATEGY))

def create_environment():

def fitness(individual):
    reward = simulate(individual)

    return fit

def simulate(individual):
    env = gym.make(args.env) #criando ambiente para cada indivíduo "pesos"
    #global model
    model.set_weights(individual)  #alterando os pesos da rede
    # here our best reward is zero
    reward = 0
    obs = env.reset()
    for step in range(max_step):
        if args.render:
            env.render()
        #print(model.predict(obs))
        action = np.argmax(model.predict(obs)) # your agent here (this takes random actions)

        obs, rew, done, info = env.step(action)
        reward+=rew

        if done:
            #print(step)
            break
    env.close()
    return reward



get_reward = None

def make_get_reward(env_name, model, _render):
    global   get_reward
    def get_reward(weights, render=_render):
        env = gym.make(env_name)
        #global model
        model.set_weights(weights)
        # here our best reward is zero
        reward = 0
        obs = env.reset()
        for step in range(max_step):
            if render:
                env.render()
            #print(model.predict(obs))
            action = np.argmax(model.predict(obs)) # your agent here (this takes random actions)

            obs, rew, done, info = env.step(action)
            reward+=rew

            if done:
                #print(step)
                break
        env.close()
        #env.reset()   
        return reward
    return get_reward


def main():
    random.seed()
    MU, LAMBDA = 10, 100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
        cxpb=0.6, mutpb=0.3, ngen=500, stats=stats, halloffame=hof)
    
    return pop, logbook, hof
    
if __name__ == "__main__":
    main()