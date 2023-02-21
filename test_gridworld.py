import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
import argparse


def get_alg_module(alg, submodule=None):
    from importlib import import_module
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['models', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def main():
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='algorithm to use', default='policy_iteration')
    parser.add_argument('--run_test', help='run test', action='store_true')
    args = parser.parse_args()
    print(args.alg)

    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    learn = get_learn_function(args.alg)
    model = learn(env, max_it=100)

    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = model.get_policy(s)
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

        model.get_log().add('trajectory', env.get_pos(s), 'trajectory', color='red', alpha=0.5)

    model.get_log().plot()

    # Plot data and save to png file
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/gridworld/test_gridworld.png')


if __name__ == '__main__':
    main()
