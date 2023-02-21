import numpy as np
import argparse
import gridworld


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='algorithm to use', default='policy_iteration')
    parser.add_argument('--run_test', help='run test', detault=False)
    args = parser.parse_args()
    print(args.alg)

    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    learn = get_learn_function(args.alg)
    model = learn(env)

    if args.run_test:
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
            a = model.step(s)
            (s, r, done) = env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)


if __name__ == '__main__':
    main()
