import os
import logging
import numpy as np
import torch
import minerl  # noqa: register MineRL envs as Gym envs.
import gym

import wandb

import pfrl



# local modules
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))
import utils
from config import setSeed, getConfig
from env_wrappers import wrap_env
from q_functions import parse_arch


from IPython import embed

logger = logging.getLogger(__name__)


def main():
    if not os.path.exists('./results'):
        os.mkdir('./results')
        
    conf = getConfig(sys.argv[1])

    exp_id = 'eval_' if conf['demo'] else 'train_'
    exp_id += conf['outdir']

    wandb.init(
        project="mineRL",
        config=conf
    )

    wandb.run.name = exp_id
    wandb.run.save()

    args = str(conf)
    outdir = pfrl.experiments.prepare_output_dir(args, 'results', exp_id=exp_id)

    log_format = '%(levelname)-8s - %(asctime)s - [%(name)s %(funcName)s %(lineno)d] %(message)s'
    logging.basicConfig(filename=os.path.join(outdir, 'log.txt'), format=log_format, level=conf['logging_level'])
    console_handler = logging.StreamHandler()
    console_handler.setLevel(conf['logging_level'])
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger('').addHandler(console_handler)  # add hander to the root logger

    logger.info('Output files will be saved in {}'.format(outdir))

    try:
        dqn_family(conf, outdir)
    except:  # noqa
        logger.exception('execution failed.')
        raise


def dqn_family(conf, outdir):
    env_id = conf['env']
    world = conf['world']
    logging_level = conf['logging_level']
    interactor = conf['interactor']
    downstream_task = conf['downstream_task']

    seed = conf['seed']
    gpu = conf['gpu']

    demo = conf['demo']
    monitor = conf['monitor']
    load = conf['load']
    eval_n_runs = conf['eval_n_runs']
    sampling = conf['sampling']

    agent_type = conf['agent']
    arch = conf['arch']

    max_episode_steps = conf['max_episode_steps']
    batch_size = conf['batch_size']
    update_interval = conf['update_interval']
    frame_skip = conf['frame_skip']
    gamma = conf['gamma']
    clip_delta = conf['clip_delta']
    num_step_return = conf['num_step_return']
    lr = conf['lr']
    adam_eps = conf['adam_eps']

    batch_accumulator = conf['batch_accumulator']
    gray_scale = conf['gray_scale']
    frame_stack = conf['frame_stack']

    final_exploration_frames = conf['final_exploration_frames']
    final_epsilon = conf['final_epsilon']
    eval_epsilon = conf['eval_epsilon']
    noisy_net_sigma = conf['noisy_net_sigma']
    replay_capacity = conf['replay_capacity']
    replay_start_size = conf['replay_start_size']
    prioritized = conf['prioritized']
    target_update_interval = conf['target_update_interval']

    enc_conf = conf['encoder']
    data_type = enc_conf['data_type']

    world_conf = getConfig('CustomWorlds/' + world)
    world_conf['downstream_task'] = downstream_task

    coords = world_conf['coords']

    os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = outdir

    # Set a random seed used in PFRL.
    pfrl.utils.set_random_seed(seed)

    # Set different random seeds for train and test envs.
    train_seed = seed  # noqa: never used in this script
    test_seed = 2 ** 31 - 1 - seed

    # Load encoder #####################################
    if os.getenv('USER') == 'juanjo':
        path_weights = Path('./results/')
        world_conf['path_world'] = Path('/home/juanjo/Documents/minecraft/mineRL/src/minerl/env/Malmo/Minecraft/run/saves/')
    elif os.getenv('USER') == 'juan.jose.nieto':
        path_weights = Path('/home/usuaris/imatge/juan.jose.nieto/mineRL/src/results')
        world_conf['path_world'] = Path('/home/usuaris/imatge/juan.jose.nieto/mineRL/src/minerl/env/Malmo/Minecraft/run/saves/')
    else:
        raise Exception("Sorry, user not identified!")


    if enc_conf['type'] == 'random':
        train_encoder = True
        input_dim = 1024
        encoder = None
    else:
        train_encoder = False
        input_dim = enc_conf[enc_conf['type']]['embedding_dim'] * 2
        encoder = utils.load_encoder(enc_conf, path_weights)


    ######################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create & wrap env
    def wrap_env_partial(env, test):
        randomize_action = test and noisy_net_sigma is None
        wrapped_env = wrap_env(
            env=env, test=test,
            monitor=monitor, outdir=outdir,
            frame_skip=frame_skip, data_type=data_type,
            gray_scale=gray_scale, frame_stack=frame_stack,
            randomize_action=randomize_action, eval_epsilon=eval_epsilon,
            encoder=encoder, device=device, sampling=sampling,
            train_encoder=train_encoder, downstream_task=downstream_task, coords=coords)
        return wrapped_env
    logger.info('The first `gym.make(MineRL*)` may take several minutes. Be patient!')
    core_env = gym.make(env_id)

    core_env.custom_update(world_conf)

    if interactor: core_env.make_interactive(port=6666, realtime=True)

    # This seed controls which environment will be rendered
    core_env.seed(0)

    # training env
    env = wrap_env_partial(env=core_env, test=False)
    # env.seed(int(train_seed))

    # evaluation env
    eval_env = wrap_env_partial(env=core_env, test=True)
    # env.seed(int(test_seed))  # TODO: not supported yet (also requires `core_eval_env = gym.make(args.env)`)

    # calculate corresponding `steps` and `eval_interval` according to frameskip
    # 8,000,000 frames = 1333 episodes if we count an episode as 6000 frames,
    # 8,000,000 frames = 1000 episodes if we count an episode as 8000 frames.
    maximum_frames = 8000000
    if frame_skip is None:
        steps = maximum_frames
        eval_interval = 2000 * 20  # (approx.) every 20 episode (counts "1 episode = 2000 steps")
    else:
        steps = maximum_frames // frame_skip
        eval_interval = 2000 * 30 // frame_skip  # (approx.) every 100 episode (counts "1 episode = 6000 steps")


    agent = get_agent(
        n_actions=4, arch=arch, n_input_channels=env.observation_space.shape[0],
        noisy_net_sigma=noisy_net_sigma, final_epsilon=final_epsilon,
        final_exploration_frames=final_exploration_frames, explorer_sample_func=env.action_space.sample,
        lr=lr, adam_eps=adam_eps,
        prioritized=prioritized, steps=steps, update_interval=update_interval,
        replay_capacity=replay_capacity, num_step_return=num_step_return,
        agent_type=agent_type, gpu=gpu, gamma=gamma, replay_start_size=replay_start_size,
        target_update_interval=target_update_interval, clip_delta=clip_delta,
        batch_accumulator=batch_accumulator, batch_size=batch_size, input_dim=input_dim,
        train_encoder=train_encoder
    )

    if load:
        agent.load(load)
        print('agent loaded')

    # experiment
    if demo:
        eval_stats = pfrl.experiments.eval_performance(env=eval_env, agent=agent, n_steps=None, max_episode_len=max_episode_steps, n_episodes=eval_n_runs)
        logger.info('n_runs: {} mean: {} median: {} stdev {}'.format(
            eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']))
    else:
        pfrl.experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=steps,
            eval_n_steps=None, eval_n_episodes=eval_n_runs, eval_interval=eval_interval,
            outdir=outdir, eval_env=eval_env, save_best_so_far_agent=True, use_tensorboard=True
        )

    env.close()
    eval_env.close()


def parse_agent(agent):
    return {'DQN': pfrl.agents.DQN,
            'DoubleDQN': pfrl.agents.DoubleDQN,
            'PAL': pfrl.agents.PAL,
            'CategoricalDoubleDQN': pfrl.agents.CategoricalDoubleDQN}[agent]


def get_agent(
        n_actions, arch, n_input_channels,
        noisy_net_sigma, final_epsilon, final_exploration_frames, explorer_sample_func,
        lr, adam_eps,
        prioritized, steps, update_interval, replay_capacity, num_step_return,
        agent_type, gpu, gamma, replay_start_size, target_update_interval, clip_delta,
        batch_accumulator,batch_size, input_dim, train_encoder
):
    # Q function
    q_func = parse_arch(arch, n_actions, n_input_channels=n_input_channels, input_dim=input_dim, train_encoder=train_encoder)

    # explorer
    if noisy_net_sigma is not None:
        pfrl.nn.to_factorized_noisy(q_func, sigma_scale=noisy_net_sigma)
        # Turn off explorer
        explorer = pfrl.explorers.Greedy()
    else:
        explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
            1.0, final_epsilon, final_exploration_frames, explorer_sample_func)

    opt = torch.optim.Adam(q_func.parameters(), lr, eps=adam_eps)

    # Select a replay buffer to use
    if prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = steps / update_interval
        rbuf = pfrl.replay_buffers.PrioritizedReplayBuffer(
            replay_capacity, alpha=0.5, beta0=0.4, betasteps=betasteps, num_steps=num_step_return)
    else:
        rbuf = pfrl.replay_buffers.ReplayBuffer(replay_capacity, num_step_return)

    # build agent
    def phi(x):
        # observation -> NN input
        return np.asarray(x)
    Agent = parse_agent(agent_type)
    agent = Agent(
        q_func, opt, rbuf, gpu=gpu, gamma=gamma, explorer=explorer, replay_start_size=replay_start_size,
        target_update_interval=target_update_interval, clip_delta=clip_delta, update_interval=update_interval,
        batch_accumulator=batch_accumulator, phi=phi, minibatch_size=batch_size)

    return agent


if __name__ == '__main__':
    main()
