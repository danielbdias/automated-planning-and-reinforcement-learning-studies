# REINFORCE adapted from algorithm implemented here: https://github.com/SwamyDev/reinforcement/blob/master/example/reinforce.py

from collections import defaultdict, namedtuple
from reinforcement_learning.structures import EpisodeStats

import itertools
import sys
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("tensorflow 1.14 required")

from reinforcement.algorithm.reinforce import Reinforce
from reinforcement.agents.basis import BatchAgent
from reinforcement import tf_operations as tf_ops

class NoLog:
    def add_summary(self, *args, **kwargs):
        pass

    def add_graph(self, *args, **kwargs):
        pass

class ValueBaseline:
    def __init__(self, session, obs_dims, summary_writer, lr=0.1):
        self._session = session
        self._in_observations = tf1.placeholder(shape=(None, obs_dims), dtype=tf.float32, name="observations")
        self._in_returns = tf1.placeholder(shape=(None,), dtype=tf.float32, name="returns")
        nn = tf1.get_variable("nn", shape=(obs_dims, 1), dtype=tf.float32, initializer=tf1.glorot_uniform_initializer())
        self._out_prediction = tf.squeeze(tf.matmul(self._in_observations, nn))
        loss = tf1.math.reduce_mean(tf.math.squared_difference(self._in_returns, self._out_prediction), name="mse_loss")
        self._train = tf1.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

        self._summary_writer = summary_writer
        logs = [tf1.summary.scalar("mean_baseline_return", tf.reduce_mean(self._in_returns)),
                tf1.summary.scalar("baseline_loss", loss)]
        self._log_summary = tf1.summary.merge(logs)
        self._cur_episode = 0

    def estimate(self, trj):
        return self._session.run(self._out_prediction,
                                 {self._in_observations: trj.observations,
                                  self._in_returns: trj.returns})

    def fit(self, trj):
        _, log = self._session.run([self._train, self._log_summary],
                                   {self._in_observations: trj.observations,
                                    self._in_returns: trj.returns})
        self._summary_writer.add_summary(log, self._cur_episode)
        self._cur_episode += 1

class ParameterizedPolicy:
    def __init__(self, session, obs_dims, num_actions, summary_writer, lr=10):
        self._session = session
        self._lr = lr
        self._in_actions = tf1.placeholder(shape=(None,), dtype=tf.uint8, name="actions")
        self._in_returns = tf1.placeholder(shape=(None,), dtype=tf.float32, name="returns")
        self._in_observations = tf1.placeholder(shape=(None, obs_dims), dtype=tf.float32, name="observations")
        theta = tf1.get_variable(f"theta", shape=(obs_dims, num_actions), dtype=tf.float32,
                                 initializer=tf1.glorot_uniform_initializer())
        self._out_probabilities = tf.nn.softmax(tf.matmul(self._in_observations, theta))
        self._train = None

        self._logs = [tf1.summary.scalar(f"mean_normalized_return", tf.reduce_mean(self._in_returns))]
        self._log_summary = tf.no_op
        self._summary_writer = summary_writer
        self._cur_episode = 0

    def set_signal_calc(self, signal_calc):
        loss = -signal_calc(tf_ops, self._in_actions, self._out_probabilities, self._in_returns)
        self._train = tf1.train.GradientDescentOptimizer(learning_rate=self._lr).minimize(loss)
        self._session.run(tf1.global_variables_initializer())
        self._finish_logs(loss)

    def _finish_logs(self, loss):
        self._logs.append(tf1.summary.scalar("loss", loss))
        self._log_summary = tf1.summary.merge(self._logs)

    def estimate(self, observation):
        return np.squeeze(
            self._session.run(self._out_probabilities, {self._in_observations: np.array(observation).reshape(1, -1)}))

    def fit(self, trajectory):
        _, log = self._session.run([self._train, self._log_summary],
                                   {self._in_observations: trajectory.observations,
                                    self._in_actions: trajectory.actions,
                                    self._in_returns: trajectory.advantages})
        self._summary_writer.add_summary(log, self._cur_episode)
        self._cur_episode += 1

def _run_episode(env, episode, agent):
    obs = env.reset()
    done, reward = False, 0

    while not done:
        obs, r, done, _ = env.step(agent.next_action(obs))
        agent.signal(r)
        reward += r

    agent.train()
    return reward

def _make_config(num_episodes, discount_factor):
    Config = namedtuple("ReinforceConfig", ["episodes", "num_trajectories", "gamma", "lr_policy", "lr_baseline"])
    return Config(episodes=num_episodes, num_trajectories=10, gamma=discount_factor, lr_policy=50, lr_baseline=0.01)

def reinforce(env, num_episodes, discount_factor=1.0, verbose=False):
    config = _make_config(num_episodes, discount_factor)
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    tf1.reset_default_graph()
    with tf1.Session() as session:
        p = ParameterizedPolicy(session, env.observation_space.shape[0], env.action_space.n, NoLog(), config.lr_policy)
        b = ValueBaseline(session, env.observation_space.shape[0], NoLog(), config.lr_baseline)
        alg = Reinforce(p, config.gamma, b, config.num_trajectories)
        agent = BatchAgent(alg)

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if ((i_episode + 1) % 100 == 0) and verbose:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            obs = env.reset()

            for t in itertools.count():
                obs, reward, done, _ = env.step(agent.next_action(obs))
                agent.signal(reward)

                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    break

            agent.train()

    return stats
