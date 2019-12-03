import gym
import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf
import collections
import datetime as dt
from tensorboard import summary as summary_lib

env = gym.make('CartPole-v1')

np.random.seed(1)


class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, state_size, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, state_size], name="state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            # state_one_hot = tf.one_hot(self.state, int(state_size))
            state_one_hot = self.state
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
policy_learning_rate = 0.0004
value_learning_rate = 0.05
render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate=policy_learning_rate)
value_estimator = ValueEstimator(state_size, learning_rate=value_learning_rate)

LOGDIR = './TensorBoard/Q1' + f"/DQLearning_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
# Start training the agent with REINFORCE algorithm
with tf.Session() as sess, tf.summary.FileWriter(LOGDIR) as tb_logger:
    sess.run(tf.global_variables_initializer())
    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    step_done = 0
    sliding_avg = collections.deque(maxlen=100)
    # step_done = tf.get_variable('step_done', shape=[])
    i = 0
    for episode in range(max_episodes):
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []

        for step in range(max_steps):

            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_transitions.append(
                    Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
            episode_rewards[episode] += reward

            if done:
                step_done = step + 1
                sliding_avg.append(step_done)
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            state = next_state

        if solved:
            break
        avg_loss = 0.0
        # Compute Rt for each time-step t and update the network's weights
        for t, transition in enumerate(episode_transitions):
            total_discounted_return = sum(
                discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt

            baseline_value = value_estimator.predict(transition.state)
            advantage = total_discounted_return - baseline_value
            value_estimator.update(transition.state, total_discounted_return)

            feed_dict = {policy.state: transition.state, policy.R_t: advantage,
                         policy.action: transition.action}
            _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
            avg_loss += loss

        tf.summary.scalar('reward', step_done)
        tf.summary.scalar('avg loss', avg_loss / step_done)
        tf.summary.scalar('reward_avg_100_eps', sum(sliding_avg) / len(sliding_avg))
