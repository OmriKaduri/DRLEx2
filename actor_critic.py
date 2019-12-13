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

    def __init__(self, state_size, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [1, state_size], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

            # This is just table lookup estimator
#             state_one_hot = tf.one_hot(, int(state_size))
#             print(state_one_hot)
            self.input_layer = tf.layers.dense(
                inputs=self.state,
                units=32,
                activation='relu',
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.hidden_layer = tf.layers.dense(
                inputs=self.input_layer,
                units=16,
                activation='relu',
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.output_layer = tf.layers.dense(
                inputs=self.hidden_layer,
                units=1,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.reduce_sum(tf.square(self.target - self.value_estimate))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
#         state = state.reshape(-1)
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, learning_rate, sess=None):
#         state = state.reshape(-1)
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.learning_rate:learning_rate}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class PolicyNetwork:
    def __init__(self, state_size, action_size, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
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
learning_rate = 0.001
value_learning_rate = 0.005
render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size)
value_estimator = ValueEstimator(state_size)

LOGDIR = './TensorBoard/Q2' + f"/DQLearning_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
# Start training the agent with REINFORCE algorithm
with tf.Session() as sess, tf.summary.FileWriter(LOGDIR) as tb_logger:
    sess.run(tf.global_variables_initializer())
    solved = False
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    step_done = 0

    sliding_avg = collections.deque(maxlen=100)
    avg_loss = 0.0
    for episode in range(max_episodes):
        if episode % 300 == 0 and episode > 0:
            learning_rate*=0.1
            value_learning_rate*=0.1
            
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []
        I = 1
        for step in range(max_steps):

            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            
            next_state = next_state.reshape([1, state_size])
            episode_rewards[episode] += reward
            reward = +1 if not done or step == 500 else -10
            if render:
                env.render()
            value_s = value_estimator.predict(state)
            value_next = 0
            if done:
                delta = reward - value_s
            else:
                value_next = value_estimator.predict(next_state)
                delta = reward + (discount_factor*value_next) - value_s
            
            total_discounted_return = I*delta
            value_estimator.update(state, reward + discount_factor*value_next, value_learning_rate)
            
            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
        
            feed_dict = {policy.state: state, policy.R_t: total_discounted_return,
                         policy.action: action_one_hot, policy.learning_rate:learning_rate}
            _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
            avg_loss += loss

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
                
#             I = discount_factor*I
            state = next_state

        if solved:
            break
            
        summary = tf.Summary(value=[tf.Summary.Value(tag='reward',
                                                     simple_value=step_done),
                                   tf.Summary.Value(tag='avg_loss',
                                                     simple_value=avg_loss / step_done),
                                   tf.Summary.Value(tag='reward_avg_100_eps',
                                                     simple_value=sum(sliding_avg) / len(sliding_avg))])
        tb_logger.add_summary(summary, episode)
