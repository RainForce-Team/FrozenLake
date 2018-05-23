import gym
import numpy as np
import tensorflow as tf

env = gym.make('FrozenLake-v0')

env.seed(1778)
np.random.seed(1778)
tf.set_random_seed(1778)


# The grid with 4x4 gives 16 possible states, hence we have an array of 16 states.
inputs = tf.placeholder(shape=[1, 16], dtype=tf.float32)
# Each state has 4 possible outcomes, hence we have 16x4 matrix with weights uniformly distributed
weights = tf.Variable(tf.random_uniform([16, 4], 0, 0.1))
# Find the dot product of inputs and the weights
Q1 = tf.matmul(inputs, weights)
# The next state will be the opted based on the argmax function.
output = tf.argmax(Q1, 1)

Q2 = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Q2 - Q1))
gdo = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updatedweights = gdo.minimize(loss)


def main():
    GAMMA = 0.9
    EPSILON = 0.8
    EPISODES = 2000

    total_Reward = 0

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    rList = []

    for i in range(EPISODES):
        state_now = env.reset()
        done = False
        reward = 0
        rAll = 0
        for j in range(100):
            # Lets find the dot product of the inputs with the weights and apply argmax on it.
            action, Y = session.run([output, Q1], feed_dict={inputs: [np.eye(16)[state_now]]})
            if EPSILON > np.random.rand(1):
                action[0] = env.action_space.sample()
                EPSILON -= 10 ** -3
            # Lets iterate to the next state Note: This can be random.
            state_next, reward, done, _ = env.step(action[0])
            Y1 = session.run(Q1, feed_dict={inputs: [np.eye(16)[state_next]]})
            change_Y = Y
            change_Y[0, action[0]] = reward + GAMMA * np.max(Y1)
            # Updating the weights
            _, new_weights = session.run([updatedweights, weights],
                                         feed_dict={inputs: [np.eye(16)[state_now]], Q2: change_Y})
            # Lets append the total number of rewards
            total_Reward += reward
            rAll += reward
            state_now = state_next
            if reward == 1:
                print ('Episode {} was successful, Agent reached the Goal'.format(i))
        rList.append(rAll)

    session.close()
    print("Score over time: " + str(sum(rList) / EPISODES))


main()
