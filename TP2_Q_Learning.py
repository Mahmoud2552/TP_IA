import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[sprime]) - Q[s, a])
    return Q

def epsilon_greedy(Q, s, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample() 
    else:
        return np.argmax(Q[s])

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    n_epochs = 10000
    max_itr_per_epoch = 100
    rewards = []

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)
            Sprime, R, done, _, info = env.step(A)
            r += R
            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)
            S = Sprime
            if done:
                break

        clear_output(wait=True)
        rewards.append(r)

        print("Avr Reward ==> ", np.mean(rewards))

    plt.plot(rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward/Epoch : Q-learning')
    plt.show()

    print("Training finished.\n")

    env.close()
