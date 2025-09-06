# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.
## PROBLEM STATEMENT

## POLICY EVALUATION FUNCTION
<img width="685" height="130" alt="image" src="https://github.com/user-attachments/assets/bb7f8627-3108-4d08-a052-de7522e75645" />

## PROGRAM
```

pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk

import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123);

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
LEFT, DOWN, RIGHT, UP = range(4)

P

init_state

pi_frozenlake = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7:LEFT,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:RIGHT,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]
print_policy(pi_frozenlake, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_frozenlake, goal_state=goal_state) * 100,
    mean_return(env, pi_frozenlake)))

pi_2 = lambda s:{
    0:RIGHT,
    1:LEFT,
    2:DOWN,
    3:UP,
    4:LEFT,
    5:DOWN,
    6:UP,
    7:DOWN,
    8:UP,
    9:RIGHT,
    10:UP,
    11:DOWN,
    12:LEFT,
    13:RIGHT,
    14:UP,
    15:LEFT #Stop
}[s]

print("Name: ALAN ZION H     ")
print("Register Number: 212223240004        ")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state) * 100,
    mean_return(env, pi_2)))

print("Policy 1 (FrozenLake mapping):")
print("  Success Rate: {:.2f}%".format(probability_success(env, pi_frozenlake, goal_state=goal_state) * 100))
print("  Avg Return: {:.4f}".format(mean_return(env, pi_frozenlake)))

print("\nPolicy 2 :")
print("  Success Rate: {:.2f}%".format(probability_success(env, pi_2, goal_state=goal_state) * 100))
print("  Avg Return: {:.4f}".format(mean_return(env, pi_2)))

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)

    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state] * (not done))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

V1 = policy_evaluation(pi_frozenlake, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=4, prec=5)

V2 = policy_evaluation(pi_2, P,gamma=0.99)
print_state_value_function(V2, P, n_cols=4, prec=5)

V1>=V2

if(np.sum(V1>=V2)>=11):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)>=11):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")y…]()
```
## OUTPUT:
 The first and second policies along with its state value function and compare them
 ```
<img width="595" height="168" alt="image" src="https://github.com/user-attachments/assets/f1558ac7-7bc5-40b4-a46a-95d8ad5b0ccf" />
<img width="774" height="45" alt="image" src="https://github.com/user-attachments/assets/a35d0cb6-d7fb-4ffa-b27e-c411c9f85f14" />

<img width="404" height="178" alt="image" src="https://github.com/user-attachments/assets/7c10327c-748f-4b47-8221-582c92ba04ba" />

<img width="654" height="137" alt="image" src="https://github.com/user-attachments/assets/5acc5a97-747c-4488-9594-852d27acf6a9" />

<img width="751" height="68" alt="image" src="https://github.com/user-attachments/assets/c986ac8d-ddff-4e87-a4fe-13b0d74ec220" />

<img width="453" height="51" alt="image" src="https://github.com/user-attachments/assets/c781690a-e901-4cda-ac3f-59702464fd23" />
```
## RESULT:
