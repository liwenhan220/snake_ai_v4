from snake_game import snake_game
env = snake_game()
import numpy as np
import random
import cv2

inp = str(input('train from last? (y/n): '))
if inp == 'y':
    MODEL = str(input('pick a model: '))
    epsilon = 0.0
else:
    MODEL = None
    epsilon = 1

ANALYSIS_SERVICE = False

if ANALYSIS_SERVICE:
    file_name = input('what name would you like for data to be save?')
GAMMA = 0.99
EPISODES = 10000
LR = 0.8
THRESH = 100


if MODEL is not None:
    print('loading model!!')
    MODEL = np.load(MODEL, allow_pickle=True)
    print('model loaded!!!')
else:
    print('building model!!!')
#    ,int((env.size-1)**2/10)
    MODEL = np.random.uniform(high=1,low=-1,size=(3,3,3,3,3,3,3,3,3,3,int((env.size-1)**2/10),4))
    print('model constructed!!!!')
try:
    last_record = int(input('last_record?'))
except:
    last_record = -1

def change(inp):
    inp *= 10
    if inp >= 10:
        x = 2
    elif inp > 0 and inp < 10:
        x = 1
    else:
        x = 0
        
    return int(x)

def preprocess(state):
    for i in range(len(state)-2):
        state[i] = change(state[i])
    for xx in range(len(state)-2,len(state)):
        state[xx] = int(state[xx])
    return tuple(state.astype(np.int))

def main():
    data = []
    counter = 0
    global epsilon
    global THRESH
    global last_record
    INIT_EPSILON = epsilon
    FINAL_EPSILON = 0.0
    TARGET_STEPS = 0.8 * EPISODES
    step_counter = 0
    for episode in range(EPISODES):
        current_state = preprocess(env.reset())
        done = False
        ep_reward = 0
        print(episode)
        while not done:
            step_counter += 1
            
            counter += 1
            if np.random.random() > epsilon:
                action = np.argmax(MODEL[current_state])
            else:
                action = np.random.randint(0, env.action_space)
            new_state, reward, done = env.step(action)
            reward = np.sign(reward)
            new_state = preprocess(new_state)
            current_q = MODEL[current_state][action]
            max_future_q = np.max(MODEL[new_state])
            
            if not done:
                MODEL[current_state][action] = (1-LR)*current_q+LR*(reward+GAMMA*max_future_q)
                
            else:
                MODEL[current_state][action] = reward
            current_state = new_state
            
            ep_reward += reward
            if episode % THRESH == 0:
                env.render()
                
            if FINAL_EPSILON <= epsilon <= INIT_EPSILON:
                epsilon = (FINAL_EPSILON - INIT_EPSILON)/TARGET_STEPS * step_counter + INIT_EPSILON
        
        if ep_reward > last_record:
            last_record = ep_reward
        if ANALYSIS_SERVICE:
            data.append([episode,ep_reward])
            np.save(file_name,data)
    np.save('model',MODEL)

if __name__ == '__main__':
    main()
