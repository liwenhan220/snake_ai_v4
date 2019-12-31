from snake_game import snake_game
env = snake_game()
import numpy as np
import random
import cv2


MODEL = str(input('pick a model: '))
GAMMA = 0.99
EPISODES = 10000
LR = 0.8
THRESH = 1
MODEL = np.load(MODEL, allow_pickle=True)
epsilon = 0
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
    #return np.array([change(x1), change(x2), change(x3), change(x4), change(x7), change(x8), change(x9), change(x10), 0,0])
    return tuple(state.astype(np.int))

def main():
    data = []
    counter = 0
    global epsilon
    global THRESH
    global last_record
    INIT_EPSILON = epsilon
    FINAL_EPSILON = 0.0
    TARGET_STEPS = 5000
     
    step_counter = 0
    for episode in range(EPISODES):
        current_state = preprocess(env.reset())
        done = False
        ep_reward = 0
        if episode % THRESH == 0:
            print('episode: {}'.format(episode))
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
            
            #print(reward,current_state)
            ep_reward += reward
            if episode % THRESH == 0:
                env.render()
                
            if FINAL_EPSILON <= epsilon <= INIT_EPSILON:
                epsilon = (FINAL_EPSILON - INIT_EPSILON)/TARGET_STEPS * step_counter + INIT_EPSILON
        
        data.append([episode,ep_reward])

if __name__ == '__main__':
    main()
