import numpy as np
import cv2
from collections import deque
import math

class snake_game:
    def __init__(self):
        self.size = 41
        self.observation_space = (4,4,1)
        self.subject = (255, 0, 0)
        self.food = (0, 0, 255)
        self.segment = (255,255,255)
        self.wall = (255,255,255)
        self.total_reward = 0
        self.action_space = 4
        

    def init_game(self):
        self.img = np.zeros((self.size,self.size,3))
        for x in range(len(self.img)):
            self.img[x][0] = self.wall
            self.img[x][self.size-1] = self.wall
        for y in range(self.size):
            self.img[0][y] = self.wall
            self.img[self.size-1][y] = self.wall
        return self.img
    
    def reset(self):
        self.n = 0
        #self.size = np.random.randint(40,200)
        self.sub_x = int(self.size/2)
        self.sub_y = int(self.size/2)
        self.food_x = np.random.randint(2,self.size-2)
        self.food_y = np.random.randint(2,self.size-2)        
        self.img = self.init_game()
        self.img[self.sub_x][self.sub_y] = self.subject
        self.img[self.food_x][self.food_y] = self.food
        self.reward = 0
        self.total_reward = 0
        self.segments = []
        self.last_n = 5
        self.last_dist = math.sqrt((self.sub_x - self.food_x)**2+(self.sub_y-self.food_y)**2)
        res_img = self.img
        #cv2.line(res_img,(self.sub_y,self.sub_x),(self.food_y,self.food_x),(0,0,125),1)
        state = []
        for i in range(2):
            try:
                s1 = np.mean(self.img[self.sub_x-2*i][self.sub_y])/255.0
            except:
                s1 = 1
            try:
                s2 = np.mean(self.img[self.sub_x+2*i][self.sub_y])/255.0
            except:
                s2 = 1
            try:
                s3 = np.mean(self.img[self.sub_x][self.sub_y-2*i])/255.0
            except:
                s3 = 1
            try:
                s4 = np.mean(self.img[self.sub_x][self.sub_y+2*i])/255.0
            except:
                s4 = 1
##            try:
##                s5 = np.mean(self.img[self.sub_x-2*i][self.sub_y-2*i])/255.0
##            except:
##                s5 = 1
##            try:
##                s6 = np.mean(self.img[self.sub_x+2*i][self.sub_y+2*i])/255.0
##            except:
##                s6 = 1
##            try:
##                s7 = np.mean(self.img[self.sub_x-2*i][self.sub_y+2*i])/255.0
##            except:
##                s7 = 1
##            try:
##                s8 = np.mean(self.img[self.sub_x+2*i][self.sub_y-2*i])/255.0
##            except:
##                s8 = 1
            state.append(s1)
            state.append(s2)
            state.append(s3)
            state.append(s4)
##            state.append(s5)
##            state.append(s6)
##            state.append(s7)
##            state.append(s8)
        if self.food_x - self.sub_x < 0:
            s9 = 1
        elif self.food_x - self.sub_x > 0:
            s9 = 2
        else:
            s9 = 0
        if self.food_y - self.sub_y < 0:
            s10 = 1
        elif self.food_y - self.sub_y > 0:
            s10 = 2
        else:
            s10 = 0
            
        state.append(s9)
        state.append(s10)
        #self.img = self.update_frame()
        state.append(math.floor(self.n/10))
        return np.array(state)
                
    def update_frame(self):
        self.img = np.zeros((self.size,self.size,3))
        self.img = self.init_game()
        self.img[self.sub_x][self.sub_y] = self.subject
        self.img[self.food_x][self.food_y] = self.food
        try:
            for i in range(len(self.segments)):
                self.img[self.segments[i][0]][self.segments[i][1]] = self.segment
        except:
            pass
                                              
        return self.img

    def render(self):
        try:
            self.img = self.update_frame()
            cv2.imshow('snake_game',cv2.resize(self.img,(500,500)))
            cv2.waitKey(25)
                
        except:
            pass

    def step(self,input_n):
        self.n += 1

        if self.n > (self.size-1) ** 2:
            res_img = self.img
            state = []
            for i in range(2):
                try:
                    s1 = np.mean(self.img[self.sub_x-2*i][self.sub_y])/255.0
                except:
                    s1 = 1
                try:
                    s2 = np.mean(self.img[self.sub_x+2*i][self.sub_y])/255.0
                except:
                    s2 = 1
                try:
                    s3 = np.mean(self.img[self.sub_x][self.sub_y-2*i])/255.0
                except:
                    s3 = 1
                try:
                    s4 = np.mean(self.img[self.sub_x][self.sub_y+2*i])/255.0
                except:
                    s4 = 1
##                try:
##                    s5 = np.mean(self.img[self.sub_x-2*i][self.sub_y-2*i])/255.0
##                except:
##                    s5 = 1
##                try:
##                    s6 = np.mean(self.img[self.sub_x+2*i][self.sub_y+2*i])/255.0
##                except:
##                    s6 = 1
##                try:
##                    s7 = np.mean(self.img[self.sub_x-2*i][self.sub_y+2*i])/255.0
##                except:
##                    s7 = 1
##                try:
##                    s8 = np.mean(self.img[self.sub_x+2*i][self.sub_y-2*i])/255.0
##                except:
##                    s8 = 1
                state.append(s1)
                state.append(s2)
                state.append(s3)
                state.append(s4)
##                state.append(s5)
##                state.append(s6)
##                state.append(s7)
##                state.append(s8)
            state.append(math.floor(self.n/10))
            return np.array(state), -100, True
            
        if self.last_n == 0 and input_n == 1 or self.last_n == 2 and input_n == 3 or self.last_n == 1 and input_n == 0 or self.last_n == 3 and input_n == 2:
                input_n = self.last_n

        self.last_n = input_n
        # moving up
        if input_n == 0:
            self.sub_x -= 2
            
        # moving down
        elif input_n == 1:
            self.sub_x += 2
            
        # moving left
        elif input_n == 2:
            self.sub_y -= 2
            
        # moving right
        elif input_n == 3:
            self.sub_y += 2


        if self.sub_x > self.size - 2 or self.sub_y > self.size - 2 or self.sub_x < 2 or self.sub_y < 2:
            res_img = self.img
            state = []
            for i in range(2):
                try:
                    s1 = np.mean(self.img[self.sub_x-2*i][self.sub_y])/255.0
                except:
                    s1 = 1
                try:
                    s2 = np.mean(self.img[self.sub_x+2*i][self.sub_y])/255.0
                except:
                    s2 = 1
                try:
                    s3 = np.mean(self.img[self.sub_x][self.sub_y-2*i])/255.0
                except:
                    s3 = 1
                try:
                    s4 = np.mean(self.img[self.sub_x][self.sub_y+2*i])/255.0
                except:
                    s4 = 1
##                try:
##                    s5 = np.mean(self.img[self.sub_x-2*i][self.sub_y-2*i])/255.0
##                except:
##                    s5 = 1
##                try:
##                    s6 = np.mean(self.img[self.sub_x+2*i][self.sub_y+2*i])/255.0
##                except:
##                    s6 = 1
##                try:
##                    s7 = np.mean(self.img[self.sub_x-2*i][self.sub_y+2*i])/255.0
##                except:
##                    s7 = 1
##                try:
##                    s8 = np.mean(self.img[self.sub_x+2*i][self.sub_y-2*i])/255.0
##                except:
##                    s8 = 1
                state.append(s1)
                state.append(s2)
                state.append(s3)
                state.append(s4)
##                state.append(s5)
##                state.append(s6)
##                state.append(s7)
##                state.append(s8)

            if self.food_x - self.sub_x < 0:
                s9 = 1
            elif self.food_x - self.sub_x > 0:
                s9 = 2
            else:
                s9 = 0
            if self.food_y - self.sub_y < 0:
                s10 = 1
            elif self.food_y - self.sub_y > 0:
                s10 = 2
            else:
                s10 = 0
                
            state.append(s9)
            state.append(s10)
            state.append(math.floor(self.n/10))
            return np.array(state), -100, True
            
            
##        if self.sub_x > self.size-1 or self.sub_x < 0 or self.sub_y < 0 or self.sub_y > self.size-1:
##            self.terminal = True
##            self.reward = -1

        if all(self.img[self.sub_x][self.sub_y] == self.wall):
            self.terminal = True
            self.reward = -100
            
        elif math.sqrt((self.food_x - self.sub_x)**2) <= 1 and math.sqrt((self.food_y - self.sub_y)**2) <= 1:
            self.n = 0
            self.terminal = False
            if self.total_reward + 5 >= ((self.size-1)/2)**2-1:
                self.terminal = True
                
            self.reward = 100
            self.food_x = np.random.randint(2,self.size-2)
            self.food_y = np.random.randint(2,self.size-2)
            self.img = self.update_frame()    
        else:
            self.terminal = False
            self.reward = 0
            self.img = self.update_frame()

        self.segments.append([self.sub_x,self.sub_y])
        if len(self.segments) > self.total_reward + 5:
            del self.segments[:(len(self.segments)-self.total_reward-5)]
        
        res_img = self.img
        state = []
        for i in range(2):
            try:
                s1 = np.mean(self.img[self.sub_x-2*i][self.sub_y])/255.0
            except:
                s1 = 1
            try:
                s2 = np.mean(self.img[self.sub_x+2*i][self.sub_y])/255.0
            except:
                s2 = 1
            try:
                s3 = np.mean(self.img[self.sub_x][self.sub_y-2*i])/255.0
            except:
                s3 = 1
            try:
                s4 = np.mean(self.img[self.sub_x][self.sub_y+2*i])/255.0
            except:
                s4 = 1
##            try:
##                s5 = np.mean(self.img[self.sub_x-2*i][self.sub_y-2*i])/255.0
##            except:
##                s5 = 1
##            try:
##                s6 = np.mean(self.img[self.sub_x+2*i][self.sub_y+2*i])/255.0
##            except:
##                s6 = 1
##            try:
##                s7 = np.mean(self.img[self.sub_x-2*i][self.sub_y+2*i])/255.0
##            except:
##                s7 = 1
##            try:
##                s8 = np.mean(self.img[self.sub_x+2*i][self.sub_y-2*i])/255.0
##            except:
##                s8 = 1
            state.append(s1)
            state.append(s2)
            state.append(s3)
            state.append(s4)
##            state.append(s5)
##            state.append(s6)
##            state.append(s7)
##            state.append(s8)
        if self.food_x - self.sub_x < 0:
            s9 = 1
        elif self.food_x - self.sub_x > 0:
            s9 = 2
        else:
            s9 = 0
        if self.food_y - self.sub_y < 0:
            s10 = 1
        elif self.food_y - self.sub_y > 0:
            s10 = 2
        else:
            s10 = 0
            
        state.append(s9)
        state.append(s10)
        state.append(math.floor(self.n/10))
        if self.reward > 0:
            self.total_reward += 1
        #self.img = self.update_frame
        self.current_dist = math.sqrt((self.sub_x - self.food_x)**2+(self.sub_y-self.food_y)**2)
##        if self.current_dist <= self.last_dist and self.reward >= 0:
##            self.reward = 10
        self.last_dist = self.current_dist

        return np.array(state), self.reward, self.terminal


