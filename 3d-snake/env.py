"""
snake  env
"""
import numpy as np
import operator
class game_snake(object):
    def __init__(self):
        super(game_snake, self).__init__()
        self.hi = 8
        self.le = 8
        self.we = 8         # 场景的 长宽高
        self.snakelength = 5
        self.state = np.zeros([3, 5], dtype=np.int)
        self.observation = np.zeros([self.we, self.le, self.hi])
        self.scenario = 10 * np.ones([self.we, self.le, self.hi])
        for i in range(2,5):
            for j in range(2,5):
                for k in range(0,3):
                    self.scenario[i,j,k] = -10


        self.direction = self.find_dir()
        self.aimnum = 3
        self.score = 100*np.random.randint(0, 5, self.aimnum)
        self.getscore = 0
    # 初始化
    def reset(self):
        self.snakelength = 5
        self.state = [[5, 5, 5], [4, 5, 5], [3, 5, 5], [2, 5, 5], [1, 5, 5]]
        self.aim = []
        for ii in range(self.aimnum):
            self.aim.append(self.gene_aim())
        self.direction = self.find_dir()
        self.observation = self.scenario.copy()
        for i in range(self.snakelength):
            self.observation[self.state[i][0]-1, self.state[i][1]-1, self.state[i][2]-1] = 20
        for i in range(self.aimnum):
            self.observation[self.aim[i][0]-1, self.aim[i][1]-1, self.aim[i][2]-1] = 100
        self.getscore = 0
        return self.observation, self.score

    def step(self, action):
        # print(self.state)
        self.direction = self.find_dir()
        done = 0
        reward = 0
        # 判断下一步动作是不是在 运动范围之内
        head = self.state[0][:]
        if action == 5: # z轴正方向
            temph = head[2] + 1
            # 判断是否在运动范围内
            if temph >= 0 and temph <= self.hi and self.direction != 5:
                if [head[0],  head[2], temph] not in self.state:
                    head[2] = head[2] + 1
                    reward = self.do_action(head)
                else:
                    done = 1
                    reward = -100
            else:
                done = 1
                reward = -100
        if action == 4: # z轴负方向
            temph = head[2] - 1
            # 判断是否在运动范围内
            if temph >= 0 and temph <= self.hi and self.direction != 4:
                if [head[0],  head[2], temph] not in self.state:
                    head[2] = head[2] - 1
                    reward = self.do_action(head)
                else:
                    done = 1
                    reward = -100
            else:
                done = 1
                reward = -100
        if action == 3: # y轴正方向
            templ = head[1] + 1
            # 判断是否在运动范围内
            if templ >= 0 and templ <= self.le and self.direction != 3:
                if [head[0], templ, head[2]] not in self.state:
                    head[1] = head[1] + 1
                    reward = self.do_action(head)
                else:
                    done = 1
                    reward = -100
            else:
                done = 1
                reward = -100
        if action == 2: # y轴负方向
            templ = head[1] - 1
            # 判断是否在运动范围内
            if templ >= 0 and templ <= self.le and self.direction != 2:
                if [head[0], templ, head[2]] not in self.state:
                    head[1] = head[1] - 1
                    reward = self.do_action(head)
                else:
                    done = 1
                    reward = -100
            else:
                done = 1
                reward = -100
        if action == 1: # x轴正方向
            tempw = head[0] + 1
            # 判断是否在运动范围内
            if tempw >= 0 and tempw <= self.we and self.direction != 1:
                if [tempw, head[1], head[2]] not in self.state:
                    head[0] = head[0] + 1
                    reward = self.do_action(head)
                else:
                    done = 1
                    reward = -100
            else:
                done = 1
                reward = -100
        if action == 0: # x轴负方向
            tempw = head[0] - 1
            # 判断是否在运动范围内

            if tempw >= 0 and tempw <= self.we and self.direction != 0:
                if [tempw, head[1], head[2]] not in self.state:
                    head[0] = head[0] - 1
                    reward = self.do_action(head)
                else:
                    done = 1
                    reward = -100
            else:
                done = 1
                reward = -100

        if head[2] >=0 and head[2]<=2:
            if head[1] >=2 and head[1] <= 4:
                if head[0] >= 2 and head[0]<= 4:
                    reward = -100
                    done = 1
        #  障碍物
        if head[2]<=6 and head[2] >= 4:
            if head[1]<=8 and head[1]>=6:
                if head[0]<=8 and head[0]>=6:
                    reward = -100
                    done = 1
        # print(self.state)
        self.observation = self.scenario.copy()
        for i in range(self.snakelength):
            self.observation[self.state[i][0]-1, self.state[i][1]-1, self.state[i][2]-1] = 20
        for i in range(self.aimnum):
            self.observation[self.aim[i][0]-1, self.aim[i][1]-1, self.aim[i][2]-1] = 100
        return self.observation, reward, done, self.state, self.aim, self.direction, self.getscore


    def find_dir(self):
        direction = 0
        da = self.state[0]
        db = self.state[1]
        delta = [da[i] - db[i] for i in range(len(da))]
        if delta == [1, 0, 0]:
            direction = 0  # 方向向x轴正方向
        elif delta == [-1, 0, 0]:
            direction = 1  # 方向向x轴负方向
        elif delta == [0, 1, 0]:
            direction = 2  # 方向向y轴正方向
        elif delta == [0, -1, 0]:
            direction = 3  # 方向向y轴负方向
        elif delta == [0, 0, 1]:
            direction = 4  # 方向向z轴正方向
        elif delta == [0, 0, -1]:
            direction = 5  # 方向向z轴负方向
        return direction

    def do_action(self, head):
        temp = self.state[:]
        temp.insert(0, head)
        flag = 0
        maxs = np.argmax(self.score)
        maxaim = self.aim[int(maxs)][:]
        for time in range(self.aimnum):
            tempaim = self.aim[time][:]
            if head is tempaim:
                reward = 100
                self.aim[time] = self.gene_aim()
                self.snakelength = self.snakelength + 1
                self.getscore += self.score[time]
                flag = 1
                break
        if flag == 0:
            newhead = np.array(head)
            reward = -3*np.sum(np.abs(newhead - maxaim))
            # reward = 0
            del(temp[self.snakelength])
        self.state = temp[:]
        return reward

    def gene_aim(self):
        aim = np.random.randint(0, 8, 3)
        flag = 1
        while flag == 1:
            if 0 <= aim[2] <= 2:
                if 2 <= aim[1] <= 4:
                    if 2 <= aim[0] <= 4:
                        flag = 0
        #  障碍物
            if 4 <= aim[2] <= 6:
                if 6 <= aim[1] <= 8:
                    if 6 <= aim[0] <= 8:
                        flag = 0
            if flag == 0:
                aim = np.random.randint(0, 8, 3)
                flag = 1
            else:
                break
        return list(aim)

if __name__ == '__main__':
    env = game_snake()
    env.reset()