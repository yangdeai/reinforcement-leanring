import numpy as np
import tensorflow as tf

from keras.layers import Conv3D, MaxPooling3D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DQN:
    def __init__(
            self,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=2000,
            batch_size=64,
            e_greedy_increment=None,
    ):
        self.n_actions = 6
        self.hi = 8
        self.le = 8
        self.we = 8         # 场景的 长宽高
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter

        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        # 经验池
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memorys = np.zeros([self.memory_size, self.we, self.le, self.hi])
        self.memoryt = np.zeros([self.memory_size, self.we, self.le, self.hi])
        self.memorya = np.zeros([self.memory_size])
        self.memoryr = np.zeros([self.memory_size])

        self.eval = self.initCNN()     # 创建eval net
        self.target = self.initCNN()     # 创建target_net
        # self.eval.load_weights('eval_net.h5')
        # self.target.load_weights('target_net.h5')


    def initCNN(self):
        model = Sequential()
        model.add(Conv3D(128, [5, 5, 5], activation='relu',
                         input_shape=[self.we, self.le, self.hi, 1], data_format="channels_last"))
        # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Conv3D(64, [3, 3, 3], activation='relu', data_format="channels_last"))
        # model.add(MaxPooling3D(pool_size=(1, 1, 1)))
        model.add(Conv3D(32, [1, 1, 1], activation='relu', data_format="channels_last"))
        model.add(Conv3D(32, [1, 1, 1], activation='relu', data_format="channels_last"))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_actions))
        opt = Adam(lr=self.lr)
        # opt = SGD(lr=self.lr)
        model.compile(optimizer=opt, loss='mean_squared_error')
        return model

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # replace the old memory with new memory
        # print(np.shape(s))
        index = self.memory_counter % self.memory_size
        self.memorys[index, :, :, :] = s
        self.memoryt[index, :, :, :] = s_
        self.memoryr[index] = r
        self.memorya[index] = a
        self.memory_counter += 1

    def train(self):

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memorys = np.expand_dims(self.memorys[sample_index, :, :, :], axis=4)
        batch_memoryt = np.expand_dims(self.memoryt[sample_index, :, :, :], axis=4)
        batch_memoryr = self.memoryr[sample_index]
        batch_memorya = [int(a) for a in self.memorya[sample_index]]
        q_eval = self.eval.predict(batch_memorys)
        q_target = self.target.predict(batch_memoryt)
        target_r = q_eval.copy()
        for time in range(self.batch_size):
            temppp = batch_memoryr[time] + self.gamma * np.max(q_target[time, :])
            target_r[time, batch_memorya[time]] = temppp
        self.eval.fit(batch_memorys, target_r, batch_size=64, epochs=1, verbose=2)
        # q_e = self.eval.predict(batch_memorys)
        # place1 = np.where(batch_memorys[1, :, :, :, :] == 1)
        # place2 = np.where(batch_memoryt[1, :, :, :, :] == 1)
        # x1, y1, z1 = place1[0], place1[1], place1[2]
        # x2, y2, z2 = place2[0], place2[1], place2[2]
        # self.draw(x1, y1, z1, x2, y2, z2)
        self.learn_step_counter += 1
    def target_replace_op(self):
        v1 = self.eval.get_weights()
        self.target.set_weights(v1)
        print("params has changed")
    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            ss = np.array(s)
            S1 = np.expand_dims(ss, axis=0)
            S2 = np.expand_dims(S1, axis=4)
            actions_value = self.eval.predict(S2, batch_size=1)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    def save_weight(self):
        self.eval.save_weights('eval_net.h5')
        self.target.save_weights('target_net.h5')



    def draw(self,datex,datey,datez,datex1,datey2,datez3):
        # plt.ion()
        # plt.clf()
        fig = plt.figure(1)
        ax = Axes3D(fig)
        # datex = np.array(state)[:, 0]
        # datey = np.array(state)[:, 1]
        # datez = np.array(state)[:, 2]
        ax.scatter(datex, datey, datez, color='red', marker='*')
        # ax.scatter(datex[0], datey[0], datez[0], color='green')
        ax.scatter(datex1, datey2, datez3, color='black', marker='1')
        # ax.scatter(datex1[0], datey2[0], datez3[0], color='red')
        # ax.scatter(aim[0][0], aim[0][1], aim[0][2], color='red', marker='*')
        # ax.scatter(aim[1][0], aim[1][1], aim[1][2], color='red', marker='1')
        # ax.scatter(aim[2][0], aim[2][1], aim[2][2], color='red', marker='>')
        ## 障碍物1
        x = 2
        y = 2
        dx = 2
        dy = 2
        z = 0
        dz = 2
        xx = [x, x, x + dx, x + dx, x]
        yy = [y, y + dy, y + dy, y, y]
        kwargs = {'alpha': 1, 'color': 'grey'}
        ax.plot3D(xx, yy, [z] * 5, **kwargs)
        ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)
        ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
        ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
        ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
        ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)
        # ## 障碍物2
        # x = 6
        # y = 6
        # dx = 2
        # dy = 2
        # z = 4
        # dz = 2
        # xx = [x, x, x + dx, x + dx, x]
        # yy = [y, y + dy, y + dy, y, y]
        # kwargs = {'alpha': 1, 'color': 'grey'}
        # ax.plot3D(xx, yy, [z] * 5, **kwargs)
        # ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)
        # ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
        # ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
        # ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
        # ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)
        # ax.text(aim[0][0], aim[0][1], aim[0][2], 'aim' + str(aim[0]))
        # ax.text(aim[1][0], aim[1][1], aim[1][2], 'aim' + str(aim[1]))
        # ax.text(aim[2][0], aim[2][1], aim[2][2], 'aim' + str(aim[2]))
        # ax.text(datex[0], datey[0], datez[0], str([datex[0], datey[0], datez[0]]))
        ax.set_xbound(0, 8)
        ax.set_ybound(0, 8)
        ax.set_zbound(0, 8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
