from env import game_snake as snake
from DQN_keras import DQN
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def run_snake():
    step = 0
    # plt.close()

    for episode in range(10000):
        # initial observation
        print('episode='+str(episode))
        observation, score = env.reset()
        record = 0
        reaction = []
        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # print(action)
            # RL take action and get next observation and reward
            observation_, reward, done, state, aim, direction, score = env.step(action)
            # draw(state, aim)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 2000) and (step % 10 == 0):
                RL.train()
            # swap observation
            observation = observation_[:]
            record += reward
            reaction.append(action)
            # break while loop when end of this episode
            if done:
                break

            step += 1
        print('the action = ', reaction)
        print('the score = ', str(score), 'the reward = ', str(record))



def draw(state, aim):
    plt.ion()
    plt.clf()
    fig = plt.figure(1)
    ax = Axes3D(fig)
    datex = np.array(state)[:, 0]
    datey = np.array(state)[:, 1]
    datez = np.array(state)[:, 2]
    ax.plot(datex, datey, datez, color='green')
    ax.scatter(datex[0], datey[0], datez[0], color='black')
    ax.scatter(aim[0][0], aim[0][1], aim[0][2], color='red', marker='*')
    ax.scatter(aim[1][0], aim[1][1], aim[1][2], color='red', marker='1')
    ax.scatter(aim[2][0], aim[2][1], aim[2][2], color='red', marker='>')
## 障碍物1
    x=2
    y=2
    dx=2
    dy=2
    z=0
    dz=2
    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]
    kwargs = {'alpha': 1, 'color': 'grey'}
    ax.plot3D(xx, yy, [z]*5, **kwargs)
    ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
# ## 障碍物2
#     x=6
#     y=6
#     dx=2
#     dy=2
#     z=4
#     dz=2
#     xx = [x, x, x+dx, x+dx, x]
#     yy = [y, y+dy, y+dy, y, y]
#     kwargs = {'alpha': 1, 'color': 'grey'}
#     ax.plot3D(xx, yy, [z]*5, **kwargs)
#     ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
#     ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
#     ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
#     ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
#     ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
    ax.text(aim[0][0], aim[0][1], aim[0][2], 'aim'+str(aim[0]))
    ax.text(aim[1][0], aim[1][1], aim[1][2], 'aim' + str(aim[1]))
    ax.text(aim[2][0], aim[2][1], aim[2][2], 'aim' + str(aim[2]))
    ax.text(datex[0], datey[0], datez[0], str([datex[0], datey[0], datez[0]]))
    ax.set_xbound(0, 8)
    ax.set_ybound(0, 8)
    ax.set_zbound(0, 8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.pause(0.0001)

if __name__ == "__main__":
    # maze game
    env = snake()
    RL = DQN(learning_rate=0.00001, reward_decay=0.9, e_greedy=0.9,
             replace_target_iter=1000, memory_size=2000, batch_size=128, e_greedy_increment=None)
    for time in range(0, 1000):
        run_snake()
        RL.save_weight()


