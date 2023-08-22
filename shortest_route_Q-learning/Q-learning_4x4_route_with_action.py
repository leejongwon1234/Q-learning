import numpy as np


reward_table = np.ones((4, 4, 4), dtype=int)
Q_table = np.zeros((4, 4, 4))

reward_table[0, :, 0] = 0
reward_table[-1, :, 1] = 0
reward_table[:, 0, 2] = 0
reward_table[:, -1, 3] = 0

reward_table[3, 2, 3] = 100
reward_table[2, 3, 1] = 100

gamma = 0.9
alpha = 0.1

def get_random_state():
    col = np.random.randint(0, 4)  # col: 첫 번째 인덱스
    row = np.random.randint(0, 4)  # row: 두 번째 인덱스


    while col == 3 and row == 3:
        # 반복문 처리, 될 때까지 다시 뽑기
        col = np.random.randint(0, 4)  # col: 첫 번째 인덱스
        row = np.random.randint(0, 4)  # row: 두 번째 인덱스

    # # 재귀함수로 처리
    # if col == 3 and row == 3:
    #     return get_random_state()
    # print((col, row))
    return (col, row)


def get_random_action(state):
    # Q_table[3, 0] -(인덱싱)-> [1, 0, 0, 1] -(np.where)-> [0, 3]
    # [0, 3]에서 둘중 하나를 랜덤으로 선택하면 되겠죠? np.random.???
    actions = reward_table[state]
    available_actions = np.where(actions != 0)[0]
    action = np.random.choice(available_actions)

    return action


def step(state, action):
    # action - 0~3 값
    # 0은 위로, 1은 아래로, 2는 왼쪽으로, 3은 오른쪽으로
    action_table = {0:(-1, 0), 1:(1, 0), 2:(0, -1), 3:(0, 1)}

    # state - (0~3, 0~3) 튜플
    # col = state[0] + action_table[action][0]
    # row = state[1] + action_table[action][1]
    # next_state = (col, row)

    next_state = np.array(state) + action_table[action]
    next_state = tuple(next_state)

    # reward
    reward = reward_table[state[0], state[1], action]

    return next_state, reward


def Q_table_update(state, action, reward, next_state):
    global Q_table
    # 1. TD 계산
    td = reward + gamma * np.max(Q_table[next_state]) - Q_table[state[0], state[1], action]

    # 2. Q_table 업데이트
    Q_table[state[0], state[1], action] = alpha * td


for _ in range(1000):
    # state는 2개의 정수형 자료를 가진 tuple
    state = get_random_state()

    action = get_random_action(state)

    # next_state 2개의 정수형 자료를 가진 tuple, 좌표
    # reward는 1, 100 중에 하나
    # print(state)
    next_state, reward = step(state, action)

    Q_table_update(state, action, reward, next_state)

print(Q_table)

# 인퍼런스
state = (0, 0)

while True:
    action = np.argmax(Q_table[state])
    next_state, _ = step(state, action)
    
    state = next_state
    print(state)

    if state == (3, 3):
        break




