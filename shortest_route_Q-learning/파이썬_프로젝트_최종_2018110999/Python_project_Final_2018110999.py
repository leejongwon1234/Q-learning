import numpy as np

#맵이름을 입력받는 함수
def get_in_map():
    while True:
        try:
            map_name = input('맵 이름을 입력하세요 : ').strip()
            f = open(map_name, 'r')
            f.close()
            return map_name
        except :
            print('파일이 존재하지 않습니다. 다시 입력하세요.')

#맵을 받아서 1차원 배열로 만들어주는 함수
def map_to_1darray(map_name):
    maps = []
    f = open(map_name, 'r')
    for line in f:
        maps.append(line.strip())
    f.close()
    return maps

#1차원 맵을 2차원 배열로 만들어주는 함수
#시작점, 도착점, 중간점을 찾아서 반환해주는 함수
#시작점, 도착점, 중간점은 각각 2, 3, 4로 표시 / numpy 배열로 반환
def read_maps(maps):
    s_location = ''
    e_location = ''
    a_location=''
    maps_r=np.zeros([len(maps),len(maps[0])])
    for i in range(len(maps)):
        for j in range(len(maps[0])):
            if maps[i][j] == 'S':
                maps_r[i][j] = 2
                s_location = (i*len(maps[0])+j)
            elif maps[i][j] == 'O':
                maps_r[i][j] = 1
            elif maps[i][j] == 'E':
                maps_r[i][j] = 3
                e_location = (i*len(maps[0])+j)
            elif maps[i][j] == 'A':
                maps_r[i][j] = 4
                a_location= (i*len(maps[0])+j)
    return maps_r, s_location, e_location, a_location

#2차원 maps_r을 0으로 패딩하는 함수
def padding(maps_r):
    maps_r_p = np.zeros([len(maps_r)+2,len(maps_r[0])+2])
    for i in range(1,len(maps_r)+1):
        for j in range(1,len(maps_r[0])+1):
            maps_r_p[i][j] = maps_r[i-1][j-1]
    return maps_r_p

#특정 맵에 대해 움직일 수 있는 R-matrix를 만들어주는 함수
def make_R(maps_r,maps_r_p):
    map_size = len(maps_r) * len(maps_r[0])
    R = np.zeros([map_size,map_size])
    count=0
    for i in range(1,len(maps_r)+1):
        for j in range(1,len(maps_r[0])+1):
            if maps_r_p[i][j] != 0:
                if maps_r_p[i-1][j] != 0:
                    R[count][count-len(maps_r[0])] = 1
                if maps_r_p[i][j+1] !=0:
                    R[count][count+1] = 1
                if maps_r_p[i][j-1] !=0:
                    R[count][count-1] = 1
                if maps_r_p[i+1][j] !=0:
                    R[count][count+len(maps_r[0])] = 1
            count+=1
    return R

#R-matrix를 이용하여 Q-matrix를 만들어 반환하는 함수
def make_Q(R):
    Q = np.zeros([len(R),len(R)])
    learning_num = 3000
    learning_gamma = 0.99
    learning_alpha = 0.7
    for i in range(learning_num):
        current_state = np.random.randint(0,len(R))
        playable_actions = []
        for j in range(len(R)):
            if R[current_state,j] > 0:
                playable_actions.append(j)
        if len(playable_actions)==0:
            pass
        else:
            next_state = np.random.choice(playable_actions)
            TD = R[current_state,next_state] + learning_gamma*Q[next_state,np.argmax(Q[next_state,])] - Q[current_state,next_state]
            Q[current_state,next_state] += learning_alpha*TD
    return Q

#최단루트를 찾아주는 함수
def find_route(Q,s_location,e_location):
    route = [s_location]
    next_location = s_location
    while (next_location != e_location):
        next_location = np.argmax(Q[s_location,])
        route.append(next_location)
        s_location = next_location

    return route


#class map
class Map:
    def __init__(self, maps):
        self.maps = maps #1차원 맵
        self.maps_r, self.s_location, self.e_location, self.a_location = read_maps(maps) #2차원 맵, 시작점, 도착점, 중간점
        self.maps_r_p = padding(self.maps_r) #2차원 맵을 0으로 패딩한 맵
        self.R_1 = make_R(self.maps_r, self.maps_r_p) #특정 맵에 대해 움직일 수 있는 R-matrix
        self.R_2 = np.copy(self.R_1) #R-matrix를 복사한 R-matrix
        self.Q_1 = np.array([])
        self.Q_2 = np.array([])
        self.path_1 = []
        self.path_2 = []

    #1차원 맵을 2차원 맵으로 만들어서 출력하는 함수
    def print_map(self):
        print('Map: (시작점 : S, A : A, 도착점 : E, 벽 : X, 이동 가능 : O)')
        for i in range(len(self.maps)):
            for j in range(len(self.maps[0])):
                print(f'{self.maps[i][j]:>3s}', end = ' ')
            print()

    #R-matrix를 R_new로 업데이트하는 함수
    def update_R(self):
        self.R_1[self.a_location,self.a_location] = 1000
        self.R_2[self.e_location,self.e_location] = 1000

    #Q-matrix를 만들어주는 함수
    def make_Q(self):
        self.Q_1 = make_Q(self.R_1)
        self.Q_2 = make_Q(self.R_2)
    
    #최단루트를 찾아주는 함수
    def find_route(self):
        self.path_1 = find_route(self.Q_1,self.s_location,self.a_location)
        self.path_2 = find_route(self.Q_2,self.a_location,self.e_location)

    #최단거리를 출력하는 함수
    def print_path(self):
        print(f'\n최단거리: {len(self.path_1)+len(self.path_2)-2}')

    #시작점, 도착점, 중간점의 좌표를 출력하는 함수
    def print_location(self):
        start_point = np.where(self.maps_r == 2)
        end_point = np.where(self.maps_r == 3)
        a_point = np.where(self.maps_r == 4)
        start_point_p = tuple([start_point[0][0], start_point[1][0]])
        end_point_p = tuple([end_point[0][0], end_point[1][0]])
        a_point_p = tuple([a_point[0][0], a_point[1][0]])
        print(f'\n시작점 좌표: {start_point_p} : {self.s_location}')
        print(f'A위치 좌표: {a_point_p} : {self.a_location}')
        print(f'도착점 좌표: {end_point_p} : {self.e_location} ')
    
    #위치를 표시한 맵을 출력하는 함수
    def print_map_with_location(self):
        print('\nlocation Map:')
        count = 0
        for i in range(len(self.maps)):
            for j in range(len(self.maps[0])):
                print(f'{count:>3d}', end = ' ')
                count+=1
            print()

    #가는 방법을 출력하는 함수 : (오른쪽 : R, 왼쪽 : L, 위쪽 : U, 아래쪽 : D)
    def print_route(self):
        print('\n가는 방법: (오른쪽 : R, 왼쪽 : L, 위쪽 : U, 아래쪽 : D)')
        print('시작점에서 A까지: ', end = ' ')
        for i in range(len(self.path_1)-1):
            if self.path_1[i+1] - self.path_1[i] == 1:
                print('R', end = ' ')
            elif self.path_1[i+1] - self.path_1[i] == -1:
                print('L', end = ' ')
            elif self.path_1[i+1] - self.path_1[i] == len(self.maps[0]):
                print('D', end = ' ')
            elif self.path_1[i+1] - self.path_1[i] == -len(self.maps[0]):
                print('U', end = ' ')
        print()
        print('A에서 도착점까지: ', end = ' ')
        for i in range(len(self.path_2)-1):
            if self.path_2[i+1] - self.path_2[i] == 1:
                print('R', end = ' ')
            elif self.path_2[i+1] - self.path_2[i] == -1:
                print('L', end = ' ')
            elif self.path_2[i+1] - self.path_2[i] == len(self.maps[0]):
                print('D', end = ' ')
            elif self.path_2[i+1] - self.path_2[i] == -len(self.maps[0]):
                print('U', end = ' ')
        print()

    #최단 루트를 출력하는 함수
    def print_shortest_route(self):
            print('\n시작점에서 A까지 최단루트 : ', self.path_1)
            print('A에서 도착점까지 최단루트 : ', self.path_2)



#main
while True:
    try:
        map_num = int(input("몇 개의 맵을 선택하시겠습니까? : ").strip())
        if map_num >= 1 and map_num <= 10:
            maps = dict()
            for i in range(map_num):
                map_name = get_in_map()
                map = map_to_1darray(map_name)
                maps[map_name] = map
            map = list(maps.values())
            map_names = list(maps.keys())
            map_object = []
            for i in range(map_num):
                map_object.append(Map(map[i]))
                map_object[-1].update_R()
                map_object[-1].make_Q()
                map_object[-1].find_route()
                print('---------------------------------------------------------------------')
                print(f'\n{i+1}번째 맵 {map_names[i]}')
                map_object[-1].print_map()
                map_object[-1].print_map_with_location()
                map_object[-1].print_location()
                map_object[-1].print_path()
                map_object[-1].print_shortest_route()
                map_object[-1].print_route()
                print()
                print('---------------------------------------------------------------------')
            break
        else:
            print("1~10 사이의 숫자를 입력해주세요.")
    except:
        print("잘못된 입력입니다.")
            



