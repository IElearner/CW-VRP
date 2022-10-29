import math
import pandas as pd
import numpy as np
import copy

def Distance(Customers):#距离矩阵，dij

    distance_matrix = pd.DataFrame(data=None, columns=range(len(Customers)), index=range(len(Customers)))
    for i in range(len(Customers)):
        xi, yi = Customers[i][0], Customers[i][1]
        for j in range(len(Customers)):
            xj, yj = Customers[j][0], Customers[j][1]
            distance_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 4)
    return distance_matrix


def linkline(point1, point2, drivingLine):#将点1、点2路径合并时寻找连接哪条线路
    for i in range(len(drivingLine)):
        if (point1 in drivingLine[i]) | (point2 in drivingLine[i]):
            return i


def linkfind(drivingline, point1, point2):
    left = drivingline[0]
    right = drivingline[-1]
    if point1 == left:
        return 0, point1, point2
    elif point2 == left:
        return 0, point2, point1
    elif point1 == right:
        return -1, point1, point2
    else:
        return -1, point2, point1#分别返回链接方向、链接点、被链接点


def print_path(drving_line): #绘制路径
    print("路径图是：")
    for line in drving_line:
        for i in line:
            print(f"{i}-",end='')
        print("\n")

def ifoverload(drving_line,demanddata,car_load):#判断线路是否超载
    for line in drving_line:
        judge=False
        load=0
        for i in line:
            load+=demanddata[i]
        if load>car_load:
            judge=True
            break
        else:
            continue
    return judge

def total_length(driving_line,dis_matrix):#计算路程总长
    total = 0
    a = len(driving_line)
    for line in range(0, a):
        b = len(driving_line[line])
        for i in range(0, b - 1):
            total += dis_matrix.iloc[driving_line[line][i], driving_line[line][i + 1]]
    return (total)

if __name__ == '__main__':
    max_load = 100  # 最大运输量
    datA = pd.read_excel("./优化算法设计作业1数据.xlsx")#和源文件放在同一目录下，数据集名字改了一下
    data1 = datA['X坐标']
    data1 = np.array(data1)
    data2 = datA['Y坐标']
    data2 = np.array(data2)
    data3 = datA['需求量']
    data3 = np.array(data3)

    CustomersCoordinates = []
    demand = []
    for i in range(0, 71):
        CustomersCoordinate = []
        CustomersCoordinate.append(data1[i])
        CustomersCoordinate.append(data2[i])
        CustomersCoordinates.append(CustomersCoordinate)
        demand.append(data3[i])

    dis_matrix = Distance(CustomersCoordinates)  # 计算客户间距离

    # 计算合并减少的里程列表，里面嵌套字典且按减少数量排序，另外还有两点坐标序号（三个键point1|point2|save_dis）
    dis_com = pd.DataFrame(data=None, columns=["point1", "point2", "save_dis"])
    for i in range(1, len(CustomersCoordinates) - 1):
        for j in range(i + 1, len(CustomersCoordinates)):
            detal = dis_matrix.iloc[0, i] + dis_matrix.iloc[0, j] - dis_matrix.iloc[i, j]
            dis_com = dis_com.append({"point1": i, "point2": j, "save_dis": detal}, ignore_index=True)
    dis_com = dis_com.sort_values(by="save_dis", ascending=False).reset_index(drop=True)

    drivingLines = [[]]  # 列表嵌套列表，一个子列表包括该路线所过点
    unfinished_point = []  # 在车辆两端的点
    finished_point = []  # 记录已完成的点
    lineDemands = [0]  # 记录车辆装载量

    for i in range(len(dis_com)): #大的先上
        if not drivingLines[-1]:  # 列表为空时
            drivingLines[0].append(int(dis_com.loc[i, 'point1']))
            drivingLines[0].append(int(dis_com.loc[i, 'point2']))
            lineDemands[0] = demand[int(dis_com.loc[i, 'point1'])] + demand[int(dis_com.loc[i, 'point2'])]
            unfinished_point.append(int(dis_com.loc[i, 'point1']))  # 全局
            unfinished_point.append(int(dis_com.loc[i, 'point2']))
            continue
        if ((int(dis_com.loc[i, 'point1']) in unfinished_point) & (int(dis_com.loc[i, 'point2']) in unfinished_point)) \
                | (int(dis_com.loc[i, 'point1']) in finished_point) | (int(dis_com.loc[i, 'point2']) in finished_point):
            continue  # 两点只接一头或一点两头都接

        elif ((int(dis_com.loc[i, 'point1']) not in unfinished_point) & (
                int(dis_com.loc[i, 'point2']) not in unfinished_point)):  # 两点都不在，新开一条线路
            drivingLines.append([int(dis_com.loc[i, 'point1']), int(dis_com.loc[i, 'point2'])])
            lineDemands.append(demand[int(dis_com.loc[i, 'point1'])] + demand[int(dis_com.loc[i, 'point2'])])
            unfinished_point.append(int(dis_com.loc[i, 'point1']))
            unfinished_point.append(int(dis_com.loc[i, 'point2']))

        else:  # 一点已装车且允许再衔接其他点，一点未装车，
            line_index = linkline(int(dis_com.loc[i, 'point1']), int(dis_com.loc[i, 'point2']), drivingLines)  # 查看在哪条线路
            link_index, link_point, point = linkfind(drivingLines[line_index], int(dis_com.loc[i, 'point1']),
                                                     int(dis_com.loc[i, 'point2']))  # 确定线路上的链接位置和链接点
            if lineDemands[line_index] + demand[point] <= max_load:
                lineDemands[line_index] += demand[point]
                if link_index == 0:#从前头链接
                    unfinished_point.remove(link_point)
                    unfinished_point.append(point)
                    finished_point.append(link_point)
                    drivingLines[line_index].insert(0, point)#线头
                else:  #从尾端链接
                    unfinished_point.remove(link_point)
                    unfinished_point.append(point)
                    finished_point.append(link_point)
                    drivingLines[line_index].append(point)#线尾
                    continue
    # 从0出发返回0
    for i in drivingLines:
        i.append(0)
        i.insert(0, 0)
    length = 0
    a = len(drivingLines)
    # 求总长

    # 画路径图,此为一个可行解
    print(f"总路程为：{total_length(drivingLines,dis_matrix)}")
    print_path(drivingLines)

    #离散搜索优化
    times=0
    better_line=[]
    for line1 in range(0, a):
        b = len(drivingLines[line1])
        for insert_point in range(1, b - 1):  # 第一大层循环，找到一个插入节点V（70个节点每个来一回）
            optlist = []  # 大列表套着优化后的新方案，每个新方案里面即每条路径+路径上每个点（三层列表）
            for line2 in range(0, a):
                if drivingLines[line1]==drivingLines[line2]:#不在原路线变换位置
                    continue
                c = len(drivingLines[line2])
                for insert_place in range(0, c - 1):    #第二大层循环，找到一个插入点
                    dlcopy = copy.deepcopy(drivingLines)
                    dlcopy[line2].insert(insert_place, drivingLines[line1][insert_point])#点V插入line2
                    del dlcopy[line1][insert_point]        #line1上删除点v
                    if not ifoverload(dlcopy,demand,max_load): #判断新方案是否超重
                        if total_length(dlcopy,dis_matrix) < total_length(drivingLines,dis_matrix):#判断新方案是否优化
                            times=times+1         #记录可行优化解数量
                            better_line=copy.deepcopy(dlcopy)

    print(times) #几个可行且更优解
    #上述程序运行后，知晓存在一个更优解（times=1）即better_line，因此再对其搜索
    times1=0
    for line1 in range(0, a):
        b = len(better_line[line1])
        for insert_point in range(1, b - 1):  # 第一大层循环，找到一个插入节点V（70个节点每个来一回）
            optlist = []  # 大列表套着优化后的新方案，每个新方案里面即每条路径+路径上每个点（三层列表）
            for line2 in range(0, a):
                if better_line[line1] == better_line[line2]:  # 不在原路线变换位置
                    continue
                c = len(better_line[line2])
                for insert_place in range(0, c - 1):  # 第二大层循环，找到一个插入点
                    dlcopy1 = copy.deepcopy(better_line)
                    dlcopy1[line2].insert(insert_place, better_line[line1][insert_point])  # 点V插入line2
                    del dlcopy1[line1][insert_point]  # line1上删除点v
                    if not ifoverload(dlcopy1, demand, max_load):  # 判断新方案是否超重
                        if total_length(dlcopy1, dis_matrix) < total_length(better_line, dis_matrix):  # 判断新方案是否优化
                            times1 = times1 + 1  # 记录可行优化解数量
                            print_path(dlcopy1)
                            better_line1 = copy.deepcopy(dlcopy1)
                            print(total_length(dlcopy1, dis_matrix))

    print(times1)#运行后times1=0,没有更优解，因此betterline即最优解
    print(f"最优路径长度为{total_length(better_line,dis_matrix)}")
    print_path(better_line)




