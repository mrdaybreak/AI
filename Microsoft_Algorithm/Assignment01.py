import requests
import re
from collections import defaultdict
import copy
r = requests.get('http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json').text
# print(r)

def get_subway(first, second, dict_name):
    subway_name = re.findall('"{}":"(.*?)"|"{}":"(.*?)"'.format(first, second), r)
    dict_list = []
    for i in subway_name:
        if i[1] != '':
            dict_list.append(i[1])
        if i[0] != '':
            dict_name[i[0]] = dict_list
            dict_list = []
    print(dict_name)
dict_name = {}
get_subway("ln", "n", dict_name)

stations = re.findall('"n":"(.*?)"', r)
positions_origin = re.findall('"p":"(.*?)"', r)
# print(positions_origin)
positions = [i.split() for i in positions_origin]
positions = [[int(a[0]), int(a[1])] for a in positions]
# print(positions)
c = dict(zip(stations, positions))
# print(dict(c))

stations_dict = defaultdict(list)
for i in dict_name:
    stations_list = dict_name.get(i, [])
    for i in stations_list:
        st_index = stations_list.index(i)
        if st_index > 1:
            if stations_list[st_index - 1] not in stations_dict[i]:
                stations_dict[i].append(stations_list[st_index - 1])
        if st_index < len(stations_list) - 1:
            if stations_list[st_index + 1] not in stations_dict[i]:
                stations_dict[i].append(stations_list[st_index + 1])
# print(stations_dict)
result = []

def search_way(start, end, path):
    if start == end:
        path_cache = copy.deepcopy(path)
        print('aa',path_cache)
        result.append(path_cache)
        return True
    if not stations_dict.get(start):
        return False

    for item in stations_dict[start]:
        print(item)
        if item in path:
            continue
        if item not in path:
            path.append(item)
            if search_way(item, end, path):
                return True
        path.pop()
    return False

search_way("霍营", '立水桥', path=[''])
# print(result)

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
nodes = []
edges = []
for i in stations_dict.items():
    nodes.append(i[0])
    for j in i[1]:
        edges.append([i[0], j])
graph = nx.Graph(edges)
nlabels = dict(zip(nodes, nodes))
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
nx.draw_networkx_nodes(graph, c, node_size=30, node_color="#6CB6FF")  # 绘制节点
nx.draw_networkx_edges(graph, c, edges)  # 绘制边
nx.draw_networkx_labels(graph, c, nlabels, font_size=6)
plt.show()
