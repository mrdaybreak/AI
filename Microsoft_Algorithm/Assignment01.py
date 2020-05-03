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
dict_name = {}
get_subway("ln", "n", dict_name)
# print(dict_name)
stations = re.findall('"n":"(.*?)"', r)
positions_origin = re.findall('"p":"(.*?)"', r)
# print(positions_origin)
positions = [i.split() for i in positions_origin]
positions = [[int(a[0]), int(a[1])] for a in positions]
# print(positions)
stations_info = dict(zip(stations, positions))

neighbor_info = defaultdict(list)
for i in dict_name:
    stations_list = dict_name.get(i, [])
    for i in stations_list:
        st_index = stations_list.index(i)
        if st_index > 0:
            if stations_list[st_index - 1] not in neighbor_info[i]:
                neighbor_info[i].append(stations_list[st_index - 1])
        if st_index < len(stations_list) - 1:
            if stations_list[st_index + 1] not in neighbor_info[i]:
                neighbor_info[i].append(stations_list[st_index + 1])
# print(stations_dict)

def search_way_BFS(start, end, path, result=[]):
    if start == end:
        path_cache = copy.deepcopy(path)
        result.append(path_cache)
        # print(result)
        return
    if not neighbor_info.get(start):
        return None

    for item in neighbor_info[start]:
        # print(item)
        if item in path:
            continue
        path.append(item)
        if len(result) < 1:
            # print(result)
            search_way_BFS(item, end, path, result)
        path.pop()
    return result

# print(search_way_BFS("霍营", '立水桥', path=['霍营']))

def search_way_DFS(start, end, path, result=[]):
    if start == end:
        path_cache = copy.deepcopy(path)
        result.append(path_cache)
        # print(result)
        return
    if not neighbor_info.get(start):
        return None

    for item in reversed(neighbor_info[start]):
        print(item)
        if item in path:
            continue
        path.append(item)
        if len(result) < 1:
            print(result)
            search_way_DFS(item, end, path, result)
        path.pop()
    return result

print(search_way_DFS("沙河", '昌平', path=['沙河']))


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(figsize=(20, 20))
# stations_graph = nx.Graph()
# stations_graph.add_nodes_from(list(stations_info.keys()))
stations_connection_graph = nx.Graph(neighbor_info)
plt.figure(figsize=(30, 30))
nx.draw(stations_connection_graph, stations_info, with_labels=True, node_size=10)
# plt.show()
