import json
import heapq
from urllib import request

# Server details
server_ip = "10.216.241.109"
server = f"http://{server_ip}:5000"
authKey = "my_auth_key"

# Map nodes with coordinates
nodes = {
    "C2": (452, 29), #Beak_St_Aquatic
    "B2": (305, 29), #Feather_St_Aquatic
    "A2": (129, 29), #Waddle_Way_Aquatic
    "Z": (213, 29), #Waterfoul_Way_Aquatic
    "B": (284, 393), #The_Circle_Breadcrumb
    "A": (181, 459), #Waddle_Way_Breadcrumb
    "J": (305, 296), #Feather_St_Circle
    "I": (273, 307), #Waterfoul_Way_Circle
    "P": (452, 293), #Beak_St_Dabbler
    "L": (350, 324), #The_Circle_Dabbler
    "Q": (585, 293), #Mallard_St_Dabbler
    "M": (452, 402), #Beak_St_Drake
    "N": (576, 354), #Mallard_St_Drake
    "D": (452, 474), #Beak_St_Duckling
    "O": (593, 354), #Mallard_St_Duckling
    "X": (452, 135), #Beak_St_Migration
    "W": (305, 135), #Feather_St_Migration
    "Y": (585, 135), #Mallard_St_Migration
    "T": (29, 135), #Quack_St_Migration
    "U": (129, 135), #Waddle_Way_Migration
    "V": (213, 135), #Waterfoul_Way_Migration
    "R": (452, 233), #Beak_St_Pondside
    "K": (305, 233), #Feather_St_Pondside
    "S": (585, 233), #Mallard_St_Pondside
    "F": (28, 329), #Quack_St_Pondside
    "H": (214, 241), #Waterfoul_Way_Pondside
    "G": (157, 266), #Waddle_Way_Pondside
    "E": (452, 465), #Beak_St_Tail
    "C": (335, 387) #The_Circle_Tail
}

# Adjacency list for connected nodes (bidirectional roads)
graph = {
    "A": ["B", "F", "G"],
    "B": ["A", "I"],
    "C": ["B", "E"],
    "D": ["C", "M"],
    "E": ["C"],
    "F": ["A"],
    "G": ["F", "H", "U"],
}


def distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def find_closest_node(x, y):
    return min(nodes.keys(), key=lambda node: distance((x, y), nodes[node]))

def dijkstra(start, goal):
    queue = [(0, start)]
    distances = {node: float('inf') for node in nodes}
    distances[start] = 0
    previous = {node: None for node in nodes}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == goal:
            path = []
            while current_node:
                path.append(current_node)
                current_node = previous[current_node]
            return path[::-1]

        for neighbor in graph.get(current_node, []):
            new_distance = current_distance + distance(nodes[current_node], nodes[neighbor])
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (new_distance, neighbor))

    return []

def choose_fare():
    res = request.urlopen(server + "/fares")
    fares = json.loads(res.read())
    best_fare = max(fares, key=lambda fare: fare['pay'])
    print("Selected Fare:", best_fare)
    return best_fare

def shortest_path(src, dest):
    src_node = find_closest_node(src['x'], src['y'])
    dest_node = find_closest_node(dest['x'], dest['y'])
    return dijkstra(src_node, dest_node)

fare = choose_fare()
print("Shortest Path:", shortest_path(fare['src'], fare['dest']))
