import json
import heapq
from urllib import request
import math

# Server details
server_ip = "10.216.241.109"
server = f"http://{server_ip}:5000"
authKey = "41"

# Map nodes with coordinates
nodes = {
    "C2": (452, 29),  # Beak_St_Aquatic
    "B2": (305, 29),  #Feather_St_Aquatic
    "Z": (129, 29),  # Waddle_Way_Aquatic
    "A2": (213, 29),  # Waterfoul_Way_Aquatic
    "B": (284, 393),  # The_Circle_Breadcrumb
    "A": (181, 459),  # Waddle_Way_Breadcrumb
    "J": (305, 296),  # Feather_St_Circle
    "I": (273, 307),  # Waterfoul_Way_Circle
    "P": (452, 293),  # Beak_St_Dabbler
    "L": (350, 324),  # The_Circle_Dabbler
    "Q": (585, 293),  # Mallard_St_Dabbler
    "M": (452, 402),  # Beak_St_Drake
    "N": (576, 354),  # Mallard_St_Drake
    "D": (452, 474),  # Beak_St_Duckling
    "O": (593, 354),  # Mallard_St_Duckling
    "X": (452, 135),  # Beak_St_Migration
    "W": (305, 135),  # Feather_St_Migration
    "Y": (585, 135),  # Mallard_St_Migration
    "T": (29, 135),  # Quack_St_Migration
    "U": (129, 135),  # Waddle_Way_Migration
    "V": (213, 135),  # Waterfoul_Way_Migration
    "R": (452, 233),  # Beak_St_Pondside
    "K": (305, 233),  # Feather_St_Pondside
    "S": (585, 233),  # Mallard_St_Pondside
    "F": (28, 329),  # Quack_St_Pondside
    "H": (214, 241),  # Waterfoul_Way_Pondside
    "G": (157, 266),  # Waddle_Way_Pondside
    "E": (452, 465),  # Beak_St_Tail
    "C": (335, 387)  # The_Circle_Tail
}

# Adjacency list for connected nodes with weights
graph = {
    # Node to Node Connection Weights
        # TYPE 1: Average straight roads: 5
        # TYPE 2: Roads with slight curves: 6
        # TYPE 3: Road with large curves: 8
        # TYPE 4: Round-a-bout: 9
        # TYPE 5: Pedestrian zones: 7
        # TYPE 6: Risk zones: 10

    "A": [("B", 6), ("F", 9), ("G", 7)],
    "B": [("A", 6), ("I", 9)],
    "C": [("B", 9), ("E", 6)],
    "D": [("C", 6), ("M", 5)],
    "E": [("C", 5)],
    "F": [("A", 9), ("T", 6), ("G", 7)],
    "G": [("F", 7), ("H", 5), ("U", 5)],
    "H": [("I", 6), ("K", 5), ("G", 5)],
    "I": [("J", 7)],
    "J": [("K", 4), ("L", 9)],
    "K": [("J", 4), ("H", 5), ("W", 5), ("R", 7)],
    "L": [("C", 9)],
    "M": [("E", 4), ("P", 5), ("N", 6)],
    "N": [("Q", 4), ("O", 5)],
    "O": [("D", 8), ("N", 5)],
    "P": [("M", 5), ("L", 5), ("R", 4)],
    "Q": [("P", 5), ("O", 4), ("S", 4)],
    "R": [("P", 4), ("K", 7), ("X", 5), ("S", 5)],
    "S": [("Q", 4), ("R", 5), ("Y", 5)],
    "T": [("Z", 8), ("U", 10), ("F", 6)],
    "U": [("T", 10), ("Z", 5), ("V", 10)],
    "V": [("H", 5), ("U", 10), ("W", 10)],
    "W": [("K", 5), ("V", 10), ("B2", 5), ("X", 5)],
    "X": [("R", 5), ("W", 5), ("C2", 5), ("Y", 5)],
    "Y": [("S", 5), ("X", 5), ("C2", 8)],
    "Z": [("T", 8), ("A2", 5)],
    "A2": [("V", 5), ("Z", 5), ("B2", 5)],
    "B2": [("W", 5), ("A2", 5), ("C2", 5)],
    "C2": [("X", 5), ("B2", 5), ("Y", 8)]
}
def find_closest_node(x, y):
    closest_node = None
    min_distance = float("inf")

    for node, (nx, ny) in nodes.items():
        distance = math.sqrt((x - nx) ** 2 + (y - ny) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_node = node

    return closest_node

import heapq

def dijkstra(graph, start, goal):
    # Priority queue for exploring the shortest path
    pq = [(0, start)]  # (cumulative_distance, current_node)
    # Stores the minimum known distance to each node
    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    # Keeps track of the path taken
    previous_nodes = {}

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # If we've reached the goal, stop and reconstruct the path
        if current_node == goal:
            path = []
            while current_node:
                path.append(current_node)
                current_node = previous_nodes.get(current_node)
            return path[::-1], distances[goal]  # Reverse the path and return the distance

        # If a shorter path to a node is already known, skip it
        if current_distance > distances[current_node]:
            continue

        # Explore neighbors
        for neighbor, weight in graph.get(current_node, []):
            distance = current_distance + weight

            # If a shorter path to the neighbor is found, update it
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return [], float("inf")  # No path found


def choose_best_fare(list_of_fares):
    # Select the fare with the maximum pay value
    selected_fare = max(list_of_fares, key=lambda fare: fare['pay'])
    return selected_fare

def choose_fare():
    res = request.urlopen(server + "/fares")
    if res.status == 200:
        # Decode JSON data
        fares = json.loads(res.read())

        # Collect all unclaimed fares
        unclaimed_fares = [fare for fare in fares if not fare['claimed']]

        # If there are no unclaimed fares, exit early
        if not unclaimed_fares:
            print("No unclaimed fares available.")
            return

        # Select the best fare based on pay
        best_fare = choose_best_fare(unclaimed_fares)
        toClaim = best_fare['id']

        # Attempt to claim the best fare
        res = request.urlopen(server + f"/fares/claim/{toClaim}?auth={authKey}")

        # Verify that we got HTTP OK
        if res.status == 200:
            # Decode JSON data
            data = json.loads(res.read())
            if data['success']:
                print("Claimed fare id", toClaim)
            else:
                print("Failed to claim fare", toClaim, "reason:", data['message'])
        else:
            print("Got status", str(res.status), "claiming fare")

    else:
        # Report HTTP request error
        print("Got status", str(res.status), "requesting fares")

    #print("Selected Fare:", best_fare)
    return best_fare


def shortest_path(src, dest):
    src_node = find_closest_node(src['x'], src['y'])
    dest_node = find_closest_node(dest['x'], dest['y'])
    print("Nearest node to source: ", src_node)
    print("Nearest node to destination: ", dest_node)

    path, weight = dijkstra(graph, src_node, dest_node)  # Unpack the tuple

    return {"Path": path, "Weight": weight}  # Return as a dictionary


