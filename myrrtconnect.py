import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


class RRTStar:
    def __init__(self, start, goal, image, max_iterations, refine_cycles, refine_cycle_iterations, step_size, goal_sample_rate, connect_circle_dist):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.image = image
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.connect_circle_dist = connect_circle_dist
        self.node_list_start = [self.start]
        self.node_list_goal = [self.goal]
        self.refine_cycles = refine_cycles
        self.refine_cycle_iterations = refine_cycle_iterations

    def plan(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image, cmap='gray')
        ax.plot(self.start.x, self.start.y, 'ro')
        ax.plot(self.goal.x, self.goal.y, 'go')

        for i in range(self.max_iterations):
            if random.random() < self.goal_sample_rate:
                if random.random() < .5:
                    rnd_node = Node(self.goal.x, self.goal.y)
                    new_node = self.connect(
                        rnd_node, self.node_list_start, ax, 'b-')
                    if new_node:
                        path = self.check(new_node, ax, self.node_list_goal, 0)
                        if path:
                            refined_path = self.refine(path, ax)
                            return refined_path
                else:
                    rnd_node = Node(self.start.x, self.start.y)
                    new_node = self.connect(
                        rnd_node, self.node_list_goal, ax, 'm-')
                    if new_node:
                        path = self.check(
                            new_node, ax, self.node_list_start, 1)
                        if path:
                            refined_path = self.refine(path, ax)
                            return refined_path

            else:
                rnd_node = Node(random.uniform(
                    0, self.image.shape[1]), random.uniform(0, self.image.shape[0]))

                new_node = self.connect(
                    rnd_node, self.node_list_start, ax, 'b-')
                if new_node:
                    path = self.check(new_node, ax, self.node_list_goal, 0)
                    if path:
                        refined_path = self.refine(path, ax)
                        return refined_path

                new_node = self.connect(
                    rnd_node, self.node_list_goal, ax, 'm-')
                if new_node:
                    path = self.check(new_node, ax, self.node_list_start, 1)
                    if path:
                        refined_path = self.refine(path, ax)
                        return refined_path

        plt.show()
        return None

    def connect(self, rnd_node, node_list, ax, colour):
        nearest_node = self.nearest_node(rnd_node, node_list)
        new_node = self.steer(nearest_node, rnd_node, self.step_size)

        if self.check_collision(nearest_node, new_node):
            near_nodes = self.near_nodes(
                new_node, self.connect_circle_dist, node_list)
            min_cost_node = nearest_node
            min_cost = self.get_cost(nearest_node) + \
                self.distance(nearest_node, new_node)

            for node in near_nodes:
                if self.check_collision(node, new_node) and self.get_cost(node) + self.distance(node, new_node) < min_cost:
                    min_cost_node = node
                    min_cost = self.get_cost(
                        node) + self.distance(node, new_node)

            new_node.parent = min_cost_node
            node_list.append(new_node)

            for node in near_nodes:
                if self.check_collision(new_node, node) and self.get_cost(new_node) + self.distance(new_node, node) < self.get_cost(node):
                    node.parent = new_node

            ax.plot([new_node.x, min_cost_node.x], [
                    new_node.y, min_cost_node.y], colour)
            plt.pause(0.001)
            return new_node

        return None

    def check(self, new_node, ax, node_list, k):
        near_nodes = self.near_nodes(new_node, self.step_size, node_list)
        near_nodes.append(self.nearest_node(new_node, node_list))

        for node in near_nodes:
            connect_node = node
            if self.check_collision(new_node, connect_node):

                if k == 1:
                    path = self.extract_path(connect_node, new_node)
                else:
                    path = self.extract_path(new_node, connect_node)

                path_coord = []
                for i in path:
                    path_coord.append((i.x, i.y))
                print(path_coord)
                ax.plot([x for (x, y) in path_coord], [
                        y for (x, y) in path_coord], 'g-')
                plt.pause(0.001)
                # plt.show()
                return path

        return None

    def distline(self, node):
        slope = (self.goal.y-self.start.y)/(self.goal.x-self.start.x)
        const = self.start.y - slope*self.start.x

        dist = abs(node.y-slope*node.x-const)/math.sqrt(slope**2+1)
        return dist

    def refine(self, path, ax):
        node_list = []
        for i in range(len(path)):
            node_list.append(path[i])
            if i:
                path[i].parent = path[i-1]

        for i in range(self.refine_cycles):
            max_dist = self.max_path_dist(path)
            print("Maximum Distance:", max_dist)

            for i in range(len(node_list)):
                if self.distline(node_list[i])>max_dist:
                    del node_list[i]
                    i-= 1

            for j in range(self.refine_cycle_iterations):
                y = random.uniform(-max_dist, max_dist)
                x = random.uniform(0, self.distance(self.start, self.goal))

                rnd_node = self.steer_refine(x, y)
                new_node = self.connect(rnd_node, node_list, ax, 'y-')
                # if new_node:
                # print(new_node.x, new_node.y, end = ", ")
            path_new = self.extract_path_refine()

        path_new = self.extract_path_refine()
        ax.plot([x for (x, y) in path_new], [
                y for (x, y) in path_new], 'r-')
        plt.pause(0.001)
        plt.show()
        return path_new

    def max_path_dist(self, path):
        max = 0
        for i in path:
            dist = self.distline(i)
            if (dist > max):
                max = dist

        return max

    def nearest_node(self, rnd_node, node_list):
        min_dist = float('inf')
        min_node = None

        for node in node_list:
            dist = self.distance(node, rnd_node)
            if dist < min_dist:
                min_dist = dist
                min_node = node

        return min_node

    def steer(self, from_node, to_node, step_size):
        dist = self.distance(from_node, to_node)

        if dist <= step_size:
            return to_node

        ratio = step_size / dist
        x = from_node.x + (to_node.x - from_node.x) * ratio
        y = from_node.y + (to_node.y - from_node.y) * ratio

        return Node(x, y)

    def steer_refine(self, x, y):
        node = Node(0, 0)
        slope = (self.goal.y-self.start.y)/(self.goal.x-self.start.x)
        x1 = self.start.x + x*math.cos(math.atan(slope))
        y1 = self.start.y + x*math.sin(math.atan(slope))
        height = self.image.shape[0]
        width = self.image.shape[1]

        while not node.x:
            x1_temp = x1-y*math.sin(math.atan(slope))
            y1_temp = y1+y*math.cos(math.atan(slope))
            if x1_temp > 0 and x1_temp < width and y1_temp > 0 and y1_temp < height:
                node.x = x1
                node.y = y1
            y /= 2

        return node

    def near_nodes(self, node, radius, node_list):
        near_nodes = []
        for near_node in node_list:
            if self.distance(near_node, node) <= radius:
                near_nodes.append(near_node)

        return near_nodes

    def get_cost(self, node):
        cost = 0.0
        while node.parent:
            cost += self.distance(node, node.parent)
            node = node.parent

        return cost

    def distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def check_collision(self, from_node, to_node):
        x1, y1 = int(from_node.x), int(from_node.y)
        x2, y2 = int(to_node.x), int(to_node.y)

        points = self.bresenham_line(x1, y1, x2, y2)
        for point in points:
            x, y = point
            if self.image[y, x] == 0:
                return False  # Collision detected

        return True  # No collision

    def bresenham_line(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        points = []
        while x0 != x1 or y0 != y1:
            points.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        points.append((x0, y0))
        return points

    def extract_path(self, node1, node2):

        path = []
        node = node1
        while node:
            path.append(node)
            node = node.parent
        path = path[::-1]

        dist = self.distance(node1, node2)
        increase_x = self.step_size/dist*(node2.x-node1.x)
        increase_y = self.step_size/dist*(node2.y-node1.y)
        x = node1.x
        y = node1.y

        for i in range(1, int(dist/self.step_size)):
            x += increase_x
            y += increase_y
            node = Node(x, y)
            path.append(node)

        print("Cost:", self.get_cost(node1)+self.get_cost(node2)+dist)
        node = node2
        while node:
            path.append(node)
            node = node.parent

        return path

    def extract_path_refine(self):
        path = []
        node = self.goal
        print("Cost:", self.get_cost(self.goal))
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]


# Example usage
start = (5, 5)
goal = (1150, 640)
image = cv2.imread("opencv/4.png", 0)

rrt_star = RRTStar(start, goal, image, max_iterations=5000, refine_cycles=2, refine_cycle_iterations=200,
                   step_size=10, goal_sample_rate=0.1, connect_circle_dist=50)
path = rrt_star.plan()

if path is None:
    print("No valid path found")
else:
    print("Path found:", path)
