
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from heapq import heappush
from typing import List


blue_points: List[List[int]] = [[2,4], [1,3], [2,3], [3,2], [2,1]]
red_points: List[List[int]] = [[5,6], [4,5], [4,6], [6,6], [5,4]]

points = {
    "red": red_points,
    "blue": blue_points
}

RED = "#FF0000"
BLUE = "#104DCA"

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

class KNN:
    def __init__(self, k: int = 3):
        self.k = k
        self.points = None

    def fit(self, points: dict):
        self.points = points

    def predict(self, new_point: List[int]):
        distances = []
        for category, points in self.points.items():
            for point in points:
                distance = euclidean_distance(point, new_point)
                heappush(distances, (distance, category))
        print(distances)
        k_closest = [pair[1] for pair in distances[:self.k]]
        return Counter(k_closest).most_common(1)[0][0]



def main():
    cluster = KNN()
    cluster.fit(points)
    new_point = [3,3]
    category_prediction = cluster.predict(new_point)

    ax = plt.subplot()
    ax.grid(True, color="#323232")
    ax.set_facecolor("black")
    ax.figure.set_facecolor("#121212")
    ax.tick_params(axis="x", color="white")
    ax.tick_params(axis="y", color="white")

    for point in points["blue"]:
        ax.scatter(x=point[0], y=point[1], color=BLUE, s=60)

    for point in points["red"]:
        ax.scatter(x=point[0], y=point[1], color=RED, s=60)

    color = RED if category_prediction == "red" else BLUE

    ax.scatter(new_point[0], new_point[1], color=color, marker="*", s=200, zorder=100)

    for point in points["blue"]:
        ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color=BLUE, linestyle="--", linewidth=1)

    for point in points["red"]:
        ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color=RED, linestyle="--", linewidth=1)

    plt.show()


if __name__ == "__main__":
    main()