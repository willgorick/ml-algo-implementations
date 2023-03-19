
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from heapq import heappush
from typing import List
import sys

POINTS = {
    "red": [(0, 4),(1, 4.9),(1.6, 5.4),(2.2, 6),(2.8, 7),(3.2, 8),(3.4, 9)],
    "blue": [(1.8, 1),(2.2, 3),(3, 4),(4, 4.5),(5, 5),(6, 5.5)]
}

RED = "#FF0000"
BLUE = "#104DCA"
GRAY = "#323232"
BLACK = "#121212"

class KNN:
    def __init__(self, k: int = 3):
        self.k = k
        self.points = None

    def fit(self, points: dict):
        self.points = points

    def euclidean_distance(self, p, q):
        """Calculate the euclidean distance between two points"""
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

    def predict(self, new_point: List[int]) -> str:
        """Predict which category the new point should 
        receive based on the k nearest numbers algorithm"""

        distances = []
        for category, points in self.points.items():
            for point in points:
                distance = self.euclidean_distance(point, new_point)
                heappush(distances, (distance, category))

        k_closest = [pair[1] for pair in distances[:self.k]]
        return Counter(k_closest).most_common(1)[0][0]
    
    def predict_weighted(self, new_point: List[int]) -> SystemError:
        """Predict which category the new point should 
        receive based on a weighted version of the
        k nearest numbers algorithm"""
        distances = []
        for category, points in self.points.items():
            for point in points:
                distance = self.euclidean_distance(point, new_point)
                heappush(distances, (distance, category))

        category_weights = defaultdict(int)
        for dist_category in distances[:self.k]:
            category_weights[dist_category[1]] += 1/dist_category[0]

        return max(category_weights, key=lambda k:category_weights[k])
    
    def plot_points(self, category_prediction: str, new_point: List[int]) -> None:
        """Plot points on a matplotlib graph with a star to 
        indicate the newly classified point"""

        ax = plt.subplot()
        ax.grid(True, color=GRAY)
        ax.set_facecolor("black")
        ax.figure.set_facecolor(BLACK)
        ax.tick_params(axis="x", color="white")
        ax.tick_params(axis="y", color="white")

        for point in self.points["blue"]:
            ax.scatter(x=point[0], y=point[1], color=BLUE, s=60)

        for point in self.points["red"]:
            ax.scatter(x=point[0], y=point[1], color=RED, s=60)

        color = RED if category_prediction == "red" else BLUE

        ax.scatter(new_point[0], new_point[1], color=color, marker="*", s=200, zorder=100)

        for point in self.points["blue"]:
            ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color=BLUE, linestyle="--", linewidth=1)

        for point in self.points["red"]:
            ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color=RED, linestyle="--", linewidth=1)

        plt.show()

def main():
    cluster = KNN(5)
    cluster.fit(POINTS)
    new_point = [2,4]

    if len(sys.argv) > 1 and sys.argv[1] == "weighted":
        category_prediction = cluster.predict_weighted(new_point)
        print(f"Weighted prediction: {category_prediction}")
    else:
        category_prediction = cluster.predict(new_point)
        print(f"Unweighted prediction: {category_prediction}")

    cluster.plot_points(category_prediction, new_point)



if __name__ == "__main__":
    main()