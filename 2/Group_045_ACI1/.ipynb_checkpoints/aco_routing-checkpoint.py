import random
import heapq

def read_input_file(filename):
    graph = {}

    with open(filename, 'r') as file:
        n = int(file.readline())

        for _ in range(n):
            u, v, w = map(int, file.readline().split())

            if u not in graph:
                graph[u] = {}

            if v not in graph:
                graph[v] = {}

            graph[u][v] = w
            graph[v][u] = w

        source = int(file.readline())
        destination = int(file.readline())

    return graph, source, destination


def dijkstra(graph, start, end):
    pq = [(0, start, [])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        path = path + [node]

        if node == end:
            return path, cost

        for neighbor in graph[node]:
            if neighbor not in visited:
                heapq.heappush(
                    pq,
                    (cost + graph[node][neighbor], neighbor, path)
                )

    return None, float('inf')


class AntColony:

    def __init__(self, graph, ants, alpha, beta, evaporation, iterations=50):

        self.graph = graph
        self.ants = ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.iterations = iterations

        self.pheromone = {}

        for u in graph:
            self.pheromone[u] = {}

            for v in graph[u]:
                self.pheromone[u][v] = 1.0

    def heuristic(self, distance):
        return 1.0 / distance

    def choose_next_node(self, current, visited):

        probabilities = []
        total = 0

        for neighbor in self.graph[current]:

            if neighbor not in visited:

                pheromone = (
                    self.pheromone[current][neighbor]
                    ** self.alpha
                )

                heuristic = (
                    self.heuristic(
                        self.graph[current][neighbor]
                    )
                    ** self.beta
                )

                prob = pheromone * heuristic

                probabilities.append((neighbor, prob))
                total += prob

        if total == 0:
            return None

        r = random.uniform(0, total)

        cumulative = 0

        for neighbor, prob in probabilities:
            cumulative += prob

            if cumulative >= r:
                return neighbor

        return probabilities[-1][0]

    def construct_path(self, start, end):

        path = [start]
        visited = set(path)

        current = start
        total_cost = 0

        while current != end:

            next_node = self.choose_next_node(current, visited)

            if next_node is None:
                return None, float('inf')

            path.append(next_node)

            total_cost += self.graph[current][next_node]

            visited.add(next_node)

            current = next_node

        return path, total_cost

    def update_pheromone(self, all_paths):

        for u in self.pheromone:
            for v in self.pheromone[u]:
                self.pheromone[u][v] *= (1 - self.evaporation)

        for path, cost in all_paths:

            if path is None:
                continue

            deposit = 1.0 / cost

            for i in range(len(path) - 1):

                u = path[i]
                v = path[i + 1]

                self.pheromone[u][v] += deposit
                self.pheromone[v][u] += deposit

    def run(self, start, end):

        best_path = None
        best_cost = float('inf')

        for iteration in range(self.iterations):

            all_paths = []

            for _ in range(self.ants):

                path, cost = self.construct_path(start, end)

                all_paths.append((path, cost))

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            self.update_pheromone(all_paths)

        return best_path, best_cost


def write_output(filename, scenario, best_path, best_cost,
                 dijkstra_path, dijkstra_cost):

    with open(filename, 'a') as file:

        file.write(f"{scenario}\\n")

        file.write(
            "ACO Best Path: "
            + " -> ".join(map(str, best_path))
            + "\\n"
        )

        file.write(
            f"ACO Minimum Latency: {best_cost}\\n"
        )

        file.write(
            "Dijkstra Path: "
            + " -> ".join(map(str, dijkstra_path))
            + "\\n"
        )

        file.write(
            f"Dijkstra Minimum Latency: "
            f"{dijkstra_cost}\\n\\n"
        )


def main():

    input_file = "inputPS13.txt"
    output_file = "outputPS13.txt"

    graph, source, destination = read_input_file(input_file)

    aco1 = AntColony(
        graph,
        ants=10,
        alpha=1.0,
        beta=2,
        evaporation=0.5
    )

    best_path1, best_cost1 = aco1.run(source, destination)

    aco2 = AntColony(
        graph,
        ants=10,
        alpha=2.5,
        beta=1.0,
        evaporation=0.3
    )

    best_path2, best_cost2 = aco2.run(source, destination)

    dijkstra_path, dijkstra_cost = dijkstra(
        graph,
        source,
        destination
    )

    open(output_file, 'w').close()

    write_output(
        output_file,
        "Scenario 1",
        best_path1,
        best_cost1,
        dijkstra_path,
        dijkstra_cost
    )

    write_output(
        output_file,
        "Scenario 2",
        best_path2,
        best_cost2,
        dijkstra_path,
        dijkstra_cost
    )

    print("Output written to outputPS13.txt")


if __name__ == "__main__":
    main()
