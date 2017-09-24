import numpy as np
import sys
import csv
import copy
import datetime


def calculate_total_distance(seg, table):
    total = 0
    for i, val in enumerate(seg[:(len(seg) - 1)]):
        total += float(table[val][seg[i + 1]])
    total += float(table[seg[len(seg) - 1]][seg[0]])
    return total


def swap_cities(perm):
    ind1 = np.random.random_integers(len(perm)-1)
    ind2 = np.random.random_integers(len(perm)-1)

    if ind1 != ind2:
        tmp = perm[ind1]
        perm[ind1] = perm[ind2]
        perm[ind2] = tmp


def hill_climber_search(table, perm, iterations):

    best_order = copy.deepcopy(perm)
    best_distance = calculate_total_distance(best_order, table)

    for n in range(iterations):

        swap_cities(perm)
        total = calculate_total_distance(perm, table)

        if total < best_distance:
            best_distance = total
            best_order = copy.deepcopy(perm)

    return best_distance, best_order


if __name__ == '__main__':

    f = open(sys.argv[1], 'r')
    reader = csv.reader(f, delimiter=';')
    l = list(reader)
    names = l[0]
    l.pop(0)

    n = 10
    max_cities = range(0, 24)
    total_cities = max_cities[:n]

    best_dist = 1000000;
    worst_dist = 0
    distance_array = []
    rounds = 20
    n_searches = 10000
    algorithm_time = 0

    for i in range(rounds):
        permus = np.random.permutation(total_cities)
        start = datetime.datetime.now()
        dist, order = hill_climber_search(l, permus, 10000)
        end = datetime.datetime.now()
        time_delta = end - start
        algorithm_time = time_delta.seconds + time_delta.microseconds/1000000

        distance_array.append(dist)
        if dist < best_dist:
            best_dist = dist
        if dist > worst_dist:
            worst_dist = dist

    print("For " + str(n) + " cities:")
    print("Running the algorithm " + str(rounds) + " times")
    print("Number of searches per round: " + str(n_searches))
    print("Best distance: " + str(best_dist))
    print("Worst distance: " + str(worst_dist))
    print("Average distance: " + str(np.mean(distance_array, dtype=np.float32)))
    print("Standard deviation: " + str(np.std(distance_array, dtype=np.float32)))
    print("Time taken per search[seconds]: " + str(algorithm_time))
    print(" ")

    n = 24
    max_cities = range(0, 24)
    total_cities = max_cities[:n]

    best_dist = 1000000;
    worst_dist = 0
    distance_array = []

    for i in range(20):
        permus = np.random.permutation(total_cities)
        start = datetime.datetime.now()
        dist, order = hill_climber_search(l, permus, 1000000)
        distance_array.append(dist)
        end = datetime.datetime.now()
        time_delta = end - start
        algorithm_time = time_delta.seconds + time_delta.microseconds / 1000000

        if dist < best_dist:
            best_dist = dist
        if dist > worst_dist:
            worst_dist = dist

    print("For " + str(n) + " cities:")
    print("Running the algorithm " + str(rounds) + " times")
    print("Number of searches per round: " + str(n_searches))
    print("Best distance: " + str(best_dist))
    print("Worst distance: " + str(worst_dist))
    print("Average distance: " + str(np.mean(distance_array, dtype=np.float32)))
    print("Standard deviation: " + str(np.std(distance_array, dtype=np.float32)))
    print("Time taken per search[seconds]: " + str(algorithm_time))
    print(" ")
