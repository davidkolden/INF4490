import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import datetime


def calcuate_total_distance(seg, table):
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
    best_distance = 1000000
    best_order = []
    for n in range(iterations):
        swap_cities(perm)
        total = calcuate_total_distance(perm, table)
        if total < best_distance:
            best_distance = total
            best_order = perm
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

    for i in range(20):
        permus = np.random.permutation(total_cities)
        dist, order = hill_climber_search(l, permus, 10000)
        distance_array.append(dist)
        if dist < best_dist:
            best_dist = dist
        if dist > worst_dist:
            worst_dist = dist

    print("For 10 cities:")
    print("Best distance: " + str(best_dist))
    print("Worst distance: " + str(worst_dist))
    print("Average distance: " + str(np.mean(distance_array, dtype=np.float32)))
    print("Standard deviation: " + str(np.std(distance_array, dtype=np.float32)))
    print(" ")

    n = 24
    max_cities = range(0, 24)
    total_cities = max_cities[:n]

    best_dist = 1000000;
    worst_dist = 0
    distance_array = []

    for i in range(20):
        permus = np.random.permutation(total_cities)
        dist, order = hill_climber_search(l, permus, 10000)
        distance_array.append(dist)
        if dist < best_dist:
            best_dist = dist
        if dist > worst_dist:
            worst_dist = dist

    print("For 24 cities:")
    print("Best distance: " + str(best_dist))
    print("Worst distance: " + str(worst_dist))
    print("Average distance: " + str(np.mean(distance_array, dtype=np.float32)))
    print("Standard deviation: " + str(np.std(distance_array, dtype=np.float32)))
