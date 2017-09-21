import hill_climber
import genetic_algorithm
import matplotlib.pyplot as plt
from enum import Enum
import sys
import csv
import numpy as np


class LearningModel(Enum):
    LAMARCKIAN = 0,
    BALDWINIAN = 1


def hybridize(parent_list, learning_model, table, n_hill_climber_searches):
    if learning_model == LearningModel.LAMARCKIAN:
        # replacing the pre search value with the new one

        for i in range(len(parent_list)-1):
            unused, parent_list[i] = hill_climber.hill_climber_search(table, parent_list[i], n_hill_climber_searches)

        list.sort(parent_list, key=lambda seg: hill_climber.calcuate_total_distance(seg, table))

    else:
        # sorting with respect to the new value found, but using the values from pre search

        distance_list = []
        tmp_list = parent_list
        for i in range(len(parent_list)-1):
            distance, unused = hill_climber.hill_climber_search(table, tmp_list[i], n_hill_climber_searches)
            distance_list.append(distance)

        distances_and_parents = zip(distance_list, parent_list)
        parent_list = sorted(distances_and_parents, key=lambda x:x[1])



def hybrid_algorithm():
    pass


if __name__ == '__main__':
    # open csv file and get the table of cities and distances
    f = open(sys.argv[1], 'r')
    reader = csv.reader(f, delimiter=';')
    table = list(reader)
    # store names in own list
    names = table[0]
    table.pop(0)

    population_size = 20
    total_cities = 10

    gene_pool = []
    for i in range(0, population_size):
        gene_pool.append(np.random.permutation(total_cities))

    print("Distances before: ")
    for i in range(len(gene_pool)):
        print(str(hill_climber.calcuate_total_distance(gene_pool[i], table)), end=" ")

    hybridize(gene_pool, LearningModel.BALDWINIAN, table, n_hill_climber_searches=10)

    print(" ")

    print("Distances after: ")
    for i in range(len(gene_pool)):
        print(str(hill_climber.calcuate_total_distance(gene_pool[i], table)), end=" ")
