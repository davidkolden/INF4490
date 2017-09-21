import recombination as recomb
import numpy as np
import exhaustive
import itertools
import sys
import csv


def mutate_inversion(permu, probability):
    if probability >= np.random.uniform(0.0, 1.0):
        ind1 = np.random.random_integers(len(permu)-1)
        ind2 = np.random.random_integers(len(permu)-1)
        while ind1 == ind2:
            ind2 = np.random.random_integers(len(permu)-1)
        if ind1 > ind2:
            permu[ind2:ind1+1] = reversed(permu[ind2:ind1+1])
        else:
            permu[ind1:ind2+1] = reversed(permu[ind1:ind2+1])


def parent_selector(parent_list, table, s):
    list.sort(parent_list, key=lambda seg: exhaustive.calcuate_total_distance(seg, table))
    pre_prob = (2-s)/len(parent_list)
    parent1 = []
    parent2 = []
    for i in range(0, len(parent_list)-1):
        prob = pre_prob + (2*(len(parent_list)-1-i) * (s-1))/(len(parent_list)*(len(parent_list) - 1))
        if prob >= np.random.uniform(0.0, 1.0):
            parent1 = parent_list[i]
            break

    for i in range(1, len(parent_list)-1):
        prob = pre_prob + (2*(len(parent_list) - 1 - i) * (s-1))/(len(parent_list)*(len(parent_list) - 1))
        if prob >= np.random.uniform(0.0, 1.0):
            parent2 = parent_list[i]
            break

    if not parent1 and not parent2:
        parent1 = parent_list[len(parent_list) - 2]
        parent2 = parent_list[len(parent_list) - 1]

    if not parent1:
        parent1 = parent_list[len(parent_list) - 1]

    if not parent2:
        parent2 = parent_list[len(parent_list) - 1]

    return parent1, parent2

def survivor_selector(parent_list)

def genetic_algorithm(parent1, parent2):
    pass
    # Choose mutation and crossover operators
    # Define and tune parameters


if __name__ == '__main__':
    f = open(sys.argv[1], 'r')
    reader = csv.reader(f, delimiter=';')
    l = list(reader)
    names = l[0]
    l.pop(0)

    permu = list(itertools.permutations([0, 1, 2, 3, 4], 5))
    parent1, parent2 = parent_selector(permu, l, 1.9)

    print("Parent1: distance: " + str(exhaustive.calcuate_total_distance(parent1, l)) + ", order: ", end=" ")
    print(parent1)

    print("Parent2: distance: " + str(exhaustive.calcuate_total_distance(parent2, l)) + ", order: ", end=" ")
    print(parent2)


    # mutate_inversion(permu, 1.0)

    # Choose three different values for population size
    # Report best, worst, average and standard deviation og 20 runs
    # Plot average fitness across generation of each run (for three population sizes)
    # Run for 10 and 24 cities