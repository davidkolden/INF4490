import hill_climber
import genetic_algorithm
import matplotlib.pyplot as plt
from enum import Enum
import sys
import csv
import numpy as np
import copy
import recombination as recomb
import datetime


class LearningModel(Enum):
    LAMARCKIAN = 0,
    BALDWINIAN = 1

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
    best_distance = calcuate_total_distance(perm, table)
    best_order = copy.deepcopy(perm)
    #print("Fitness before[in hill_climber_search]: " + str(best_distance))
    for n in range(iterations):
        swap_cities(perm)
        total = calcuate_total_distance(perm, table)
        if total < best_distance:
            best_distance = total
            best_order = copy.deepcopy(perm)
    #print("Fitness after[in hill_climber_search]: " + str(calcuate_total_distance(best_order, table)))
    return best_distance, best_order


def hybridize(parent_list, learning_model, table, n_hill_climber_searches):

    if learning_model == LearningModel.LAMARCKIAN:
        # replacing the pre search value with the new one

        for i in range(len(parent_list)):
            unused, parent_list[i] = hill_climber.hill_climber_search(table, parent_list[i], n_hill_climber_searches)

        list.sort(parent_list, key=lambda seg: hill_climber.calculate_total_distance(seg, table))

        return parent_list

    else:
        # sorting with respect to the new value found, but using the values from pre search

        distance_list = []

        for i in range(len(parent_list)):
            distance, unused = hill_climber.hill_climber_search(table, parent_list[i], n_hill_climber_searches)
            distance_list.append(distance)

        # sort parent_list based on values in distance_list
        parent_list = [x for _,x in sorted(zip(distance_list, parent_list))]

        return parent_list


def hybrid_algorithm(parent_list,
                     table,
                     s,
                     mutation_prob,
                     n_run,
                     learning_model,
                     n_hill_climber_searches,
                     n_children
                     ):

    parent_list = (hybridize(parent_list, learning_model, table, n_hill_climber_searches))
    best_individuals = []
    crossover_algorithm = recomb.Crossover()

    for c in range(n_run):

        children = []
        for i in range(n_children):
            children.append(recomb.Genotype([]))

        for i in range(0, n_children, 2):
            parent1 = recomb.Genotype([])
            parent2 = recomb.Genotype([])
            parent1.data, parent2.data = genetic_algorithm.parent_selector(parent_list, table, s)
            children[i], children[i + 1] = crossover_algorithm.cycle_cross_over(parent1, parent2)
            genetic_algorithm.mutate_inversion(children[i].data, mutation_prob)
            genetic_algorithm.mutate_inversion(children[i + 1].data, mutation_prob)

        genetic_algorithm.survivor_selector_genitor(parent_list, children)

        list.sort(parent_list, key=lambda seg: hill_climber.calculate_total_distance(seg, table))

        best_individuals.append(calcuate_total_distance(parent_list[0], table))

        parent_list = hybridize(parent_list, learning_model, table, n_hill_climber_searches)

    return best_individuals, parent_list


def run_algorithm(total_cities,
                  population_size,
                  n_rounds,
                  s,
                  mutation_prob,
                  table,
                  n_run,
                  names,
                  learning_model,
                  n_hill_climber_searches,
                  n_children
                  ):

    best_distance = 1000000
    worst_distance = 0
    best_distance_array = []
    best_individual_per_run = []

    start = datetime.datetime.now()
    for n in range(0, n_rounds):
        # create initial gene pool
        gene_pool = []
        for i in range(0, population_size):
            gene_pool.append(list(np.random.permutation(total_cities)))

        # fetching the strongest individual every algorithm cycle
        best_individual, gene_pool = hybrid_algorithm(gene_pool, table, s, mutation_prob, n_run, learning_model, n_hill_climber_searches, n_children)
        best_individual_per_run.append(best_individual)

        # getting the strongest individual of this algorithm round
        tmp_individual = hill_climber.calculate_total_distance(gene_pool[0], table)

        if tmp_individual < best_distance:
            best_distance = tmp_individual

        if tmp_individual > worst_distance:
            worst_distance = tmp_individual

        best_distance_array.append(tmp_individual)

    end = datetime.datetime.now()
    delta_t = end - start
    time_taken = delta_t.microseconds/1000000 + delta_t.seconds

    print(
        "Search: " +
        str(total_cities) +
        " cities, population size: " +
        str(population_size) +
        ", number of generations: " +
        str(n_run) +
        ", number of rounds: " +
        str(n_rounds) +
        ", number of children: " +
        str(n_children) +
        ", number of hill climb iterations: " +
        str(n_hill_climber_searches) +
        ": "
    )
    print("Best distance: " + str(best_distance))
    print("Worst distance: " + str(worst_distance))
    print("Average distance: " + str(np.mean(best_distance_array, dtype=np.float32)))
    print("Standard deviation: " + str(np.std(best_distance_array, dtype=np.float32)))
    print("Time [seconds]: " + str(time_taken))
    print("Best order of travel: ")
    for k in range(len(gene_pool[0])-1):
        print(names[gene_pool[0][k]], end=" ")
    print(names[gene_pool[0][0]])
    print(" ")

    return best_individual_per_run


if __name__ == '__main__':
    # open csv file and get the table of cities and distances
    f = open(sys.argv[1], 'r')
    reader = csv.reader(f, delimiter=';')
    table = list(reader)
    # store names in own list
    names = table[0]
    table.pop(0)


    fig = plt.figure("Hybrid Algorithm - Baldwinian learning model")
    fig.suptitle("Average fitness of best fit individual in each generation")
    plt.ylabel("Distance")
    plt.xlabel("Number of generations")

    total_cities = 24
    population_size = 10
    n_rounds = 20
    s = 1
    mutation_prob = 0.5
    number_of_algorithm_runs = 500
    learning_model = LearningModel.BALDWINIAN
    n_hill_climber_searches = 3
    n_children = 4


    print("---- BALDWINIAN LEARNING MODEL ----")

    best_distance_matrix = []
    best_distance_matrix = run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    best_distance_average1 = []
    for i in range(len(best_distance_matrix[0])):
        sum = 0
        for j in range(len(best_distance_matrix)):
            sum += best_distance_matrix[j][i]

        best_distance_average1.append(sum / len(best_distance_matrix))

        population_size = 50

    best_distance_matrix = run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    best_distance_average2 = []
    for i in range(len(best_distance_matrix[0])):
        sum = 0
        for j in range(len(best_distance_matrix)):
            sum += best_distance_matrix[j][i]

        best_distance_average2.append(sum / len(best_distance_matrix))

        population_size = 100

    best_distance_matrix = run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    best_distance_average3 = []
    for i in range(len(best_distance_matrix[0])):
        sum = 0
        for j in range(len(best_distance_matrix)):
            sum += best_distance_matrix[j][i]

        best_distance_average3.append(sum / len(best_distance_matrix))

    total_cities = 10
    population_size = 10

    run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    population_size = 50

    run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    population_size = 100

    run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    plt.plot(range(number_of_algorithm_runs), best_distance_average1, 'b', label='Population size 10')
    plt.plot(range(number_of_algorithm_runs), best_distance_average2, 'r', label='Population size 50')
    plt.plot(range(number_of_algorithm_runs), best_distance_average3, 'g', label='Population size 100')

    plt.legend()
    if (len(sys.argv) > 2):
        plt.savefig(sys.argv[2] + "_baldwinian" + ".pdf", format="pdf")

    print("---- LAMARCKIAN LEARNING MODEL ----")

    plt.clf()
    fig = plt.figure("Hybrid Algorithm - Lamarckian learning model")
    fig.suptitle("Average fitness of best fit individual in each generation")
    plt.ylabel("Distance")
    plt.xlabel("Number of generations")

    learning_model = LearningModel.LAMARCKIAN
    total_cities = 24
    population_size = 10

    best_distance_matrix = []
    best_distance_matrix = run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    best_distance_average1 = []
    for i in range(len(best_distance_matrix[0])):
        sum = 0
        for j in range(len(best_distance_matrix)):
            sum += best_distance_matrix[j][i]

        best_distance_average1.append(sum / len(best_distance_matrix))

        population_size = 50

    best_distance_matrix = run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    best_distance_average2 = []
    for i in range(len(best_distance_matrix[0])):
        sum = 0
        for j in range(len(best_distance_matrix)):
            sum += best_distance_matrix[j][i]

        best_distance_average2.append(sum / len(best_distance_matrix))

        population_size = 100

    best_distance_matrix = run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    best_distance_average3 = []
    for i in range(len(best_distance_matrix[0])):
        sum = 0
        for j in range(len(best_distance_matrix)):
            sum += best_distance_matrix[j][i]

        best_distance_average3.append(sum / len(best_distance_matrix))

    total_cities = 10
    population_size = 10

    run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    population_size = 50

    run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    population_size = 100

    run_algorithm(
        total_cities=total_cities,
        population_size=population_size,
        n_rounds=n_rounds,
        s=s,
        mutation_prob=mutation_prob,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names,
        learning_model=learning_model,
        n_hill_climber_searches=n_hill_climber_searches,
        n_children=n_children
    )

    plt.plot(range(number_of_algorithm_runs), best_distance_average1, 'b', label='Population size 10')
    plt.plot(range(number_of_algorithm_runs), best_distance_average2, 'r', label='Population size 50')
    plt.plot(range(number_of_algorithm_runs), best_distance_average3, 'g', label='Population size 100')

    plt.legend()
    if (len(sys.argv) > 2):
        plt.savefig(sys.argv[2] + "_lamarckian" + ".pdf", format="pdf")

    f.close()

