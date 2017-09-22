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


def hybridize(parent_list, learning_model, table, n_hill_climber_searches):
    if learning_model == LearningModel.LAMARCKIAN:
        # replacing the pre search value with the new one

        for i in range(len(parent_list)):
            unused, parent_list[i] = hill_climber.hill_climber_search(table, parent_list[i], n_hill_climber_searches)

        list.sort(parent_list, key=lambda seg: hill_climber.calcuate_total_distance(seg, table))

        return parent_list

    else:
        # sorting with respect to the new value found, but using the values from pre search

        distance_list = []
        tmp_list = copy.deepcopy(parent_list)
        for i in range(len(tmp_list)):
            distance, unused = hill_climber.hill_climber_search(table, tmp_list[i], n_hill_climber_searches)
            distance_list.append(distance)

        new_list = [x for _,x in sorted(zip(distance_list, parent_list))]

        return new_list


def hybrid_algorithm(parent_list, table, s, mutation_prob, n_run, learning_model, n_hill_climber_searches):
    parent_list = hybridize(parent_list, learning_model, table, n_hill_climber_searches)
    best_individuals = []
    crossover_algorithm = recomb.Crossover()
    for c in range(n_run):
        parent1 = recomb.Genotype([])
        parent2 = recomb.Genotype([])
        child1 = recomb.Genotype([])
        child2 = recomb.Genotype([])

        parent1.data, parent2.data = genetic_algorithm.parent_selector(parent_list, table, s)

        child1, child2 = crossover_algorithm.cycle_cross_over(parent1, parent2)

        genetic_algorithm.mutate_inversion(child1.data, mutation_prob)
        genetic_algorithm.mutate_inversion(child2.data, mutation_prob)

        genetic_algorithm.survivor_selector_genitor(parent_list, child1.data, child2.data)

        parent_list = hybridize(parent_list, learning_model, table, n_hill_climber_searches)
        best_individuals.append(hill_climber.calcuate_total_distance(parent_list[0], table))

    return best_individuals


def run_algorithm(total_cities, population_size, n_rounds, s, mutation_prob, table, n_run, names, learning_model, n_hill_climber_searches):

    best_distance = 1000000
    worst_distance = 0
    best_distance_array = []
    best_individual_per_run = []

    start = datetime.datetime.now()
    for n in range(0, n_rounds):
        # create initial gene pool
        gene_pool = []
        for i in range(0, population_size):
            gene_pool.append(np.random.permutation(total_cities))

        # fetching the strongest individual every algorithm cycle
        best_individual_per_run.append(hybrid_algorithm(gene_pool, table, s, mutation_prob, n_run, learning_model, n_hill_climber_searches))

        # getting the strongest individual of this algorithm round
        tmp_individual = hill_climber.calcuate_total_distance(gene_pool[0], table)

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


    fig = plt.figure("Hybrid Algorithm")
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
    n_hill_climber_searches = 5

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
        n_hill_climber_searches=n_hill_climber_searches
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
        n_hill_climber_searches=n_hill_climber_searches
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
        n_hill_climber_searches=n_hill_climber_searches
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
        n_hill_climber_searches=n_hill_climber_searches
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
        n_hill_climber_searches=n_hill_climber_searches
    )

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
        n_hill_climber_searches=n_hill_climber_searches
    )

    plt.plot(range(number_of_algorithm_runs), best_distance_average1, 'b', label='Population size 10')
    plt.plot(range(number_of_algorithm_runs), best_distance_average2, 'r', label='Population size 50')
    plt.plot(range(number_of_algorithm_runs), best_distance_average3, 'g', label='Population size 100')

    plt.legend()
    if (len(sys.argv) > 2):
        plt.savefig(sys.argv[2] + ".pdf", format="pdf")
    f.close()

