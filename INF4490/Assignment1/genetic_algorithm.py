import recombination as recomb
import numpy as np
import exhaustive
import sys
import csv
import matplotlib.pyplot as plt
import datetime


def mutate_inversion(child, probability):
    if probability >= np.random.uniform(0.0, 1.0):
        ind1 = np.random.random_integers(len(child)-1)
        ind2 = np.random.random_integers(len(child)-1)
        while ind1 == ind2:
            ind2 = np.random.random_integers(len(child)-1)
        if ind1 > ind2:
            child[ind2:ind1+1] = reversed(child[ind2:ind1+1])
        else:
            child[ind1:ind2+1] = reversed(child[ind1:ind2+1])


def parent_selector(parent_list, table, s):
    pre_prob = (2-s)/len(parent_list)
    parent1 = []
    parent2 = []
    for i in range(0, len(parent_list)-1):
        prob = pre_prob + (2 * (len(parent_list)-1-i) * (s-1))/(len(parent_list) * (len(parent_list) - 1))
        if prob >= np.random.uniform(0.0, 1.0):
            parent1 = parent_list[i]
            break

    for i in range(1, len(parent_list)-1):
        prob = pre_prob + (2*(len(parent_list) - 1 - i) * (s-1))/(len(parent_list)*(len(parent_list) - 1))
        if prob >= np.random.uniform(0.0, 1.0):
            parent2 = parent_list[i]
            break

    if len(parent1) == 0 and len(parent2) == 0:
        parent1 = parent_list[len(parent_list) - 2]
        parent2 = parent_list[len(parent_list) - 1]

    if len(parent1) == 0:
        parent1 = parent_list[len(parent_list) - 1]

    if len(parent2) == 0:
        parent2 = parent_list[len(parent_list) - 1]

    return list(parent1), list(parent2)

def survivor_selector_genitor(parent_list, child1, child2):
    parent_list[len(parent_list) - 1] = child1
    parent_list[len(parent_list) - 2] = child2


def genetic_algorithm(parent_list, table, s, mutation_prob, n_run):
    list.sort(parent_list, key=lambda seg: exhaustive.calcuate_total_distance(seg, table))
    best_individuals = []
    for c in range(n_run):

        parent1 = recomb.Genotype([])
        parent2 = recomb.Genotype([])
        child1 = recomb.Genotype([])
        child2 = recomb.Genotype([])

        parent1.data, parent2.data = parent_selector(parent_list, table, s)

        crossover_algorithm = recomb.Crossover()
        child1, child2 = crossover_algorithm.cycle_cross_over(parent1, parent2)
        mutate_inversion(child1.data, mutation_prob)
        mutate_inversion(child2.data, mutation_prob)
        survivor_selector_genitor(parent_list, child1.data, child2.data)
        list.sort(parent_list, key=lambda seg: exhaustive.calcuate_total_distance(seg, table))
        best_individuals.append(exhaustive.calcuate_total_distance(parent_list[0], table))

    return best_individuals

def run_algorithm(total_cities, population_size, n_rounds, s, mutation_prob, table, n_run, names):

    best_distance = 1000000
    worst_distance = 0
    best_distance_array = []
    best_individual_per_run = []
    best_gene_pool = []

    start = datetime.datetime.now()
    for n in range(0, n_rounds):
        # create initial gene pool
        gene_pool = []
        for i in range(0, population_size):
            gene_pool.append(np.random.permutation(total_cities))

        # fetching the strongest individual every algorithm cycle
        best_individual_per_run.append(genetic_algorithm(gene_pool, table, s, mutation_prob, n_run))

        # getting the strongest individual of this algorithm round
        tmp_individual = exhaustive.calcuate_total_distance(gene_pool[0], table)

        if tmp_individual < best_distance:
            best_distance = tmp_individual
            best_gene_pool = gene_pool[0]

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
    number_of_algorithm_runs = 2000



    fig = plt.figure("Genetic Algorithm")
    fig.suptitle("Average fitness of best fit individual in each generation")
    plt.ylabel("Distance")
    plt.xlabel("Number of generations")

    best_distance_matrix = []
    best_distance_matrix = run_algorithm(
        total_cities=24,
        population_size=10,
        n_rounds=20,
        s=1,
        mutation_prob=0.5,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names
    )

    best_distance_average1 = []
    for i in range(len(best_distance_matrix[0])):
        sum = 0
        for j in range(len(best_distance_matrix)):
            sum += best_distance_matrix[j][i]

        best_distance_average1.append(sum/len(best_distance_matrix))

    best_distance_matrix = run_algorithm(
        total_cities=24,
        population_size=50,
        n_rounds=20,
        s=1,
        mutation_prob=0.5,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names
    )


    best_distance_average2 = []
    for i in range(len(best_distance_matrix[0])):
        sum = 0
        for j in range(len(best_distance_matrix)):
            sum += best_distance_matrix[j][i]

        best_distance_average2.append(sum / len(best_distance_matrix))

    best_distance_matrix = run_algorithm(
        total_cities=24,
        population_size=100,
        n_rounds=20,
        s=1,
        mutation_prob=0.5,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names
    )

    run_algorithm(
        total_cities=10,
        population_size=10,
        n_rounds=20,
        s=1,
        mutation_prob=0.5,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names
    )

    run_algorithm(
        total_cities=10,
        population_size=50,
        n_rounds=20,
        s=1,
        mutation_prob=0.5,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names
    )

    run_algorithm(
        total_cities=10,
        population_size=100,
        n_rounds=20,
        s=1,
        mutation_prob=0.5,
        table=table,
        n_run=number_of_algorithm_runs,
        names=names
    )

    best_distance_average3 = []
    for i in range(len(best_distance_matrix[0])):
        sum = 0
        for j in range(len(best_distance_matrix)):
            sum += best_distance_matrix[j][i]

        best_distance_average3.append(sum / len(best_distance_matrix))

    plt.plot(range(number_of_algorithm_runs), best_distance_average1, 'b', label='Population size 10')
    plt.plot(range(number_of_algorithm_runs), best_distance_average2, 'r', label='Population size 50')
    plt.plot(range(number_of_algorithm_runs), best_distance_average3, 'g', label='Population size 100')


    plt.legend()
    if(len(sys.argv) > 2):
        plt.savefig(sys.argv[2] + ".pdf", format="pdf")
    f.close()





    # Choose three different values for population size
    # Report best, worst, average and standard deviation og 20 runs
    # Plot average fitness across generation of each run (for three population sizes)
    # Run for 10 and 24 cities