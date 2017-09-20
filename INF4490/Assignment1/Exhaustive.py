import csv
import sys
import itertools
# import timeit


def exhaustive_search(perm_list, table):
    best_distance = 10000000
    best_order = []

    for seg in perm_list:
        total = 0
        for i, val in enumerate(seg[:(len(seg)-1)]):
           total += float(table[val][seg[i + 1]])
        total += float(table[seg[len(seg) - 1]][seg[0]])
        if total < best_distance:
            best_distance = total
            best_order = seg
    return best_distance, best_order


if __name__ == '__main__':
    f = open(sys.argv[1], 'r')

    max_cities = range(0, 24)
    n_cities = int(sys.argv[2])
    total_cities = max_cities[:n_cities]
    permus = list(itertools.permutations(total_cities, n_cities))
    print("Number of permutations: " + str(len(permus)))
    reader = csv.reader(f, delimiter=';')
    l = list(reader)
    names = l[0]
    l.pop(0)

    winner_distance, winner_sequence = exhaustive_search(permus, l)
    print("Best distance: " + str(winner_distance))
    print("Best sequence: " + str(winner_sequence))
    print("Best order of travel:", end=" ")
    for i, val in enumerate(winner_sequence):
        print(names[val], end=" ")
    print(names[winner_sequence[0]])
    f.close()
