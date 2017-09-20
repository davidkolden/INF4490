import csv
import sys
import itertools
import datetime
import matplotlib.pyplot as plt


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
    reader = csv.reader(f, delimiter=';')
    l = list(reader)
    names = l[0]
    l.pop(0)

    max_cities = range(0, 24)
    delta_t = []
    n = range(6, 11)

    for n_cities in range(6, 11):
        total_cities = max_cities[:n_cities]
        permus = list(itertools.permutations(total_cities, n_cities))
        start_t = datetime.datetime.now()
        winner_distance, winner_sequence = exhaustive_search(permus, l)
        end_t = datetime.datetime.now()
        total_t = end_t - start_t
        total = total_t.microseconds/1000000 + total_t.seconds
        delta_t.append(total)
        print("For n_cities = " + str(n_cities) + ":")
        print("Best distance: " + str(winner_distance))
        print("Best sequence: " + str(winner_sequence))
        print("Best order of travel:", end=" ")

        for i, val in enumerate(winner_sequence):
            print(names[val], end=" ")
        print(names[winner_sequence[0]])
        print(" ")

    fig = plt.figure("Exhaustive search")
    fig.suptitle("Time taken as function of how many cities visited")
    plt.ylabel("[s]")
    plt.xlabel("Number of cities")
    plt.plot(n, delta_t, 'ro')
    print("Time spent[seconds]:", end=" ")
    print(delta_t)
    if len(sys.argv) > 2:
        plt.savefig(sys.argv[2] + ".pdf", format="pdf")
        plt.show()
    else:
        plt.show()
    f.close()
