class Genotype:
    def __init__(self, init_data):
        self.data = init_data
        self._size = len(init_data)

    def set_segment(self, seg1, seg2):
        if seg1 > seg2:
            self.segment_index[0] = seg2
            self.segment_index[1] = seg1
        else:
            self.segment_index[0] = seg1
            self.segment_index[1] = seg2

    def print_data(self):
        print(self.data)

    _size = 0

    data = []

    segment_index = [0]*2


class Crossover:
    def __init__(self):
        pass

    def pmx(self, parent1, parent2):
        child = Genotype([None]*(len(parent1.data)))
        for i in range(parent1.segment_index[0], parent1.segment_index[1] + 1):
            child.data[i] = parent1.data[i]
        child = self._parent_segment_search(parent2, child)
        child = self._copy_remainding_values(parent2, child)
        return child

    def order_cross_over(self, parent1, parent2):
        child = Genotype([None] * (len(parent1.data)))
        for i in range(parent1.segment_index[0], parent1.segment_index[1] + 1):
            child.data[i] = parent1.data[i]
        child = self._copy_values_after_segment(parent2, child)
        return child

    def cycle_cross_over(self, parent1, parent2):
        child1 = Genotype([None] * (len(parent1.data)))
        child2 = Genotype([None] * (len(parent1.data)))
        listOfCycles = self._generate_cycles(parent1, parent2)
        child1 = self._insert_cycle(child1, parent1, listOfCycles[0])
        child2 = self._insert_cycle(child2, parent2, listOfCycles[0])
        for i in range(1, len(listOfCycles)):
            if i%2 != 0:
                child1 = self._insert_cycle(child1, parent2, listOfCycles[i])
                child2 = self._insert_cycle(child2, parent1, listOfCycles[i])
            else:
                child1 = self._insert_cycle(child1, parent1, listOfCycles[i])
                child2 = self._insert_cycle(child2, parent2, listOfCycles[i])
        return child1, child2



    def _parent_segment_search(self, parent, child):
        for i in range(parent.segment_index[0], parent.segment_index[1] + 1):
            val = parent.data[i]
            if val not in child.data:
                child = self._find_value_and_insert(val, i, parent, child)

        return child

    def _find_value_and_insert(self, val, pos, parent, child):
        new_pos = parent.data.index(child.data[pos])
        if child.data[new_pos] is None:
            child.data[new_pos] = val
        else:
            child = self._find_value_and_insert(val, new_pos, parent, child)
        return child

    def _copy_remainding_values(self, parent, child):
        for i in range(0, len(parent.data)):
            if parent.data[i] not in child.data:
                child.data[i] = parent.data[i]
        return child

    def _copy_values_after_segment(self, parent, child):
        i = parent.segment_index[1] + 1
        i = self._get_next_free_position(child, i)
        for j in range(parent.segment_index[1] + 1, len(parent.data)-1):
            if parent.data[j] not in child.data:
                child.data[i] = parent.data[j]
                i = self._get_next_free_position(child, i)
        for k in range(0, parent.segment_index[1] + 1):
            if parent.data[k] not in child.data:
                child.data[i] = parent.data[k]
                i = self._get_next_free_position(child, i)
        return child

    def _get_next_free_position(self, child, i):
        counter = 0
        while child.data[i] is not None and counter < len(child.data):
            i += 1
            counter += 1
            if i >= len(child.data):
                i = 0
        return i

    def _generate_cycles(self, parent1, parent2):
        nCycle = 0
        pos = 0
        counter = 0
        listOfCycles = []
        anelelist = [None]*(len(parent1.data))
        totallist = [None]*(len(parent1.data))
        while counter < len(parent1.data):
            element = parent1.data[pos]
            newElement = None
            while element != newElement:
                anelelist[pos] = 1
                totallist[pos] = 1
                newElement = parent2.data[pos]
                counter += 1
                pos = parent1.data.index(newElement)
            listOfCycles.append(anelelist)
            nCycle += 1
            for i in range(0, len(totallist)-1):
                if totallist[i] is None:
                    pos = i
                    break
            anelelist = [None] * (len(parent1.data))
        return listOfCycles

    def _insert_cycle(self, child, parent, cycle):
        for i in range(0, len(parent.data)):
            if cycle[i] == 1:
                child.data[i] = parent.data[i]
        return child


if __name__ == '__main__':

    parent1 = Genotype([1, 2, 3, 4, 5, 6, 7, 8, 9])
    parent1.set_segment(3, 6)
    parent2 = Genotype([9, 3, 7, 8, 2, 6, 5, 1, 4])
    parent1.set_segment(3, 6)
    crossover = Crossover()
    print("PMX:")
    child1 = crossover.pmx(parent1, parent2)
    child1.print_data()
    child2 = crossover.pmx(parent2, parent1)
    child2.print_data()
    print(" ")

    child1 = crossover.order_cross_over(parent1, parent2)
    print("Order crossover:")
    child1.print_data()
    print(" ")

    child1, child2 = crossover.cycle_cross_over(parent1, parent2)
    print("Cycle crossover:")
    child1.print_data()
    child2.print_data()