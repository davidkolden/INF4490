def exhaustive(f, step, start, stop):
    highest = -99999
    x = start
    while x < stop:
        y = f(x)
        if y > highest:
            highest = y

        print("x = " + str(x) + " y = " + str(y))
        x += step

    return highest

def f(x):
 return (-1)*(x**4) + 2*x**3 + 2*x**2 - x

if __name__ == "__main__":
    max = exhaustive(f, 0.5, -2, 3)
    print("highest = " + str(max))