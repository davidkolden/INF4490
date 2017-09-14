#x0 = -2: finner platået mellom x = -1 og x = 0.
#x0 = 1 : finner toppen rundt x = 2.

def f(x):
 return (-1)*(x**4) + 2*x**3 + 2*x**2 - x

def df(x):
 return (-4)*x**3 + 6*x**2 + 4*x - 1

def gradientAscent(x0):
    x = x0
    alpha = 0.001
    delta = 1
    i = 0
    iMax = 100
    print("x0 = " + str(x0))

    while i < iMax and delta > 10**(-5):
        xOld = x
        x += alpha*df(x)
        delta = ((xOld - x)**2)
        print("[" + str(i) + "]: y = " + str(f(x)) + " x = " + str(x))
        i += 1

if __name__ == "__main__":
    gradientAscent(3)

