import numpy as np
import matplotlib.pyplot as plt

def f(x):
 return (-1)*(x**4) + 2*x**3 + 2*x**2 - x

def df(x):
 return (-4)*x**3 + 6*x**2 + 4*x - 1

#def f(t):
#    return t**3

#def df(t):
#    return 3*t**2

def subplot():

 x = np.linspace(-2, 3, 100)
 fig = plt.figure("Plot: f(x) and df(x)")
 
 plt.subplot(211)
 plt.plot(x,f(x))
 plt.subplot(212)
 plt.plot(x,df(x))
 plt.show()

if __name__ == "__main__":
    subplot()
