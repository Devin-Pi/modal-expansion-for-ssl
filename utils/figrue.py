import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot2D_two(x1, y1, x2, y2, title, xlabel, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xlim(0, 250)
    plt.grid()
    # plt.show()
    return plt

def plot2D(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xlim(0, 250)
    plt.grid()
    # plt.show()
    return plt

def plot2D_semilogx(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(15, 5))
    plt.semilogx(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    # plt.show()
    return plt

def impulse_response():
    pass

def pressure_to_spl(pressure):
    # spl calculation
    spl = -20 * np.log10(np.abs(pressure))
    return spl