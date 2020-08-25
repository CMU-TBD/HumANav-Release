import numpy as np
from metrics.cost_utils import *


def asym_gauss_from_vel(x, y, velx, vely, xc=0, yc=0):
    """
    computation of the value of an arbitrarily rotated (by theta)
    centered at (xc, yc)
    Asymmetric Gaussian at some point (x, y)
    Obviously, the velocities are for the peds
    around whom the gaussian is centered
    Variances are:
    sig_theta: in direction of motion
    sig_r: opp direction of motion (rear)
    sig_s: variance to the sides

    can calculate sig_theta = max(2*velocity, 0.5) [Rachel Kirby thesis 2005?]
    """
    speed = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(vely, velx)
    sig_theta = vel2sig(speed)
    return asym_gauss(x, y, theta, sig_theta, xc=xc, yc=yc)


def asym_gauss(x, y, theta=0, sig_theta=2, xc=0, yc=0):
    """
    computation of the value of an arbitrarily rotated (by theta)
    centered at (xc, yc)
    Asymmetric Gaussian at some point (x, y)
    Variances are:
    sig_theta: in direction of motion
    sig_r: opp direction of motion (rear)
    sig_s: variance to the sides

    can calculate sig_theta = max(2*velocity, 0.5) [Rachel Kirby thesis 2005?]
    """
    alpha = np.arctan2(y - yc, x - xc) - theta + np.pi / 2
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    # print(alpha[np.where(alpha>np.pi)])
    # sigma = np.zeros_like(x)
    # sigma = sig_r if alpha <= 0 else sig_h

    sig_s = sig_theta / 4
    sig_r = sig_theta / 3
    sigma = np.where(alpha <= 0, sig_r, sig_theta)

    a = ((np.cos(theta) / sigma) ** 2 + (np.sin(theta) / sig_s) ** 2) / 2
    b = np.sin(2 * theta) * (1 / (sigma ** 2) - 1 / (sig_s ** 2)) / 4
    c = ((np.sin(theta) / sigma) ** 2 + (np.cos(theta) / sig_s) ** 2) / 2

    agxy = np.exp(-(a * (x - xc) ** 2 + 2 * b * (x - xc) * (y - yc) + c * (y - yc) ** 2))

    return agxy
