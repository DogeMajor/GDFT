import matplotlib.pyplot as plt
from gdft import *

import numpy as np

T = 2*np.pi


thetas2 = np.array([1.60572185, 1.74142426, 2.41460439, 2.82635103, 3.85204569, 3.85642993,
                    4.07064942, 5.5316874])


thetas3 = np.array([0.25459162, 2.04967221, 3.25969375, 3.96192626, 3.97595069, 4.52237336,
                    5.27781178, 6.09722023])

thetas4 = np.array([0.32736529, 0.58833421, 0.81941579, 3.96784893, 4.15335528,
                    4.42212678, 5.12125129, 5.86585884])


thetas5 = np.array([0.51715249, 0.85460258, 0.98599866, 2.01495428, 4.4041303,
                    4.98815164, 5.08690901, 5.67177735])

thetas6 = np.array([0.93482628, 1.07355423, 3.48993049, 3.8805253 , 3.96152232,
                    5.58643223, 6.04383695, 6.12363789])


plt.grid(True)

def rotate(thetas, deg_angle):
    angle = deg_angle*(np.pi/180)
    return (thetas - angle) % T

def order(thetas):
    return np.sort(thetas)

def _normalize(thetas):
    avg = thetas.mean()
    return thetas - avg

def _to_x(theta):
    return np.cos(theta)

def _to_y(theta):
    return np.sin(theta)

def to_coords(thetas):
    return _to_x(thetas), _to_y(thetas)

#Visually estimated how to rotate the angles so that they resemble the same structure
rot_thetas2 = order(rotate(thetas2, 65))
rot_thetas3 = order(rotate(thetas3, 195))
rot_thetas4 = order(rotate(thetas4, -133))
rot_thetas5 = order(rotate(thetas5, -115))
rot_thetas6 = order(rotate(thetas6, -177))

all_thetas = [rot_thetas2, rot_thetas3, rot_thetas4, rot_thetas5, rot_thetas6]

def generate_points(thetas):
    length = thetas.shape[0]
    args = np.array(list(range(length)))
    return args, thetas

def fit_polynome(thetas, grade):
    args, thetas = generate_points(thetas)
    z = np.polyfit(args, thetas, grade)
    f = np.poly1d(z)
    return f

def plot_fitted_polynome(thetas, grade):
    args, thetas = generate_points(thetas)
    pol_fn = fit_polynome(thetas, grade)
    x_new = np.linspace(args[0], args[-1], 50)
    y_new = pol_fn(x_new)
    plt.plot(args, thetas, 'o', x_new, y_new)


def plot_angles(args, thetas):
    plt.plot(args, thetas, 'x')

def polar_plot_angles(thetas):
    x, y = to_coords(thetas)
    plt.plot(x, y, 'o')


#plt.plot(args, rot_thetas2, 'x')
from datetime import datetime
#plt.savefig("4_"+str(datetime.now())[:10]+".png")

def plot_fn(fn, dim):
    x_new = np.linspace(0, dim-1, 50)
    y_new = fn(x_new)
    plt.plot(x_new, y_new)

def cand_fn(x, dim=8):
    return 0.5*np.pi*(1 + np.sin(np.pi*(x-dim/2)/(dim)))

def cand_fn2(x, dim=8):
    return 0.5*np.pi*(1 + np.sin(np.pi*(x+1-dim/2)/(dim-1)))

if __name__ == "__main__":
    #polar_plot_angles(thetas2)
    #polar_plot_angles(rot_thetas6)

    for thetas in all_thetas[2:]:
        plot_fitted_polynome(thetas, 3)


    plot_fn(cand_fn2, 8)

    plt.show()
