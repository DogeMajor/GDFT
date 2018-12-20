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


#-----------only values between 0 and pi are allowed in the following theta vecs-------------

thetas7 = np.array([0.15847839, 0.32878537, 0.43786295, 1.21297503, 2.04867586,
                    2.80857661, 2.98617521, 3.14135541])

thetas8 = np.array([0.30030045, 0.39933246, 0.40867739, 1.28074787, 2.28478356,
                    2.94454651, 3.00267416, 3.11595278])

thetas9 = np.array([0.31230405, 0.40569868, 0.40757623, 1.28524824, 2.30053787,
                    2.94532072, 2.99596616, 3.13543097])

thetas10 = np.array([0.08858783, 0.22655229, 0.35270027, 1.14690321, 2.03414003,
                     2.82637154, 2.84963627, 2.98563405])

thetas11 = np.array([1.50625972e-05, 1.37137054e-01, 2.60749707e-01, 1.05664117e+00,
                     1.94641101e+00, 2.73948941e+00, 2.75852795e+00, 2.89283675e+00])

thetas12 = np.array([0.09668558, 0.21427694, 0.27117106, 0.9284712, 1.9343563,
                     2.80765764, 2.815155, 2.9135714])

thetas13 = np.array([5.14709405e-04, 1.19545652e-01, 1.76027032e-01, 8.32504730e-01,
                     1.83900738e+00, 2.71271675e+00, 2.71960025e+00, 2.81781812e+00])



#---unordered thetas between 0 and pi-----------------

thetas14 = np.array([0.05249126, 2.72382596, 0.26529619, 2.00643631, 2.96519007,
                     3.14156252, 1.23441539, 0.48058796])

thetas15 = np.array([0.1042914, 0.94738587, 2.9437974, 2.85669262, 1.98720398,
                     0.33532913, 2.88312519, 0.30104954])

thetas16 = np.array([0.15212298, 1.00405324, 3.00929904, 2.93102814, 2.07037379,
                     0.42733653, 2.98396613, 0.41072977])

thetas17 = np.array([0.39839151, 1.19332787, 3.14159265, 3.00633618, 2.08870024,
                     0.38866847, 2.8883232, 0.25809249])

thetas18 = np.array([0.40953218, 2.99805248, 0.45673289, 2.1150655, 2.99101179,
                     3.08456032, 1.09461761, 0.25798233])

thetas19 = np.array([2.78752335, 2.01232491, 0.08380702, 0.23881077, 1.17619691,
                     2.89597022, 0.41607719, 3.06603981])

thetas20 = np.array([2.71203408, 0.13401165, 2.68585628, 1.03803098, 0.17259165,
                     0.0895361, 2.08999672, 2.93713857])

thetas21 = np.array([0.06428542, 0.89823397, 2.88550414, 2.78925131, 1.91062375,
                     0.24960325, 2.78825635, 0.1970401])

thetas22 = np.array([0, 0.87459821, 2.90250666, 2.84689823, 2.00891082,
                     0.38853476, 2.96783281, 0.41726085])

thetas23 = np.array([0.19618612, 1.03911766, 3.03537403, 2.94810336, 2.07845692,
                     0.42642006, 2.97405462, 0.39182177])

unordered_thetas = [thetas14, thetas15, thetas16, thetas17, thetas18, thetas19, thetas20,
                    thetas21, thetas22, thetas23]

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

def rotate_to_center(thetas, deg_angle):
    angle = deg_angle * (np.pi / 180)
    return thetas - angle

#Visually estimated how to rotate the angles so that they resemble the same structure
rot_thetas2 = order(rotate(thetas2, 65))
rot_thetas3 = order(rotate(thetas3, 195))
rot_thetas4 = order(rotate(thetas4, -133))
rot_thetas5 = order(rotate(thetas5, -115))
rot_thetas6 = order(rotate(thetas6, -177))

all_thetas = [rot_thetas2, rot_thetas3, rot_thetas4, rot_thetas5, rot_thetas6]
limited_thetas = [rotate_to_center(thetas7, 5), rotate_to_center(thetas8, 8), rotate_to_center(thetas9, 9),
                  rotate_to_center(thetas10, -2), rotate_to_center(thetas11, -8), rotate_to_center(thetas12, -3),
                  rotate_to_center(thetas13, -9)]

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

def plot_polynome_roots(polynome):
    coeffs = polynome.r
    x, y = coeffs.real, coeffs.imag
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
    '''for thetas in unordered_thetas:
        polar_plot_angles(thetas)'''

    #polar_plot_angles(limited_thetas[6])

    #for thetas in unordered_thetas:
    #    plot_fitted_polynome(thetas, 7)

    #plot_fitted_polynome(unordered_thetas[0], 7)


    for thetas in unordered_thetas:
        polynome = fit_polynome(thetas, 7)
        plot_polynome_roots(polynome)

    #plot_polynome_roots(polynome)

    '''plot_fn(cand_fn2, 8)
    for thetas in unordered_thetas:
        print(thetas)
        #print(16*(1/np.pi)*(thetas-0.5*np.pi))'''
    plt.show()
