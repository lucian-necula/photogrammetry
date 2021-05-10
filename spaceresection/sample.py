from imageresection import *

photo = np.array(
    [[86.421, -83.977],
     [-100.916, 92.582],
     [-98.322, -89.161],
     [78.812, 98.123]]
)

xp = photo[:, 0]
yp = photo[:, 1]

# focal length
f = 152.916

# control points coordinates
XYZ = np.array(
    [[1268.102,  1455.027, 22.606],
     [732.181, 545.344, 22.299],
     [1454.553, 731.666, 22.649],
     [545.245, 1268.232, 22.336]]
)

omega = 0
phi = 0
[xo, yo, zo, kappa] = aproxvalues(XYZ, xp, yp, f)

wpk = np.array([[omega, phi, kappa, xo, yo, zo]])

[Tx, Ty, Tz, w2, p2, k2, sigma, sigmaii, v] = imageresection(XYZ, xp, yp, wpk, f)

X0 = Tx
Y0 = Ty
Z0 = Tz

print('---------Parametrii de orientare exteriora----------')
print('omega: {0:.4f} +/- {1:.4f} grade'.format(w2, sigmaii[0]*180/pi))
print('phi: {0:.4f} +/- {1:.4f} grade'.format(p2, sigmaii[1]*180/pi))
print('kappa: {0:.4f} +/- {1:.4f} grade'.format(k2, sigmaii[2]*180/pi))
print('X0: {0:.4f} +/- {1:.4f} m'.format(X0, sigmaii[3]))
print('Y0: {0:.4f} +/- {1:.4f} m'.format(Y0, sigmaii[4]))
print('Z0: {0:.4f} +/- {1:.4f} m'.format(Z0, sigmaii[5]))

print('-------Rezidurile coordonatelor imagine x / y ---------')
for i in range(0, len(v), 2):
    print('{0:.4f} / {1:.4f}'.format(v[i][0], v[i+1][0]))


print('---------Deviatia standard-----------')
print('Std: {0:.4}'.format(sigma))
