#!env python
# -*- coding: utf-8 -*-
# Thin Plate Spline (2D, 3D) demo.
# reference: http://step.polymtl.ca/~rv101/thinplates/
import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ThinPlateSpline(object):
    u'''k-D Thin Plate Spline'''
    def __init__(self):
        pass
    def U(self, r):
        rsq = r**2.0
        if rsq == 0.0: return 0.0
        val = rsq*np.log(rsq)
        if np.isnan(val): return 0.0
        return val
    def fit(self, Xs, hs):
        u'''fit a thin plate spline on k dimentional space: R^k -> R
        @param[in] Xs (N samples * k dimention) input points
        @param[in] hs (N samples) interpolant
        '''
        self.Xs = Xs = np.array(Xs)
        self.hs = hs = np.array(hs).ravel()
        assert len(Xs.shape) == 2
        N, k = Xs.shape
        P = np.hstack([np.ones((N, 1)), Xs])
        K = np.zeros((N, N), np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                K[j, i] = K[i, j] = self.U(np.linalg.norm(P[i] - P[j]))
        L = np.vstack([
            np.hstack([K, P]),
            np.hstack([P.T, np.zeros((P.shape[1], P.shape[1]))]),
            ])
        Y = np.hstack([hs, np.zeros(k + 1)]).T
        # L * (W | a0 a1 a2 .. aN-1) = Y
        W_a = sp.linalg.solve(L, Y)
        self.W = W_a[:N]
        self.a_s = W_a[N:]
        self.P = P
        self.L = L
        self.k = k
        self.N = N

    def interpolate(self, Xs):
        Xs = np.array(Xs)
        N, k = self.Xs.shape
        if len(Xs.shape) != 2:
            Xs = Xs.reshape((-1, k))
        N_input, k_input = Xs.shape
        assert k_input == k
        P_input = np.hstack([np.ones((N_input, 1)), Xs]).astype(np.float32)
        res = (self.a_s*P_input).sum(axis=1)
        for i in range(N):
            if len(P_input) > 1:
                for j, row in enumerate(P_input):
                    res[j] += self.W[i]*self.U(np.linalg.norm(self.P[i] - row))
            else:
                res += self.W[i]*self.U(np.linalg.norm(self.P[i] - P_input))
        return res

class ThinPlateSpline2D(object):
    def __init__(self):
        pass
    def U(self, r):
        rsq = r**2.0
        if rsq == 0.0: return 0.0
        val = rsq*np.log(rsq)
        if np.isnan(val): return 0.0
        return val
    def fit(self, xs, ys, hs):
        self.xs = xs = np.array(xs).ravel()
        self.ys = ys = np.array(ys).ravel()
        self.hs = hs = np.array(hs).ravel()
        N = len(xs)
        P = np.vstack([np.ones_like(xs), xs, ys]).T
        K = np.zeros((N, N), np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                K[j, i] = K[i, j] = self.U(np.linalg.norm(P[i] - P[j]))
        L = np.vstack([
            np.hstack([K, P]),
            np.hstack([P.T, np.zeros((P.shape[1], P.shape[1]))]),
            ])
        Y = np.hstack([hs, np.zeros(3)]).T
        # L * (W | a0 a1 a2) = Y
        W_a0_a1_a2 = sp.linalg.solve(L, Y)
        self.W = W_a0_a1_a2[:N]
        self.a_s = W_a0_a1_a2[N:]
        self.P = P
        self.L = L

    def interpolate(self, x_or_xs, y_or_ys):
        a0, a1, a2 = self.a_s[0], self.a_s[1], self.a_s[2]
        x_or_xs = np.array(x_or_xs).ravel().astype(np.float32)
        y_or_ys = np.array(y_or_ys).ravel().astype(np.float32)
        res = a0 + a1*x_or_xs + a2*y_or_ys
        N = len(self.xs)
        P_input = np.vstack([np.ones_like(x_or_xs), x_or_xs, y_or_ys]).T
        for i in range(N):
            if len(P_input) > 1:
                for j, row in enumerate(P_input):
                    res[j] += self.W[i]*self.U(np.linalg.norm(self.P[i] - row))
            else:
                res += self.W[i]*self.U(np.linalg.norm(self.P[i] - P_input))
        return res

def thin_plate_spline_2d(use_2d_class):
    S = 30
    N = 10
    use_random = True
    if use_random:
        xs = range(S); np.random.shuffle(xs); xs = xs[:N]
        ys = range(S); np.random.shuffle(ys); ys = ys[:N]
        hs = np.random.randn(N)*0.5 + 1.0
    else:
        xs = [0, 0, S/3, S/2, S/2, S-1, S-1]
        ys = [0, S-1, S/3, S/4, S-1, S-1, 0]
        hs = [0.0, 0.0, 0.8, 0.9, 0.5, 0.0, 0.0]

    org = np.zeros((S, S))
    for x, y, h in zip(xs, ys, hs):
        org[int(x), int(y)] = h

    if use_2d_class:
        tps2d = ThinPlateSpline2D()
        tps2d.fit(xs, ys, hs)

        ixs, iys = np.mgrid[:S, :S]
        interpolated = tps2d.interpolate(ixs, iys).reshape((S, S))
        interpolation_diff = [tps2d.interpolate(x, y) - h for x, y, h in zip(xs, ys, hs)]
    else:
        tps2d = ThinPlateSpline()
        tps2d.fit(np.vstack([xs, ys]).T, hs)

        ixs, iys = np.mgrid[:S, :S]
        interpolated = tps2d.interpolate(np.vstack([ixs.ravel(), iys.ravel()]).T).reshape((S, S))
        interpolation_diff = [tps2d.interpolate([x, y]) - h for x, y, h in zip(xs, ys, hs)]
    print 'original point mean abs error', np.mean(np.abs(interpolation_diff))

    fig = plt.figure()
    r, c, i = 2, 2, 1
    cmap = 'jet'
    ax = fig.add_subplot(2, 2, i, projection='3d'); i += 1
    ax.scatter(xs, ys, hs)
    ax.set_aspect(1.0)
    ax.set_title('original')
    ax = fig.add_subplot(2, 2, i); i += 1
    ax.matshow(tps2d.L)
    ax.set_title('matrix L')
    ax = fig.add_subplot(2, 2, i); i += 1
    ax.matshow(org, vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_title('original')
    ax = fig.add_subplot(2, 2, i); i += 1
    ax.matshow(interpolated, vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_title('interpolated')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.suptitle('2D Thin Plate Spline: f: R^2 -> R (%s)' % tps2d.__class__.__name__)

def thin_plate_spline_3d():
    S = 5
    N = 30
    radius = S*np.sqrt(2.0)
    xs = np.linspace(-S, S, 2*N); np.random.shuffle(xs); xs = np.array(xs[:N]).astype(np.float32)
    ys = np.linspace(-S, S, 2*N); np.random.shuffle(ys); ys = np.array(ys[:N]).astype(np.float32)
    zs = np.linspace(-S, S, 2*N); np.random.shuffle(zs); zs = np.array(zs[:N]).astype(np.float32)
    Xs = np.vstack([xs, ys, zs]).T
    hs = np.random.randn(N)*0.5 + 1.0

    tps = ThinPlateSpline()
    tps.fit(Xs, hs)

    Ni = 8
    ixs, iys, izs = np.mgrid[:Ni, :Ni, :Ni]*2.0/Ni*S - S
    interpolated = tps.interpolate(np.vstack([ixs.ravel(), iys.ravel(), izs.ravel()]).T)
    interpolation_diff = [tps.interpolate([x, y, z]) - h for x, y, z, h in zip(xs, ys, zs, hs)]
    print 'original point mean abs error', np.mean(np.abs(interpolation_diff))

    fig = plt.figure()
    r, c, i = 2, 2, 1
    ax = fig.add_subplot(2, 2, i, projection='3d'); i += 1
    ax.scatter(xs, ys, zs, c=hs)
    ax.set_aspect(1.0)
    ax.set_title('original')
    ax = fig.add_subplot(2, 2, i); i += 1
    ax.matshow(tps.L)
    ax.set_title('matrix L')
    ax = fig.add_subplot(2, 2, i, projection='3d'); i += 1
    ax.scatter(ixs.ravel(), iys.ravel(), izs.ravel(), c=interpolated.ravel())
    fig.suptitle('3D Thin Plate Spline: f: R^3 -> R')

if __name__=='__main__':
    thin_plate_spline_2d(True)
    thin_plate_spline_2d(False)
    thin_plate_spline_3d()
    plt.show()

