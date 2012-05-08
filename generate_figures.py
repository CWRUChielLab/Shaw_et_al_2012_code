#!/usr/bin/python

# Copyright (c) 2011, Kendrick Shaw
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# use TeX for labels
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])
from matplotlib import pyplot as plt

from scipy import integrate
from scipy import optimize
import numpy as np
import math
import multiprocessing
import tempfile

import iris
import prc

default_lambda = 2.
sample_a_vals = [1e-3, 0.1, 0.2, 0.24]

def nomenclature_fig():
    # set up the model parameters for this figure
    a = 0.1
    l_cw = -default_lambda
    l_ccw = 1
    X = Y = 1.
    def saddlefunc(y,t):
        return [y[0]*l_cw, y[1]*l_ccw]

    # create a new figure
    fig = plt.figure(figsize=(6,6))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # draw the axes
    iris.draw_saddle_neighborhood(axes, -X, -Y, 2*X, 2*Y, True, False,
            scale=0.7)

    axes.text(-1.05, 0.5, '$Y$',
            horizontalalignment = 'right', verticalalignment='center')
    axes.text(0.5, -1.05, '$X$',
            horizontalalignment = 'center', verticalalignment='top')

    # draw the stable limit cycle
    r0 = iris.iris_fixedpoint(a, l_ccw, l_cw, X, Y, guess=1e-6*X)
    vals = integrate.odeint(saddlefunc, [X,r0],
            np.linspace(0, iris.dwell_time(r0, l_ccw, l_cw, X, Y), 1000),
            )#args=(a, l_ccw, l_cw, X, Y))
    axes.plot(vals[:,0], vals[:,1], 'k', lw=2)

    # draw a sample trajectory
    x0 = [X, r0*2]
    vals = integrate.odeint(saddlefunc, x0,
            np.linspace(0, iris.dwell_time(x0[1], l_ccw, l_cw, X, Y), 1000),
            )#args=(a, l_ccw, l_cw, X, Y))
    axes.plot(vals[:,0], vals[:,1], 'b', lw=1)

    # center the plot and clean up the scale bars
    axes.set_xlim(-X, X)
    axes.set_ylim(-Y, Y)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_frame_on(False)
    return fig

def iris_fig(a, border=1., label_theta0=True):
    # set up the model parameters for this figure
    l_cw = -default_lambda
    l_ccw = 1
    X = Y = 1.

    # create a new figure
    fig = plt.figure(figsize=(5,5))
    axes = fig.add_axes([0., 0., 1., 1.])

    iris.draw_fancy_iris(axes, a, l_ccw, l_cw, X, Y)

    # add an arrow indicating theta = 0
    if label_theta0:
        r0s = iris.iris_fixedpoint(a, l_ccw, l_cw, X, Y, guess=1e-6*X)
        if r0s != None:
            axes.annotate(r'$\theta = 0$',
                xy=(a/2, -X + r0s - a/2), xycoords='data',
                xytext=(15,15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->',
                    connectionstyle='angle,angleA=180,angleB=240,rad=10')
                )
        r0u = iris.iris_fixedpoint(a, l_ccw, l_cw, X, Y, guess=1*X)
        if a != 0 and r0u != None:
            axes.annotate(r'',
                xy=(-X + r0u - a/2, -a/2), xycoords='data',
                xytext=(-15,15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='r',
                    connectionstyle='angle,angleA=-180,angleB=-45,rad=0')
                )
        if a != 0 and r0u != None:
            x0 = [-a/2, a/2 + Y - (0.9*r0u + 0.1*r0s)]
        elif a == 0:
            x0 = [-a/2, a/2 + Y - 0.9]
        else:
            x0 = [-a/2, 0.9*Y]
        axes.annotate(r'',
            xy=x0, xycoords='data',
            xytext=x0 + np.r_[-1.0e-3, 0.5e-3], textcoords='data',
            arrowprops=dict(arrowstyle='->', color='b',
                connectionstyle='arc3,rad=0')
            )

    # center the plot and clean up the scale bars
    axes.set_xlim(-2*X-border, 2*X+border)
    axes.set_ylim(-2*Y-border, 2*Y+border)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_frame_on(False)
    return fig

def sine_fig(mu = -0.1):
    alpha = 0.23333
    k = 1

    # create a new figure
    fig = plt.figure(figsize=(3.5,3.5))
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    #axes = fig.add_axes([0.1, 0.1, 0.89, 0.89])
    axes = fig.add_axes([0.07, 0.07, 0.92, 0.92])

    # draw the vector field
    X, Y = np.meshgrid(
            np.linspace(-math.pi, math.pi, 6*4+1)+ math.pi/(6*4),
            np.linspace(-math.pi, math.pi, 6*4+1)+ math.pi/(6*4)
            )
    (U,V) = iris.sine_system(np.asarray([X, Y]), 0., mu, alpha, k)
    axes.quiver(X, Y, U, V)

    # draw the x nullclines

    ys = xs = np.linspace(-math.pi, math.pi, 4*200)

    # Each x nullcline exists within a diagonal band;
    # calculate the band's height from the intersection of the nullclines
    # when mu = 0.
    h1 = 2 * math.asin(2*alpha) + math.pi
    h2 = math.pi - 2 * math.asin(2*alpha)

    nullx1 = np.array([
        optimize.brentq(
        lambda y : iris.sine_system(np.asarray([x,y]), 0., mu, alpha, k)[0],
        x + h1/2, x - h1/2
        ) for x in xs
        ])
    axes.plot(xs, nullx1, 'k')
    axes.plot(xs, nullx1 - 2*math.pi, 'k')
    axes.plot(xs, nullx1 + 2*math.pi, 'k')

    nullx2 = np.array([
        optimize.brentq(
        lambda y : iris.sine_system(np.asarray([x,y]), 0., mu, alpha, k)[0],
        x + math.pi + h2/2, x + math.pi - h2/2
        ) for x in xs
        ])
    axes.plot(xs, nullx2, 'k')
    axes.plot(xs, nullx2 - 2*math.pi, 'k')
    axes.plot(xs, nullx2 + 2*math.pi, 'k')

    # plot the y nullclines
    nully1 = np.array([
        optimize.brentq(
        lambda x : iris.sine_system(np.asarray([x,y]), 0., mu, alpha, k)[1],
        -y + h1/2, -y - h1/2
        ) for y in ys
        ])
    axes.plot(nully1, ys, 'k', dashes=(2,1))
    axes.plot(nully1 - 2*math.pi, ys, 'k', dashes=(2,1))
    axes.plot(nully1 + 2*math.pi, ys, 'k', dashes=(2,1))

    nully2 = np.array([
        optimize.brentq(
        lambda x : iris.sine_system(np.asarray([x,y]), 0., mu, alpha, k)[1],
        -y + math.pi + h2/2, -y + math.pi - h2/2
        ) for y in ys
        ])
    axes.plot(nully2, ys, 'k', dashes=(2,1))
    axes.plot(nully2 - 2*math.pi, ys, 'k', dashes=(2,1))
    axes.plot(nully2 + 2*math.pi, ys, 'k', dashes=(2,1))

    # draw the limit cycle and limit points
    iris.draw_fancy_sine_system(axes, mu, alpha, k, scale=1.)

    # add a trajectory
    x0 = [0, 0.1]
    vals = integrate.odeint(iris.sine_system, x0,
            np.linspace(0, 200, 20000),
            args=(mu, alpha, k))
    axes.plot(vals[:,0], vals[:,1], 'b', lw=1)

    # center the plot and clean up the scale bars
    border = 0.0
    axes.set_xlim(-math.pi/2-border, math.pi/2+border)
    axes.set_ylim(-math.pi/2-border, math.pi/2+border)
    plt.xticks(np.linspace(-2, 2, 5)*math.pi/2,
            [r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$',
                r'$\pi$'])
            #[r'$-\pi$', r'$-\pi/2$', '$0$', r'$\pi/2$', r'$\pi$'])
    plt.yticks(np.linspace(-2, 2, 5)*math.pi/2,
            [r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$',
                r'$\pi$'])


    return fig


def sine_prc_fig(mu_vals = [-1e-3, -0.1, -0.3, -0.40]):
    # set up the model parameters for this figure
    # mu = -0.2
    alpha = 0.23333
    k = 1
    n_phis = np.linspace(0, 2*math.pi, 4*20+1)
    dx = 1e-4
    dy = 0.
    mag = math.sqrt(dx**2 + dy**2)
    phasescale = 4 / (2 * math.pi) # convert from (0,2 \pi) to (0,4)

    # create a new figure
    fig = plt.figure(figsize=(6,6))

    width = 1./len(mu_vals)
    padding = 0.2*width

    for i in range(len(mu_vals)):
        mu = mu_vals[i]

        axes = plt.axes((2*padding, 1-(i+1) * width+padding,
            1 - width - 2*padding, width - 1.5*padding))

        # draw the orthogonal prc found numerically
        T, y0, error = iris.sine_limit_cycle(mu, alpha, k)
        n_prc_o = np.array([
            iris.sine_phase_reset(phi, dx=-dy, dy=dx, mu=mu, alpha=alpha, k=k,
                y0=y0, T=T,
                steps_per_cycle=100000)
            for phi in n_phis
            ])
        orthogonal_color = '0.8'
        axes.plot(n_phis, n_prc_o/mag*phasescale, '--', markersize=3,
                color=orthogonal_color)
        axes.plot(n_phis, n_prc_o/mag*phasescale, 'bo', markersize=3,
                color=orthogonal_color, markeredgecolor=orthogonal_color)
        axes.set_xlim(0, 2*math.pi)

        # draw the prc found numerically
        T, y0, error = iris.sine_limit_cycle(mu, alpha, k)
        n_prc = np.array([
            iris.sine_phase_reset(phi, dx=dx, dy=dy, mu=mu, alpha=alpha, k=k,
                y0=y0, T=T,
                steps_per_cycle=100000)
            for phi in n_phis
            ])
        axes.plot(n_phis, n_prc/mag*phasescale, '--k', markersize=3)
        axes.plot(n_phis, n_prc/mag*phasescale, 'bo', markersize=3)
        axes.set_xlim(0, 2*math.pi)
        plt.xticks(np.arange(5.)*math.pi/2,
                #['$0$', '$\\pi/2$', '$\\pi$',
                #    '$3\\pi/2$', '$2\\pi$'])
                # phase rescaled from 0 to 4 to match analysis
                ['$0$', '$1$', '$2$', '$3$', '$4$'])
        # make the y-axis symmetric around zero
        ymaxabs = np.max(np.abs(axes.get_ylim()))
        axes.set_ylim(-ymaxabs, ymaxabs)

        # draw the phase plot for reference
        axes = plt.axes((1-width+padding, 1-(i+1) * width + padding,
            width - 1.5 * padding, width - 1.5 * padding))
        iris.draw_fancy_sine_system(axes, mu, alpha, k, scale=3.)

        # center the plot and clean up the scale bars
        border = 0.2
        axes.set_xlim(-math.pi/2-border, math.pi/2+border)
        axes.set_ylim(-math.pi/2-border, math.pi/2+border)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_frame_on(False)

    return fig


def iris_prc_fig(a_vals = sample_a_vals, border = 0.3):
    # set up the model parameters for this figure
    l_cw = -default_lambda
    l_ccw = 1
    X = Y = 1.
    n_phis = np.linspace(0, 2*math.pi, 20*4 + 1)
    a_phis = np.linspace(0, 2*math.pi, 100*4 + 1)
    dx = 1e-4
    dy = 0.
    mag = math.sqrt(dx**2 + dy**2)
    phasescale = 4 / (2 * math.pi) # convert from (0,2 \pi) to (0,4)

    # create a new figure
    fig = plt.figure(figsize=(6,6))

    width = 1./len(a_vals)
    padding = 0.2*width

    for i in range(len(a_vals)):
        a = a_vals[i]

        axes = plt.axes((2*padding, 1-(i+1) * width+padding,
            1 - width - 2*padding, width - 1.5*padding))

        # draw the orthogonal prc found analytically
        a_prc_o = np.array([
            iris.analytic_phase_reset(phi, dx=-dy, dy=dx, a=a,
                l_ccw=l_ccw, l_cw=l_cw)
            for phi in a_phis
            ])
        axes.plot(a_phis, a_prc_o/mag*phasescale, color='0.8')

        # draw the orthogonal prc found numerically
        n_prc_o = np.array([
            iris.phase_reset(phi, dx=-dy, dy=dx, a=a, l_ccw=l_ccw, l_cw=l_cw,
                steps_per_cycle=100000)
            for phi in n_phis
            ])
        axes.plot(n_phis, n_prc_o/mag*phasescale, 'o', markersize=3,
                markeredgecolor='0.8', color='0.8')

        # draw the prc found analytically
        a_prc = np.array([
            iris.analytic_phase_reset(phi, dx=dx, dy=dy, a=a,
                l_ccw=l_ccw, l_cw=l_cw)
            for phi in a_phis
            ])
        axes.plot(a_phis, a_prc/mag*phasescale, 'k')

        # draw the prc found numerically
        n_prc = np.array([
            iris.phase_reset(phi, dx=dx, dy=dy, a=a, l_ccw=l_ccw, l_cw=l_cw,
                steps_per_cycle=100000)
            for phi in n_phis
            ])
        axes.plot(n_phis, n_prc/mag*phasescale, 'bo', markersize=3)

        # clean up the axes
        axes.set_xlim(0, 2*math.pi)
        plt.xticks(np.arange(5.)*math.pi/2,
                #['$0$', '$\\pi/2$', '$\\pi$',
                #    '$3\\pi/2$', '$2\\pi$'])
                # phase rescaled from 0 to 4 to match analysis
                ['$0$', '$1$', '$2$', '$3$', '$4$'])

        # make the y-axis symmetric around zero
        ymaxabs = np.max(np.abs(axes.get_ylim()))
        axes.set_ylim(-ymaxabs, ymaxabs)


        # draw the phase plot for reference
        axes = plt.axes((1-width, 1-(i+1) * width, width, width))
        iris.draw_fancy_iris(axes, a, l_ccw, l_cw, X, Y,
                scale=3.0, x0=np.nan)

        # center the plot and clean up the scale bars
        axes.set_xlim(-2*X-border, 2*X+border)
        axes.set_ylim(-2*Y-border, 2*Y+border)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_frame_on(False)

    return fig

def iris_isochron_fig(a, border=1., phase_offset=0., l = default_lambda):
    # set up the model parameters for this figure
    l_cw = -l
    l_ccw = 1
    X = Y = 1.

    # calculate the isochrons
    mesh_points = 400
    XX, YY = np.meshgrid(
            np.linspace(-(2*X + a/2), (2*X + a/2), mesh_points),
            np.linspace(-(2*Y + a/2), (2*Y + a/2), mesh_points),
            )
    isochrons = iris.iris_isochron(XX, YY, a=a, l_ccw=l_ccw,
            l_cw=l_cw, X=X, Y=Y)

    # create a new figure
    fig = plt.figure(figsize=(6,6))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    indexes = np.array([[i, j] for i in range(-2,3) for j in range(-2,3)])
    offsets = np.array([[i*2*X - a*j, j*2*Y + i*a] for i,j in indexes])

    for i in range(len(offsets)):
        if indexes[i].sum() % 2 == 0:
            iris.draw_fancy_iris(axes, a, l_ccw, l_cw, X, Y, x0=np.nan,
                    offset = offsets[i,:2])

    contours = phase_offset + np.linspace(
            -3*math.pi, 14*math.pi, 128, endpoint=False)
    cool_colors = [(0, 0.6 + 0.3*math.sin(c), 0.7 + 0.3*math.cos(c))
                for c in contours]
    hot_colors = [(0.6 + 0.3*math.cos(c), 0., 0.7 + 0.3*math.sin(c))
                for c in contours]

    # draw the other limit cycles
    for i in range(len(offsets)):
        if indexes[i].sum() % 2 == 0:
            axes.contour(XX + offsets[i, 0], YY + offsets[i, 1], isochrons,
                    contours,
                    colors=(cool_colors, hot_colors)[indexes[i, 0] % 2])

    # add lines at the edge of the squares to hide the contour discontinuity
    # where the phase wraps from 2 \pi to zero
    for i in range(len(offsets)):
        axes.plot([ a/2,  a/2] + offsets[i, 0],
                [-a/2, -(2*Y+a/2)] + offsets[i, 1], 'k')
        axes.plot([-a/2, -(2*X+a/2)] + offsets[i, 0],
                [-a/2, -a/2] + offsets[i, 1], 'k')

    # center the plot and clean up the scale bars
    axes.set_xlim(-3*X-border, 3*X+border)
    axes.set_ylim(-3*Y-border, 3*Y+border)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_frame_on(False)
    return fig


def iris_return_map_fig(a=0.2):
    l_cw = -default_lambda
    l_ccw = 1
    numpoints = 100
    y_i = np.r_[[[1]*numpoints],[np.linspace(0.,1.,numpoints)]].transpose()
    t, y_f = prc.vector_integrate_until_event(
            prc.f_saddle,
            prc.plane_crossing_test((0.,1.),1.),
            y_i, fparams=[-l_cw, l_ccw]
            )

    # create a new figure
    fig = plt.figure(figsize=(6,6))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    axes.plot(y_i[:,1], y_f[:,0] + a)
    axes.plot([0,1], [0,1])

    axes.set_ylim(0, 1)
    axes.set_xlim(0, 1)

    return fig

def iris_return_time_fig(a=0.2):
    l_cw = -default_lambda
    l_ccw = 1
    numpoints = 100
    y_i = np.r_[[[1]*numpoints],[np.linspace(0.,1.,numpoints)]].transpose()
    t, y_f = prc.vector_integrate_until_event(
            prc.f_saddle,
            prc.plane_crossing_test((0.,1.),1.),
            y_i, fparams=[-l_cw, l_ccw]
            )

    # create a new figure
    fig = plt.figure(figsize=(6,6))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    axes.plot(y_i[:,1], t)

    return fig

def iris_bifurcation_fig(L_max=20):
    numpoints = 100
    L = np.linspace(1 + 1e-13, L_max, numpoints)
    a_fold = L**(1./(1-L)) - L**(L/(1-L))

    # create a new figure
    fig = plt.figure(figsize=(6,6))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    dashed_points = numpoints/8
    axes.plot(L[:-dashed_points], a_fold[:-dashed_points], 'k', lw=2)
    axes.plot(L[-dashed_points:], a_fold[-dashed_points:], 'k--', lw=2)
    axes.plot([0,L_max], [0,0], 'k', lw=2)
    nudge_y = 1e-3
    axes.text(L[len(L)/2], a_fold[len(L)/2]/2, 'stable limit cycle',
            horizontalalignment = 'left', verticalalignment='center')
    axes.text(L[len(L)/8], 1./2 + a_fold[len(L)/2]/2,
            'trajectories spiral into center',
            horizontalalignment = 'left', verticalalignment='center')
    axes.annotate('fold bifurcation',
            xy=(L[len(L)/2], a_fold[len(L)/2]), xycoords='data',
            xytext=(15,-20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->',
                connectionstyle='angle,angleA=180,angleB=120,rad=10')
            )
    axes.annotate('heteroclinic orbit',
            xy=(L[len(L)/2], 0.+nudge_y), xycoords='data',
            xytext=(15,15), textcoords='offset points',
            arrowprops=dict(arrowstyle='->',
                connectionstyle='angle,angleA=180,angleB=240,rad=10')
            )
    #axes.plot(1, 0, 'ok')
    axes.annotate('neutrally stable orbits',
            xy=(1.+3*L_max*nudge_y, 0.+3*nudge_y), xycoords='data',
            xytext=(25,20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->',
                connectionstyle='angle,angleA=180,angleB=225,rad=10')
            )
    #axes.plot([default_lambda for a in sample_a_vals], sample_a_vals, 'ok')
    axes.set_xlim(0, L_max)
    axes.set_ylim(0, 1)
    axes.set_xlabel('$\lambda$')
    axes.set_ylabel('$a$')

    axes.set_xticks([0,1,L_max])
    return fig


def iris_timeplot_fig(a_vals = sample_a_vals, border = 0.3):
    # set up the model parameters for this figure
    l_cw = -default_lambda
    l_ccw = 1
    X = Y = 1.
    n_phis = np.linspace(0, 2*math.pi, 20*4 + 1)
    a_phis = np.linspace(0, 2*math.pi, 100*4 + 1)
    dx = 1e-4
    dy = 0.
    mag = math.sqrt(dx**2 + dy**2)
    phasescale = 4 / (2 * math.pi) # convert from (0,2 \pi) to (0,4)

    # create a new figure
    fig = plt.figure(figsize=(6,6))

    width = 1./len(a_vals)
    padding = 0.2*width

    for i in range(len(a_vals)):
        a = a_vals[i]

        axes = plt.axes((2*padding, 1-(i+1) * width+padding,
            1 - width - 2*padding, width - 1.5*padding))

        # draw the trajectory components vs. time
        r0 = iris.iris_fixedpoint(a, l_ccw, l_cw, X, Y, guess=1e-6*X)
        T = 4 * iris.dwell_time(r0, l_ccw, l_cw, X, Y)
        ts = np.linspace(0, 3*T, 1000);
        vals = integrate.odeint(iris.iris,
                [-a/2, -a/2 - Y + r0],
                ts,
                args=(a, l_ccw, l_cw, X, Y))
        axes.plot(ts, vals[:,1], '-', color='0.8', lw=2)
        axes.plot(ts, vals[:,0], 'k-', lw=2)

        axes.set_xlim(0, 3*T)

        # make the y-axis symmetric around zero
        #ymaxabs = np.max(np.abs(axes.get_ylim()))
        ymaxabs = 1.2
        axes.set_ylim(-ymaxabs, ymaxabs)


        # draw the phase plot for reference
        axes = plt.axes((1-width, 1-(i+1) * width, width, width))
        iris.draw_fancy_iris(axes, a, l_ccw, l_cw, X, Y,
                scale=3.0, x0=np.nan)

        # center the plot and clean up the scale bars
        axes.set_xlim(-2*X-border, 2*X+border)
        axes.set_ylim(-2*Y-border, 2*Y+border)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_frame_on(False)

    return fig


def sine_timeplot_fig(mu_vals = [-1e-3, -0.1, -0.3, -0.40]):
    # set up the model parameters for this figure
    # mu = -0.2
    alpha = 0.23333
    k = 1
    n_phis = np.linspace(0, 2*math.pi, 4*20+1)
    dx = 1e-4
    dy = 0.
    mag = math.sqrt(dx**2 + dy**2)
    phasescale = 4 / (2 * math.pi) # convert from (0,2 \pi) to (0,4)

    # create a new figure
    fig = plt.figure(figsize=(6,6))

    width = 1./len(mu_vals)
    padding = 0.2*width

    for i in range(len(mu_vals)):
        mu = mu_vals[i]

        axes = plt.axes((2*padding, 1-(i+1) * width+padding,
            1 - width - 2*padding, width - 1.5*padding))

        # draw the trajectory components vs. time
        T, y0, error = iris.sine_limit_cycle(mu, alpha, k)
        ts = np.linspace(0, 3*T, 1000);
        vals = integrate.odeint(iris.sine_system,
                [0, -y0],
                ts,
                args=(mu, alpha, k))
        axes.plot(ts, vals[:,1], '-', color='0.8', lw=2)
        axes.plot(ts, vals[:,0], 'k-', lw=2)

        axes.set_xlim(0, 3*T)

        # make the y-axis symmetric around zero
        #ymaxabs = np.max(np.abs(axes.get_ylim()))
        ymaxabs = 0.6*math.pi
        axes.set_ylim(-ymaxabs, ymaxabs)

        # draw the phase plot for reference
        axes = plt.axes((1-width+padding, 1-(i+1) * width + padding,
            width - 1.5 * padding, width - 1.5 * padding))
        iris.draw_fancy_sine_system(axes, mu, alpha, k, scale=3.)

        # center the plot and clean up the scale bars
        border = 0.2
        axes.set_xlim(-math.pi/2-border, math.pi/2+border)
        axes.set_ylim(-math.pi/2-border, math.pi/2+border)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_frame_on(False)

    return fig


def generate_figure(function, args, filenames, title="", title_pos=(0.5,0.95)):
    # workaround for python bug where forked processes use the same random
    # filename.
    tempfile._name_sequence = None;
    fig = function(*args)
    fig.text(title_pos[0], title_pos[1], title, ha='center')
    if type(filenames) == list:
        for name in filenames:
            fig.savefig(name)
    else:
        fig.savefig(filenames)



def main():
    # Old single-threaded code...
    #nomenclature_fig().savefig('nomenclature_fig.pdf')
    #iris_fig(0., 0.15).savefig('iris_a_fig.pdf')
    #iris_fig(0.05, 0.15).savefig('iris_b_fig.pdf')
    #iris_fig(0.2, 0.15).savefig('iris_c_fig.pdf')
    #iris_fig(0.255, 0.15).savefig('iris_d_fig.pdf')
    #iris_prc_fig().savefig('iris_prc_fig.pdf')
    #iris_isochron_fig(0.2, 0.15).savefig('iris_isochron_fig.pdf')
    #iris_return_map_fig(0.2).savefig('iris_return_map_fig.pdf')
    #iris_return_time_fig(0.2).savefig('iris_return_time_fig.pdf')
    #iris_bifurcation_fig().savefig('iris_bifurcation_fig.pdf')
    #sine_prc_fig().savefig('sine_prc_fig.pdf')
    #sine_fig(0.).savefig('sine_flow_shc.pdf')
    #sine_fig(-1e-2).savefig('sine_flow_lc.pdf')

    # function, args, filename
    figures = [
        (nomenclature_fig, [], ['nomenclature_fig.eps','nomenclature_fig.pdf']),
        (iris_fig, [0., 0.2], ['iris_a_fig.eps','iris_a_fig.pdf']),
        (iris_fig, [0.05, 0.2], ['iris_b_fig.eps','iris_b_fig.pdf']),
        (iris_fig, [0.2, 0.2], ['iris_c_fig.eps','iris_c_fig.pdf']),
        (iris_fig, [0.255, 0.2], ['iris_d_fig.eps','iris_d_fig.pdf']),
        (iris_prc_fig, [], ['iris_prc_fig.eps','iris_prc_fig.pdf']), # ~ 4:00.
        (iris_isochron_fig, [0.2, 0.15],
            ['iris_isochron_fig.eps','iris_isochron_fig.pdf']), # ~ 0:30.
        (iris_return_map_fig, [0.2],
            ['iris_return_map_fig.eps','iris_return_map_fig.pdf']),
        (iris_return_time_fig, [0.2],
            ['iris_return_time_fig.eps','iris_return_time_fig.pdf']),
        (iris_bifurcation_fig, [],
            ['iris_bifurcation_fig.eps','iris_bifurcation_fig.pdf']),
        (iris_timeplot_fig, [],
            ['iris_timeplot_fig.eps','iris_timeplot_fig.pdf']),
        (sine_prc_fig, [], ['sine_prc_fig.eps','sine_prc_fig.pdf']), # ~ 4:00.
        (sine_fig, [0.], ['sine_flow_shc.eps','sine_flow_shc.pdf']),
        (sine_fig, [-1e-2], ['sine_flow_lc.eps','sine_flow_lc.pdf']),
        (sine_timeplot_fig, [],
            ['sine_timeplot_fig.eps','sine_timeplot_fig.pdf']),
        ]

    # set up one process per figure
    processes = [multiprocessing.Process(target=generate_figure, args=args)
            for args in figures]

    # start all of the processes
    for p in processes:
        p.start()

    # wait for everyone to finish
    for p in processes:
        p.join()

    # run things sequentially to work around a ghostscript bug
    #for p in processes:
    #    p.start()
    #    p.join()

if __name__ == '__main__':
    main()
