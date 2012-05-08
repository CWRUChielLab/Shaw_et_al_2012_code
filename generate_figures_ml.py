#!/usr/bin/python

# Copyright (c) 2011, Youngmin Park, Kendrick Shaw
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

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'], size=10)
from matplotlib import pyplot as plt

import multiprocessing
import numpy as np

from lib_ml import *

from generate_figures import generate_figure

def trajectory_fig(dy_dt, y0, axes, T, iapp=None):
    sol = integrate_ml(dy_dt, y0, iapp=iapp, max_time=T)
    #axes = plt.axes([.1, .1, .8, .8])
    axes.plot(sol[:,0], sol[:,1], 'k-', lw=2)

    # draw saddle and unstable focus
    saddle = saddle_point(iapp)
    unstable_pt = unstable_focus(iapp)

    axes.scatter(unstable_pt[0], unstable_pt[1], facecolors='none', s=15)
    axes.scatter(saddle[0], saddle[1], facecolors='none', s=15)

    axes.set_xlim(-25, 20)
    axes.set_ylim(-.08, .6)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_frame_on(False)


def trajectory_fig_transformed(dy_dt, y0, iapp=None):
    sol = integrate_ml(dy_dt, y0, iapp=iapp, max_time=1600)
    transformed_sol = coordinate_transform(sol)

    fig = plt.figure()
    axes = plt.axes([.1, .1, .8, .8])
    axes.plot(transformed_sol[:,0], transformed_sol[:,1], '-', lw=2)

    axes.set_xlim(-1, 30)
    axes.set_ylim(-1, 30)
    #axes.set_xticks([])
    #axes.set_yticks([])
    axes.set_frame_on(False)

    return fig


def transformed_displacement(dy_dt, y0, iapp=None):
    max_time = 400
    max_steps = 8000
    t = np.linspace(0, max_time, max_steps)
    sol = integrate_ml(dy_dt, y0, max_time=max_time, max_steps=max_steps, iapp=iapp)
    transformed_sol = coordinate_transform(sol)

    fig = plt.figure(figsize=(6,6))
    axes = plt.axes([.1, .1, .8, .8])
    axes.plot(transformed_sol, '-', lw=2)

    #axes.plot(t, transformed_sol[:,0], '-', lw=2)
    #axes.plot(t, transformed_sol[:,1], '-', lw=2)

    axes.set_xlim(0, max_steps)
    axes.set_ylim(0, 35)
    #axes.set_xticks([])
    #axes.set_yticks([])
    axes.set_frame_on(False)

    return fig


def lc_fig(dy_dt, y0, iapp=None):
    max_time = 800.
    max_steps = 800000.
    period, init, err, lc_max = ml_limit_cycle(dy_dt, y0, iapp=iapp, max_time=max_time, max_steps=max_steps)

    t = np.linspace(0, max_time, max_steps)
    v_coord = brentq(ml_limit_cycle_exact, init[0]-1e-6, init[0]+1e-6, args=(init[1], iapp))
    print "V coordinate is", v_coord, " for iapp="+str(iapp)+", hmax="+str(hmax)+", max_time and max_steps default."

    max_steps = (period/max_time)*(max_steps)
    sol = integrate_ml(dy_dt, init, axes, max_time=period*1.5, max_steps=max_steps, iapp=iapp)

    transformed_sol = sol
    fig = plt.figure()
    axes = plt.axes([.1, .1, .8, .8])
    axes.plot(transformed_sol[:,0], transformed_sol[:,1], '-', color='0', lw=2)
    axes.set_xlim(-25, 20)
    axes.set_ylim(0, .45)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_frame_on(False)

def lc_transformed_fig(dy_dt, y0, iapp=None):
    max_time = 800
    max_steps = 400000
    period, init, err, lc_max = ml_limit_cycle(dy_dt, y0, iapp=iapp, max_time=max_time, max_steps=max_steps)

    max_steps = (period/800.)*(max_steps)
    sol = integrate_ml(dy_dt, init, max_time=period*1.5, max_steps=max_steps, iapp=iapp)

    transformed_sol = coordinate_transform(sol)
    fig = plt.figure()
    axes = plt.axes([.1, .1, .8, .8])
    axes.plot(transformed_sol[:,0], transformed_sol[:,1], '-', lw=2)
    axes.set_xlim(-1, 30)
    axes.set_ylim(-1, 30)
    #axes.set_xticks([])
    #axes.set_yticks([])
    axes.set_frame_on(False)

    return fig


def displacement(dy_dt, y0, iapp_vals, max_time=400, max_steps=400000):

    fig = plt.figure(figsize=(6,6))

    width = 1./len(iapp_vals)
    padding = 0.23*width

    for i in range(len(iapp_vals)):
        print "iteration", i
        iapp = iapp_vals[i]

        axes2 = plt.axes((2*padding, 1-(i+1) * width+padding,
            1 - width - 2*padding, width - 1.5*padding))

        T, init, err, lc_max = ml_limit_cycle(dy_dt, y0, iapp=iapp)
        t = np.linspace(0, 3*T, (3.*T/400.)*max_steps)
        sol = integrate_ml(dy_dt, init, max_time=3*T, max_steps=(3.*T/400.)*max_steps, iapp=iapp)

        axes2.plot(t, sol[:,0], 'k-', lw=2)
        #axes2.set_ylabel('w')

        axes1 = axes2.twinx()
        axes1.plot(t, sol[:,1], '-', color='.8', lw=2)

        axes1.set_xlim(0, 3*T)
        axes1.set_ylim(0, .45)

        axes2.set_xlim(0, 3*T)
        axes2.set_ylim(-25, 20)

        axes2 = plt.axes((1-width+padding, 1-(i+1) * width + padding,
            width - 1.5 * padding, width - 1.5 * padding))

        trajectory_fig(dy_dt, init, axes2, T, iapp=iapp)

    return fig



def prc_fig(number_of_perts, iapp_vals, dx_vals, dy=0.0):
    fig = plt.figure(figsize=(6,6))
    width = 1./len(iapp_vals)
    padding = 0.2*width

    for i in range(len(iapp_vals)):
        iapp = iapp_vals[i]
        dx = dx_vals[i]
        n_phis = np.linspace(0,1,number_of_perts)
        axes = plt.axes((2*padding, 1-(i+1) * width+padding,
            1 - width - 2*padding, width - 1.5*padding))

        n_prc, err = ml_phase_reset(n_phis, iapp=iapp, dx=dx, hmax=1e-2)

        if dx != 0.0:
            n_prc = n_prc/dx

        print n_prc

        # draw numerical PRC
        axes.plot(n_phis, n_prc, 'bo', markersize=3)
        axes.plot(n_phis, n_prc, 'b--', markersize=3)

        T, init, err, lc_max = ml_limit_cycle(dy_dt, [0,.089], iapp=iapp)

        axes = plt.axes((1-width+padding, 1-(i+1) * width + padding,
            width - 1.5 * padding, width - 1.5 * padding))

        trajectory_fig(dy_dt, init, axes, T, iapp=iapp)



    return fig


def function1(max_steps, step_size):
    fig = plt.figure()
    return fig

def function2(max_steps, step_size):
    fig = plt.figure()
    return fig

def function3(max_steps, step_size):
    fig = plt.figure()
    return fig

#def generate_figure(function, args, filename, title="", title_pos=(0.5,0.95)):
#    fig = function(*args)
#    fig.text(title_pos[0], title_pos[1], title, ha='center')
#    fig.savefig(filename)
#    return fig

# No need to generate figures for untransformed space
#(trajectory_fig, [dy_dt, [0,.1], None], 'untransformed_trajectory_iNone.png')
#(trajectory_fig, [dy_dt, [0,.1], 35.5], 'untransformed_trajectory_i35.5.png')
#(trajectory_fig, [dy_dt, [0,.1], 37], 'untransformed_trajectory_i37.png')
#(trajectory_fig, [dy_dt, [0,.1], 38.9], 'untransformed_trajectory_i38.9.png')


#        (prc_fig, [20, homoclinic, 1e-6], 'prc_i'+str(homoclinic)+'_dx1e-6.png'),
#        (prc_fig, [20, 35.5, 1e-6], 'prc_i38.9_dx1e-6.png'),
#        (prc_fig, [20, 37, 1e-6], 'prc_i38.9_dx1e-6.png'),
#        (prc_fig, [20, 38.9, 1e-6], 'prc_i38.9_dx1e-6.png'),
#        (prc_fig, [100, 38.9, 1e-6], 'prc_i38.9_dx1e-6.png')

#        (lc_fig, [dy_dt, [0,.1], homoclinic], 'transformed_lc_i'+str(homoclinic)+'.png'),
#        (lc_fig, [dy_dt, [0,.1], 35.5], 'transformed_lc_i35.5.png'),
#        (lc_fig, [dy_dt, [0,.1], 37], 'transformed_lc_i37.png'),
#        (lc_fig, [dy_dt, [0,.1], 38.9], 'transformed_lc_i38.9.png'),
#        (lc_fig, [dy_dt, [0,.1], 40], 'transformed_lc_i40.png'),

#        (trajectory_fig_transformed, [dy_dt, [0,.1], homoclinic], 'transformed_trajectory_i'+str(homoclinic)+'.png'),
#        (trajectory_fig_transformed, [dy_dt, [0,.1], 35.5], 'transformed_trajectory_i35.5.png'),
#        (trajectory_fig_transformed, [dy_dt, [0,.1], 37], 'transformed_trajectory_i37.png'),
#        (trajectory_fig_transformed, [dy_dt, [0,.1], 38.9], 'transformed_trajectory_i38.9.png'),

def main():
    homoclinic = 35.009 #35.00675
    #homoclinic, 35.1, 36.5, 40.0
    #homoclinic, 35.05, 36.0, 40.0
    figures = [
        (displacement, [dy_dt, [0,.089], [homoclinic, 35.05, 36.2, 40.0]], 
            ['morris-lecar_time_plots.eps','morris-lecar_time_plots.pdf']),
        (prc_fig, 
            [75, [homoclinic, 35.05, 36.2, 40.0], [1e-4, 1e-4, 1e-4, 1e-2]], 
            ['morris-lecar_prcs.eps','morris-lecar_prcs.pdf']) 
            #100 default points
        ]


    #for args in figures:
    #    generate_figure(args[0], args[1], args[2])

    # p.start() works fine, but no plots are generated, even after
    # (or during?)  p.join()
    processes = [multiprocessing.Process(target=generate_figure, args=args)
            for args in figures]

    # start all of the processes
    for p in processes:
        p.start()

    # wait for everyone to finish
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
