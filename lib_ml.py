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

import sys
import math
import numpy as np
import matplotlib.pylab as mp
import numpy.linalg as linalg

#sys.path.append("../../../../papers/shc/iris")
#sys.path.append("../../../../papers/shc/siamds-iris")
#import prc

from scipy import integrate
from scipy.optimize import brentq, fsolve

#parameters from Ermentrout and Terman 2010 Pg 51, chapter on Morris-Lecar model.
def return_params(iapp=None): #35.0067362
	if iapp == None:
		iapp = 35.1
	assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
	ep,iapp,vinit,winit = [20,iapp,-10,0.1] #35.00067369
	gl,el,ek,gk = [2,-60,-84,8]
	gca,v1,v2,v3 = [4,-1.2,18,12]
	v4,eca,phi = [17.4,120,0.23]
	return ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi


# Place ML functions here
def nullcline1(v, iapp=None):
	assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
	ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params(iapp)
	return (-eca*gca - 2.0*el*gl - 2.0*iapp + gca*v +\
                2.0*gl*v - eca*gca*np.tanh((v - v1)/v2) +\
                gca*v*np.tanh((v - v1)/v2))/(2.0*gk*(ek - v))

def nullcline2(v, iapp=None):
	assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
	ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params(iapp)
	return (0.5 * (1.0 + np.tanh((v - v3)/v4)))

# Taken from pg 50 Ermentrout and Terman Foundations of Neuroscience
def dy_dt(y,t, iapp=None, dv=None):
	assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
	ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params(iapp)
	return np.array([(-gca*0.5*(1.0 + np.tanh((y[0] - v1)/v2))*(y[0] - eca) -\
		gk*y[1]*(y[0] - ek) - gl*(y[0] - el) + iapp)/ep,
		phi*np.cosh((y[0] - v3)/(2.0*v4))*(0.5*(1.0 + np.tanh((y[0] - v3)/v4)) - y[1])])

def sx_dt(y,t, iapp=None, dv=None):
	assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
	ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params(iapp)
	return np.array([(gca*0.5*(1.0 + np.tanh((y[0] - v1)/v2))*(y[0] - eca) +\
		gk*y[1]*(y[0] - ek) + gl*(y[0] - el) - iapp)/ep,
		-1.*phi*np.cosh((y[0] - v3)/(2.0*v4))*(0.5*(1.0 + np.tanh((y[0] - v3)/v4)) + y[1])])

def dy_dt_ode(t,y,iapp=None, dv=None):
	assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
	ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params(iapp)
	return np.array([(-gca*0.5*(1.0 + np.tanh((y[0] - v1)/v2))*(y[0] - eca) -\
		gk*y[1]*(y[0] - ek) - gl*(y[0] - el) + iapp)/ep,
		phi*np.cosh((y[0] - v3)/(2.0*v4))*(0.5*(1.0 + np.tanh((y[0] - v3)/v4)) - y[1])])

def jac(t,y, iapp=None):
	assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
	ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params(iapp)
	return np.array([[(-1.*gl - gk*y[1] - (gca*(-1.*eca + y[0])*(1./np.cosh((y[0] - v1)/v2))**2)/(2.*v2) - .5*gca*(1. + np.tanh((y[0] - v1)/v2)))/ep,-1.*(gk*(-1.*ek + y[0]))/ep],[(phi*np.cosh((y[0] - v3)/(2.*v4))*(1./np.cosh((y[0] - v3)/v4))**2)/(2.*v4) + (phi*np.sinh((y[0] - v3)/(2.*v4))*(-1.*y[1] + .5*(1. + np.tanh((y[0] - v3)/v4))))/(2.*v4),-1.*phi*np.cosh((y[0] - v3)/(2.*v4))]])

def get_eigenvectors(coord, iapp=None):
        assert (type(coord) is list or type(coord) is np.ndarray)
	assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
	eigenvalues, eigenvectors = linalg.eig(jac(0, coord, iapp=iapp))
	return eigenvectors

# Numerical Saddle
# Return saddle point given iapp

def saddle_point(iapp=None):
    assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
    ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params(iapp)
    if iapp < 39:
        def saddle_function(v):
            return (1/(2*gk*(ek - v)))*(-eca*gca - 2*el*gl - 2*iapp + gca*v + 2*gl*v - eca*gca*np.tanh((v-v1)/v2) + gca*v*np.tanh((v-v1)/v2)) - 0.5*(1 + np.tanh((v - v3)/v4))
        #roots = fsolve(saddle_function, np.array([-60,0])) # numerical instabilities
        root_v = brentq(saddle_function, -30, 0)
        root_w = nullcline1(root_v)
        #print root_v, root_w
        return np.array([root_v, root_w])
    else:
        print "No saddle exists for I set to", iapp
        return np.array([-50, 0])



def unstable_focus(iapp=None):
    assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
    ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params(iapp)
    def saddle_function(v):
        return (1/(2*gk*(ek - v)))*(-eca*gca - 2*el*gl - 2*iapp + gca*v + 2*gl*v - eca*gca*np.tanh((v-v1)/v2) + gca*v
*np.tanh((v-v1)/v2)) - 0.5*(1 + np.tanh((v - v3)/v4))
    #roots = fsolve(saddle_function, np.array([-60,0])) # numerical instabilities
    root_v = brentq(saddle_function, 0, 20)
    root_w = nullcline1(root_v)
    #print root_v, root_w
    return np.array([root_v, root_w])


def coordinate_transform(old_coord_data, iapp=None):
	# Translate, then transform
	assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
	saddle = saddle_point(iapp)
	eigenvectors = get_eigenvectors(saddle, iapp)

	eigenvectors[:,0] = eigenvectors[:,0]/linalg.norm(eigenvectors[:,0])
	eigenvectors[:,1] = eigenvectors[:,1]/linalg.norm(eigenvectors[:,1])

	raw_matrix = eigenvectors
	transformed = linalg.inv(raw_matrix)

	if len(old_coord_data)==2 and np.rank(old_coord_data)==1:
		new_coord = np.zeros([2,1])
		new_coord[0] = old_coord_data[0] - saddle[0]
		new_coord[1] = old_coord_data[1] - saddle[1]
		new_coord_final = np.dot(transformed, np.array([new_coord[0], new_coord[1]]))
		return new_coord_final


	elif old_coord_data.shape[0] == 2 and np.rank(old_coord_data)==2:
		old_coord_data[0] = old_coord_data[0] - saddle[0]
		old_coord_data[1] = old_coord_data[1] - saddle[1]
		for i in range(len(old_coord_data[0])):
			old_coord_data[:,i] = np.dot(transformed, old_coord_data[:,i])
		return old_coord_data

	elif old_coord_data.shape[1] == 2 and np.rank(old_coord_data)==2:
		old_coord_data[:,0] = old_coord_data[:,0] - saddle[0]
		old_coord_data[:,1] = old_coord_data[:,1] - saddle[1]
		for i in range(len(old_coord_data)):
			old_coord_data[i] = np.dot(transformed, old_coord_data[i])
		return old_coord_data
	else:
		raise Exception("couldn't find appropriate manipulation for", old_coord_data.shape)

def coordinate_reverse_transform(old_coord_data, iapp=None):
	# Translate, then transform
	assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
	saddle = saddle_point(iapp)
	eigenvectors = get_eigenvectors(saddle, iapp)

	eigenvectors[:,0] = eigenvectors[:,0]/linalg.norm(eigenvectors[:,0])
	eigenvectors[:,1] = eigenvectors[:,1]/linalg.norm(eigenvectors[:,0])

	raw_matrix = eigenvectors

	#transformed = linalg.inv(raw_matrix)
	transformed = raw_matrix

	if len(old_coord_data)==2 and np.rank(old_coord_data)==1:
		old_coord_data = np.dot(transformed, old_coord_data)
		old_coord_data[0] = old_coord_data[0] + saddle[0]
		old_coord_data[1] = old_coord_data[1] + saddle[1]
		return old_coord_data

	elif old_coord_data.shape[1] == 2 and np.rank(old_coord_data)==2:
		for i in range(len(old_coord_data)):
			old_coord_data[i] = np.dot(transformed, old_coord_data[i])
		old_coord_data[:,0] = old_coord_data[:,0] + saddle[0]
		old_coord_data[:,1] = old_coord_data[:,1] + saddle[1]
		return old_coord_data

	elif old_coord_data.shape[0] == 2 and np.rank(old_coord_data)==2:

		for i in range(len(old_coord_data[0])):
			old_coord_data[:,i] = np.dot(transformed, old_coord_data[:,i])
		old_coord_data[0] = old_coord_data[0] + saddle[0]
		old_coord_data[1] = old_coord_data[1] + saddle[1]
		return old_coord_data

	else:
		raise Exception("couldn't find appropriate manipulation for", old_coord_data.shape)


def separatrix(iapp=None, dv=None, max_time=200, max_steps=200000, hmax=1e-2):
    assert (type(iapp) is float or type(iapp) is np.float64 or iapp == None)
    if iapp < 39.:
        saddle = saddle_point(iapp=iapp)
        eigenvectors = get_eigenvectors(saddle)

        y0 = saddle + eigenvectors[:,1] #*(1./1000000.)
        t = np.linspace(0, max_time, max_steps)

        sol1 = integrate.odeint(sx_dt, y0, t, args=(iapp, dv), hmax=hmax)
        sol2 = integrate.odeint(dy_dt, y0, t, args=(iapp, dv), hmax=hmax)
        return sol1, sol2
    else:
        return np.array([0,0])


# Solve for solution
def integrate_ml(dy_dt, y0, iapp=None, dv=None, max_time=200, max_steps=200000, hmax=1e-2):
	assert (type(iapp) is float or np.float64 or iapp == None)
	#evaluate ML system
	t = np.linspace(0, max_time, max_steps)
	sol = integrate.odeint(dy_dt, y0, t, args=(iapp, dv), hmax=hmax)
	return sol

def ml_limit_cycle(dy_dt, y0, iapp=None, dv=None, max_time=400, max_steps=400000, hmax=1e-2):
    assert (type(iapp) is float or np.float64 or iapp == None)
    # run for a while
    t = np.linspace(0, max_time, max_steps)
    vals = integrate.odeint(dy_dt,
        y0,
        t,
        args=(iapp, dv), hmax=hmax)

    max_v_index = np.argmax(vals[:,0])
    max_w = vals[:,1][max_v_index]
    max_v = vals[:,0][max_v_index]

    # calculate the most recent time a new cycle was started
    crossings = (vals[:-1,1]<=max_w) * (vals[1:,1]>max_w) * (vals[1:,0]>0)

    if crossings.sum() < 3:
        raise RuntimeError("No limit cycle detected")

    else:
        # linearly interpolate between the two nearest points
        crossing_fs = ((vals[1:,1][crossings] - max_w)
                / (vals[1:,1][crossings]-vals[:-1,1][crossings]) )

        crossing_ys = (crossing_fs * vals[:-1,0][crossings]
                + (1-crossing_fs) * vals[1:,0][crossings])
        crossing_times = (crossing_fs * t[:-1][crossings]
                + (1-crossing_fs) * t[1:][crossings])

        return ( crossing_times[-1] - crossing_times[-2], np.array([crossing_ys[-1], max_w]),
                abs(crossing_ys[-1]- crossing_ys[-2]), np.array([max_v, max_w]))


def ml_limit_cycle_exact(v, w, iapp=None, dv=None, max_time=400, max_steps=400000, hmax=1e-2):
    y0 = np.array([v, w])

    t = np.linspace(0, max_time, max_steps)
    vals = integrate.odeint(dy_dt,
        y0,
        t,
        args=(iapp, dv), hmax=hmax)

    max_w = w

    # calculate the most recent time a new cycle was started
    crossings = (vals[:-1,1]<=max_w) * (vals[1:,1]>max_w) * (vals[1:,0]>0)

    if crossings.sum() < 3:
        #return -1
        raise RuntimeError("No limit cycle detected")

    else:
        # linearly interpolate between the two nearest points
        crossing_fs = ((vals[1:,1][crossings] - max_w)
                / (vals[1:,1][crossings]-vals[:-1,1][crossings]) )

        crossing_ys = (crossing_fs * vals[:-1,0][crossings]
                + (1-crossing_fs) * vals[1:,0][crossings])
        crossing_times = (crossing_fs * t[:-1][crossings]
                + (1-crossing_fs) * t[1:][crossings])

        print "calculating exact limit cycle for iapp="+str(iapp)+". error is", v - crossing_ys[1]

        return v - crossing_ys[1]



# Thank you Kendrick for this function
def ml_phase_reset(theta_vals, iapp=None, dv=None, dx=0., dy=0.,
        steps_per_cycle=10000, num_cycles=15, return_intermediates=False,
        y0=None, T=None, max_w=None, init=[0,.1], hmax=1e-2):

    assert (type(iapp) is float or np.float64 or iapp == None)
    T, init, err, max_lc = ml_limit_cycle(dy_dt, init, iapp, hmax=hmax)
    v_coord = brentq(ml_limit_cycle_exact, init[0]-10*err, init[0]+10*err, args=(init[1], iapp))
    y0 = np.array([v_coord, init[1]])

    n_phis = []
    err = []
    print "V coordinate is", v_coord, " for iapp="+str(iapp)+", hmax="+str(hmax)+",\
        max_time=default, max_steps=default"

    for i in range(len(theta_vals)):
        theta = theta_vals[i]


        #print err, "limit cycle error"
        steps_before = int(theta * steps_per_cycle) + 1

        # run up to the perturbation
        t1 = np.linspace(0, theta * T, steps_before)
        vals1 = integrate.odeint(dy_dt, y0, t1, args=(iapp, dv), hmax=hmax)

        # run after the perturbation
        t2 = np.linspace(theta * T, T * num_cycles,
            steps_per_cycle * num_cycles - steps_before)

        vals2 = integrate.odeint(dy_dt,
            list(vals1[-1,:] + np.array([dx, dy])),
            t2, args=(iapp, dv), hmax=hmax)


        # calculate crossings
        max_w = max_lc[1]
        crossings = (vals2[:-1,1]<=max_w) * (vals2[1:,1]>max_w) * (vals2[1:,0]>0)

        if crossings.sum() == 0:
            raise RuntimeError("No complete cycles after the perturbation")
        crossing_fs = ((vals2[1:,1][crossings] - max_w)/(vals2[1:,1][crossings]-vals2[:-1,1][crossings]) )
        crossing_times = (crossing_fs * t2[:-1][crossings] + (1-crossing_fs) * t2[1:][crossings])
        crossing_phases = np.fmod(crossing_times, T)/T
        crossing_phases[crossing_phases > 0.5] -= 1
        print theta_vals[i], crossing_phases[-1], np.abs(vals2[-1][0] - max_lc[0])
        err.append(np.abs(vals2[-1][0] - max_lc[0]))
        n_phis.append(-crossing_phases[-1])

    if return_intermediates:
        pass
        #return dict(t1=t1, vals1=vals1, t2=t2, vals2=vals2,
        #    crossings=crossings,
        #    crossing_times=crossing_times,
        #    crossing_phases=crossing_phases)
    else:
        return np.array(n_phis), np.array(err)


def sample_trajectory(y0, iapp=None, dv=None, barrier_coordinate=None, box_corner=None, t0=0.0, dt=0.01, max_time=500, max_steps=500000, return_intermediates=False, hmax=0.0):
    assert (type(hmax) is float or np.float64) and (type(initial_t) is np.ndarray)
    if barrier_coordinate != None and box_corner != None:
        period = 0
        egress_ingress_coord = 0
        egress_ingress_time = 0
        egress_egress_coord = 0
        egress_egress_time = 0

        t = np.linspace(0, max_time, max_steps)
        vals = integrate.odeint(dy_dt,
            y0,
            t,
            hmax=hmax,
            args=(iapp, dv))

        valstemp = vals.copy()
        y0temp = y0.copy()
        barriertemp = barrier_coordinate.copy()
        vals_t = coordinate_transform(valstemp)
        y0_t = coordinate_transform(y0temp)

        barrier_t = coordinate_transform(barriertemp)


        # IndexError: invalid index to scalar variable
        # Caused when attempting to call an element of a scalar
        # i.e. 2[0]
        # last seen with barrier_coordinate and y0

        barrier1 = (vals_t[:-1,1]>=barrier_t[1][0]) * (vals_t[1:,1]<barrier_t[1][0]) * (vals_t[1:,0]<y0_t[0][0])
        if True in barrier1:
            barrier1_position = list(barrier1).index(True)
        else:
            raise Exception("No egress-ingress Crossing")

        if len(barrier1) == 0: # redundant, but keeping just in case
            raise RuntimeError("No barrier crossings")

        crossing_ratio1 = ((vals_t[1:,1][barrier1] - barrier_t[1][0])/(vals_t[1:,1][barrier1]-vals_t[:-1,1][barrier1]) )
        crossing_positions1 = (crossing_ratio1 * vals_t[:-1,0][barrier1] + (1-crossing_ratio1) * vals_t[1:,0][barrier1])
        crossing_times1 = (crossing_ratio1 * t[:-1][barrier1] + (1-crossing_ratio1) * t[1:][barrier1])

        egress_ingress_time = crossing_times1[0]
        egress_ingress_coord_t = np.array([crossing_positions1[0], barrier_t[1][0]])
        egress_ingress_coord = egress_ingress_coord_t
        #temp = egress_ingress_coord_t.copy()
        #egress_ingress_coord = coordinate_reverse_transform(temp)

        #time_before = t[:-1][barrier1][0]
        #time_after = t[1:][barrier1][0]
        #t_tot = time_after - time_before
        #print t_tot, "t_tot"

        #coordinate_before = vals[:-1][barrier1][0]
        #coordinate_after = vals[1:][barrier1][0]
        #coordinate_before_t = vals_t[:-1][barrier1][0]
        #coordinate_after_t = vals_t[1:][barrier1][0]

        #print coordinate_before_t[1], "coordinate_before_t"
        #print barrier_t[1], "barrier_t"
        #print coordinate_after_t[1], "coordinate_after_t"

        #print box_corner, "box_corner"

        #temp = coordinate_before_t.copy()
        #temp2 = coordinate_after_t.copy()
        #cb = coordinate_reverse_transform(temp)
        #ca = coordinate_reverse_transform(temp2)
        #cb = np.array([cb[0], cb[1]])
        #ca = np.array([ca[0], ca[1]])
        #print cb, "cb"
        #print barrier_coordinate, "barrier_coordinate"
        #print ca, "ca"



        #egress_ingress_tof = brentq(border_crossing,
        #    0,
        #    t_tot,
        #    args=(cb, ca, barrier_coordinate, box_corner, 1))


        if egress_ingress_time == 0.0:
            egress_ingress_time = crossing_times1[1]
            egress_ingress_coord_t = np.array([crossing_positions1[1], barrier_t[1][0]])
            temp = egress_ingress_coord_t.copy()
            egress_ingress_coord = coordinate_reverse_transform(temp)

        #t = np.linspace(0.0, egress_ingress_tof, 1000)
        #egress_ingress_coord_x = integrate.odeint(dy_dt, coordinate_before, t)[-1][0]
        #tempx = egress_ingress_coord_x.copy()
        #egress_ingress_coord_x_t = coordinate_transform(tempx)
        #egress_ingress_coord = np.array([egress_ingress_coord_x_t, barrier_t[1][0]])
        #egress_ingress_time = time_before+egress_ingress_tof


        try:
            barrier2 = (vals_t[:-1,0]<=y0_t[0][0]) * (vals_t[1:,0]>y0_t[0][0]) * (vals_t[1:,1]<barrier_t[1][0])

            if True in barrier2:
                barrier2_position = list(barrier2).index(True)
            else:
                raise Exception("No egress-egress Crossing")
            if len(barrier2) == 0:
                raise RuntimeError("No egress-egress crossings")
            crossing_ratio2 = ((vals_t[1:,0][barrier2] - y0_t[0][0])/(vals_t[1:,0][barrier2]-vals_t[:-1,0][barrier2]) )
            crossing_positions2 = (crossing_ratio2 * vals_t[:-1,1][barrier2] + (1-crossing_ratio2) * vals_t[1:,1][barrier2])
            crossing_times2 = (crossing_ratio2 * t[:-1][barrier2] + (1-crossing_ratio2) * t[1:][barrier2])
            egress_egress_coord = np.array([y0_t[0][0], crossing_positions2[0]])


            if egress_egress_time == 0.0:
                egress_egress_time = crossing_times2[1]
                egress_egress_coord = np.array([y0_t[0][0], crossing_positions2[1]])
            #    egress_egress_coord = np.array([y0_t[0][0], crossing_positions2[1]])
            #evaluate trajectory up to egress_egress
            t = np.linspace(0, egress_egress_time, max_steps)
            #t = np.linspace(0, egress_ingress_time, max_steps)
            trajectory = integrate.odeint(dy_dt,
                y0,
                t, hmax=hmax)

        except IndexError:
            egress_egress_coord = np.array([0, 0])
            egress_egress_time = 0
            t = np.linspace(0, egress_ingress_time, max_steps)
            trajectory = integrate.odeint(dy_dt,
              y0,
              t, hmax=hmax)

        if return_intermediates:
            return dict(t=t, vals=vals,
                    barrier1=barrier1,
                    barrier2=barrier2,
                    crossing_ratio1=crossing_ratio1,
                    crossing_ratio2=crossing_ratio2,
                    crossing_times1=crossing_times1,
                    crossing_times2=crossing_times2)
        else:
            return [trajectory, y0, egress_ingress_coord, egress_ingress_time, egress_egress_coord, egress_egress_time]
    else:
        t = np.linspace(0, max_time, max_steps)
        vals = integrate_ml(dy_dt, y0, max_time=max_time, max_steps=max_steps)
        return vals


def diffeq(y, t, iapp, dummyvar):
    """
    ml function with iapp parameter, for use in sims that
    require autonomous changing of iapp

    y: input tuple or list
    t: time array
    iapp: bifurcation parameter
    dummyvar: A dummy variable for brentq to work properly
    """
    assert (type(hmax) is float or np.float64) and (type(initial_t) is np.ndarray)
    ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params(iapp=iapp)
    return np.array([(-gca*0.5*(1.0 + np.tanh((y[0] - v1)/v2))*(y[0] - eca) -\
        gk*y[1]*(y[0] - ek) - gl*(y[0] - el) + iapp)/ep,
    phi*np.cosh((y[0] - v3)/(2.0*v4))* \
    (0.5*(1.0 + np.tanh((y[0] - v3)/v4)) - y[1])])


def homoclinic_bifurcation(iapp):

    """
    A function called by brentq or bisect to find the homoclinic bifurcation.
    Returns 1 if a limit cycle exists
    Returns -1 if no limit cycle exists

    iapp: float, any value for iapp
    """

    # iapp must be a float
    assert (type(iapp) is float or np.float64 or iapp == None)

    y0=[0,.09] # initial condition
    T, limit_cycle_init, err, max_w = ml_limit_cycle(diffeq, y0, iapp=iapp, max_time=800, max_steps=400000, hmax=1e-2)
    if limit_cycle_init == None:
        print "-1"
        return -1.
    else:
        print "+1"
        return +1.




def border_crossing(x, cb, ca, barrier, corner, axis):
    """
    t: time at which to evaluate the differential equation.
    y0: initial condition -- use the very first egress point.
    barrier: barrier coordinates.
    axis: axis of interest (0 for x and 1 for y).
    """

    """
    print ca, cb, "ca, cb"
    v = cb - ca
    u = np.array([-v[1], v[0]])
    N = u/linalg.norm(u)
    d = np.dot(N, barrier)

    print d, "d"
    print np.dot(N, corner)
    print np.dot(N, cb), "np.dot(N, cb)"
    print np.dot(N, ca), "np.dot(N, ca)"
    """
    if x == 0.0:
        """
        print d - np.dot(N, cb), "x==0.0"
        return d - np.dot(N, cb)
        """

    else:
        """
        t = np.linspace(0.0, x, 5)
        sol = integrate.odeint(dy_dt, cb, t)
        print d - np.dot(N, sol[-1]), "other dot product"
        return d - np.dot(N, sol[-1])
        """

def numerics_check(hmax, initial_t=np.array([1e-3, 5e-4]),
    box_corner_t=np.array([1e-3,1e-3]), barrier_coordinate_t=np.array([0.0, 1e-3])):

    """
    hmax: max step size for use in scipy integrator
    initial_t: transformed initial condition
    box_corner_t: transformed box corner
    barrier_coordinate_t: transformed top border of box
    """

    # I'm keeping all default values and input values as numpy arrays
    # Make sure to import numpy as np
    assert (type(hmax) is float or np.float64) and (type(initial_t) is np.ndarray)
    assert (type(box_corner_t) is np.ndarray) and (type(barrier_coordinate_t) is np.ndarray)

    # Transform all coordinates into the original space
    initial = coordinate_reverse_transform(initial_t)
    box_corner = coordinate_reverse_transform(box_corner_t)
    barrier_coordinate = coordinate_reverse_transform(barrier_coordinate_t)

    # For the sake of convenience, make sure rank == 1
    # and shape == (2,).  This depends
    # greatly upon coordinate_reverse_transform
    assert (np.rank(initial) == 1 and initial.shape == (2,))
    assert (np.rank(box_corner) == 1 and box_corner.shape == (2,))
    assert (np.rank(barrier_coordinate) == 1 and barrier_coordinate.shape == (2,))

    # Get the data from the trajectory function
    trajectory, initial, egress_ingress_coord, egress_ingress_time, egress_egress_coord, egress_egress_time = sample_trajectory(
        initial,
        barrier_coordinate=barrier_coordinate,
        box_corner=box_corner,
        hmax=hmax)

    # Return only what we need
    return egress_ingress_coord
