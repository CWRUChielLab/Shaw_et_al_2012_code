#!/usr/bin/env python2

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

import numpy as np
from scipy import integrate
from scipy import optimize
import math

def f_saddle(unused_t, y, l_s, l_u):
    r"""Compute the gradient in the region around a saddle.

    Computes the gradient around a saddle point at (0,0):

    .. math::

        \frac{d\mathbf{y}}{dt} =
            \left( \begin{array}{cc}
            -\lambda_s & 0\\
            0 & \lambda_u \\
            \end{array} \right) \mathbf{y}

    :param unused_t: the simulation time when the gradient is sampled
    :type unused_t: float
    :param y: the (two dimensional) point where the gradient is sampled
    :param l_s: the magnitude of the stable (first) Eigenvalue
        (should be positive)
    :param l_u: the magnitude of the unstable (second) Eigenvalue

    :rtype: A two dimensional vector
    """
    return y * np.array([-l_s, l_u])


def f_sin(unused_t, y, mu, alpha, k):
    r"""Compute the gradient of the sine system.

    Computes the gradient of sine system:

    .. math::

        \frac{d\mathbf{y}}{dt} =
            \left( \begin{array}{cc}
            1 & -\mu \\
            \mu & 1 \\
            \end{array} \right)
            \left( \begin{array}{c}
            \cos(y_0) \sin(y_1) + \alpha \sin(2 k \; y_0) \\
            -\sin(y_0) \cos(y_1) + \alpha \sin(2 k \; y_1) \\
            \end{array} \right)

    :param unused_t: the simulation time when the gradient is sampled
    :type unused_t: float
    :param y: the (two dimensional) point where the gradient is sampled
    :param mu: a parameter controlling scaling and rotation of the system
    :param alpha: a parameter controlling the local strength of the central
        focus's attraction/repulsion
    :param k: a parameter controlling the number of limit cycles
    :type k: int

    :rtype: A two dimensional vector
    """
    f = math.cos(y[0])*math.sin(y[1]) + alpha*math.sin(2*k*y[0])
    g = -math.sin(y[0])*math.cos(y[1]) + alpha*math.sin(2*k*y[1])
    return np.array([f - mu*g, g + mu*f])


def find_iris_region(y, l_s, l_u, a):
    r""" Find the saddle of the iris system that contains this point.

    Finds the saddle containing the current point.  Returns the saddle location
    and its jacobian.

    :param y: the (two dimensional) point where the gradient is sampled
    :param l_s: the magnitude of the stable (first) Eigenvalue
        (should be positive)
    :param l_u: the magnitude of the unstable (second) Eigenvalue
    :param a: the offset between adjacent saddles/

    :rtype: A two dimensional vector for the saddle's position and a two by
        two matrix for its Jacobian.
    """
    # project the point into a reference frame where the saddles can be
    # connected by lines to form 1x1 squares.
    u = np.dot(np.array([[-a/2., 1.], [1., a/2.]]), y)/(4. + a*a)

    # compute the offset in squares from the central square
    o = np.floor(u + np.r_[1./2, 1./2])

    # every other square has the stabilities reversed
    flip = 1. - 2.*(int(o.sum()) % 2 != 0)

    # translate the position so it's in the central square
    t = np.dot(2 * np.array([[-a/2., 1.], [1., a/2.]]), o)
    v = y - t

    # figure out which saddle we're near
    if v[0] > -a/2. and v[1] > a/2.:
        s = np.r_[1 - a/2., 1 + a/2.] + t
        j = flip * np.array([[-l_s, 0.],[0., l_u]])
    elif v[0] > a/2. and v[1] < a/2.:
        s = np.r_[1 + a/2., -1 + a/2.] + t
        j = flip * np.array([[l_u, 0],[0, -l_s]])
    elif v[0] < a/2. and v[1] < -a/2.:
        s = np.r_[-1 + a/2., -1 - a/2.] + t
        j = flip * np.array([[-l_s, 0.],[0., l_u]])
    elif v[0] < -a/2. and v[1] > -a/2.:
        s = np.r_[-1 - a/2., 1 - a/2.] + t
        j = flip * np.array([[l_u, 0],[0, -l_s]])
    else:
        # not defined, but we'll throw in a central spiral
        s = np.r_[0.,0.] + t
        j = flip * np.array([[-l_s, l_u],[-l_u, -l_s]])

    return s, j


def f_iris(unused_t, y, l_s, l_u, a):
    r"""Compute the gradient in the iris system

    Computes the gradient of the iris system, with saddles at
    :math:`(1 - a/2, 1 + a/2)`,
    :math:`(1 + a/2, -1 + a/2)`,
    :math:`(-1 + a/2, -1 - a/2)`, and
    :math:`(-1 - a/2, 1 - a/2)`.
    The flow is linearized in a 2x2 region around each saddle.

    :param unused_t: the simulation time when the gradient is sampled
    :type unused_t: float
    :param y: the (two dimensional) point where the gradient is sampled
    :param l_s: the magnitude of the stable (first) Eigenvalue
        (should be positive)
    :param l_u: the magnitude of the unstable (second) Eigenvalue
    :param a: the offset between adjacent saddles/

    :rtype: A two dimensional vector
    """
    s, j = find_iris_region(y, l_s, l_u, a)
    return np.dot(j, y-s)


def integrate_until_event(f, event_test, y0, t0=0., fparams=[],
        return_path=False,
        max_dt=0.1, t_max=1e2,
        subdivisions=10, subdivision_depth=10,
        atol=1e-6, rtol=1e-6):
    r"""Integrate until just after an event occurs

    Integrate a trajectory that ends just after an given function returns true.
    For example, one could integrate until the trajectory just leaves a region
    of interest or just crosses a Poincare section.

    :param f: a function *f(t, y, ...)* which computes the gradient
    :param event_test: a function *event_test(t, y, t_last, y_last, ...)* which
        returns True iff the event occured between t and t_last.
    :param y0: the initial conditions.
    :param t0: the starting time.
    :param fparams: extra parameters for the gradient and event functions
    :param return_path: should the intermediate integration steps along the
        path be returned, or just the final position?
    :type return_path: bool
    :param max_dt: the largest step the numerical integrator can safely take.
    :param t_max: the maximum time to run the integrator before giving up
    :param subdivisions: the number of smaller steps to divide a step into that
        has triggered an event, so that it can be retried with the smaller
        steps to find the crossing time more accurately.
    :param subdivision_depth: the number of times to recursively subdivide the
        crossing step into smaller steps.

    :rtype: If return_path is true, a tuple of times and positions forming the
        path that was integrated.  Otherwise, return a tuple containing the
        final time and final position, or
        *nan* for each if the maximum time was exceeded.
    """

    times = []
    positions = []

    t_last = t0
    y_last = y0.copy()
    ode = integrate.ode(f)
    ode.set_integrator("vode", atol=atol, rtol=rtol)
    ode.set_initial_value(y0, t0).set_f_params(*fparams)
    while ode.successful() and ode.t < t_max and not (
            event_test(ode.t, ode.y, t_last, y_last, *fparams)):
        t_last = ode.t
        y_last[:] = ode.y[:]
        if return_path:
            times.append(t_last)
            positions.append(y_last.copy())
        ode.integrate(ode.t + max_dt)

    if ode.t < t_max and subdivision_depth > 0:
        t_last, y_last = integrate_until_event(
                y0= y_last, t0= t_last,
                max_dt= (ode.t-t_last)/subdivisions,
                subdivision_depth= subdivision_depth-1,
                f= f, event_test= event_test, t_max= t_max,
                subdivisions= subdivisions, fparams= fparams);
        if return_path:
            times.append(t_last)
            positions.append(y_last)

    if return_path:
        return (np.array(times), np.array(positions))
    elif ode.t >= t_max:
        return (np.nan, np.nan*ode.y)
    elif subdivision_depth == 0:
        # return a time guaranteed to be after the event, in case the
        # caller wants to continue integrating until the next occurrence.
        return (ode.t, ode.y)
    else:
        return (t_last, y_last)

def vector_integrate_until_event(f, event_test, y0s, *args, **kwargs):
    """ Integrates multiple trajectories until an event occurs

    This is a vectorized version of :meth:`integrate_until_event` which
    accepts multiple initial conditions *y0s*.

    :rtype: A tuple containing an array of final times and an array of final
        positions.
    """
    result = np.array([np.r_[
        integrate_until_event(
            f, event_test, y0, *args, **kwargs)
        ] for y0 in y0s])
    return (result[:,0], result[:,1:])


def plane_crossing_test(norm, dist):
    """ Create a function that returns true when a trajectory crosses a plane

    This generates a function of the form
    *event_test(t, y, t_last, y_last, ...)* which returns True iff y_last
    is on the back side of the given plane and y is on the front of the plane.

    :param norm: The normal vector to the front of the plane
    :param dist: The distance along the normal vector from the origin to the
        plane - positive if the origin is behind the plane, negative otherwise.

    :rtype: True iff the plane has been crossed.
    """
    def event_test(unused_t, y, unused_t_last, y_last, *unused_params):
        return (np.dot(y, norm) >= dist) and (np.dot(y_last, norm) < dist)

    return event_test


def return_false(*params, **kwparams):
    """ Ignore any parameters and return False.

    This is a function that always returns False.  This useful for things like
    checking for an event that never happens.

    :rtype: False
    """
    return False


def find_limit_cycle(f, norm, dist, guess, uncertainty=0.3, fparams=[],
        max_dt=0.1, t_max=1e2, subdivisions=10, subdivision_depth=10,
        atol=1e-6, rtol=1e-6):
    """ Finds where a limit cycle crosses a plane

    Finds a point on a local section where a limit cycle exists.

    :param f: a function *f(t, y, ...)* which computes the gradient
    :param norm: The normal vector to the front of the plane
    :param dist: The distance along the normal vector from the origin to the
        plane - positive if the origin is behind the plane, negative otherwise.
    :param guess: An initial guess on or near the plane.
    :param max_dt: the largest step the numerical integrator can safely take.
    :param t_max: the maximum time to run the integrator before giving up
    :param subdivisions: integration parameter - see
        :meth:`integrate_until_event`.
    :param subdivision_depth: integration parameter - see
        :meth:`integrate_until_event`.
    :param fparams: extra parameters for the gradient function

    :rtype: A point on or just after the plane in question.

    """
    norm = norm / math.sqrt(np.dot(norm, norm)) # make sure it's a unit norm
    if len(norm) == 2:
        # 2d version
        antinorm = np.array([norm[1], -norm[0]])

        def delta_f(u):
            # map the point onto a point near the plane
            y0 = u*antinorm + dist*norm

            # nudge the point if needed to make sure we start on or just past
            # the plane
            diff = np.dot(y0, norm) - dist
            if diff < 0:
                y0 += 2*diff*norm

            # integrate one full cycle
            t, y_next = integrate_until_event(
                    f, plane_crossing_test(norm, dist), y0,
                    max_dt= max_dt, t_max= t_max, subdivisions=subdivisions,
                    subdivision_depth = subdivision_depth, fparams= fparams,
                    atol= atol, rtol= rtol)

            return np.dot(antinorm, y_next - y0)

        u = optimize.brentq(delta_f,
                np.dot(guess, antinorm) - uncertainty,
                np.dot(guess, antinorm) + uncertainty,
                )
        return dist*norm + u*antinorm

    else:
        # This works, but we may need to project onto lower dimensional
        # subspace for fmin to converge quickly

        def delta_g(y):
            # map the point onto a point near the plane
            y0 = y + (dist - np.dot(norm, y))*norm

            # nudge the point if needed to make sure we start on or just past
            # the plane
            diff = np.dot(y0, norm) - dist
            if diff < 0:
                y0 += 2*diff*norm

            # integrate one full cycle
            t, y_next = integrate_until_event(
                    f, plane_crossing_test(norm, dist), y0,
                    max_dt= max_dt, t_max= t_max, subdivisions=subdivisions,
                    subdivision_depth = subdivision_depth, fparams= fparams,
                    atol= atol, rtol= rtol)

            return math.sqrt( ((y_next - y0)**2).sum() )

        return optimize.fmin(delta_g, guess)


def find_prc(f, norm, dist, y_lc, delta_y, num_points=10,
        num_cycles=10, fparams=[], max_dt=0.1, t_max=1e2,
        subdivisions=10, subdivision_depth=10,
        atol=1e-6, rtol=1e-6):

    planetest = plane_crossing_test(norm, dist)

    # find the full cycle time
    # First run forward the remaining number of cycles
    t_last, y_last = 0., y_lc.copy()
    for unused_j in range(num_cycles):
        t_last, y_last = integrate_until_event(
                f, planetest, y_last, t0=t_last,
                max_dt= max_dt, t_max= t_max,
                subdivision_depth = subdivision_depth, fparams= fparams,
                atol= atol, rtol= rtol)
    t_full = t_last / num_cycles
    #t_full, y = integrate_until_event(
    #        f, planetest, y_lc,
    #        max_dt= max_dt, t_max= t_max, fparams= fparams)

    t_perterb = (np.linspace(0., t_full, num_points, endpoint=False)
            + t_full/num_points/2)
    delta_T = np.empty_like(t_perterb)

    for i in range(len(t_perterb)):
        # run until the perterbation
        t_last, y_last = integrate_until_event(
                f, (lambda t,*unused: t > t_perterb[i]), y_lc,
            max_dt= max_dt, t_max= t_max,
            subdivision_depth = subdivision_depth, fparams= fparams,
            atol= atol, rtol= rtol)

        # apply the perterbation
        y_last += delta_y;

        # if the perterbation took us across the plane, we may need to
        # increase or decrease the number of cycles to ignore this
        # added crossing.
        #
        # TODO: harder than it looks, because the cycle will typically cross
        # the plane forwards and backwards and we're only counting the forward
        # crossings.  For now, just use small perturbations.
        #
        #if np.dot(delta_y, norm) > 0 and planetest(
        #        0, y_last, 0, y_last-delta_y):
        #    next_num_cycles = num_cycles - 1
        #elif np.dot(delta_y, norm) < 0 and planetest(
        #        0, y_last-delta_y, 0, y_last):
        #    next_num_cycles = num_cycles + 1
        #else:
        #    next_num_cycles = num_cycles
        next_num_cycles = num_cycles

        # run forward the remaining number of cycles
        for unused_j in range(next_num_cycles):
            t_last, y_last = integrate_until_event(
                    f, planetest, y_last, t0=t_last,
                    max_dt= max_dt, t_max= t_max,
                    subdivision_depth = subdivision_depth, fparams= fparams,
                    atol= atol, rtol= rtol)

        #delta_T[i] = t_last/num_cycles - t_full
        delta_T[i] = t_last / t_full - num_cycles

    return np.array([t_perterb / t_full, delta_T])




