# driven damped pendulum and solvers.
# djli

import numpy as np
from rk4 import RungeKutta

def equations_of_motion(t, x, w0, w, b, g):
    x1, x2 = x
    return np.array([x2, -w0**2*np.sin(x1) - 2*b*x2 + g*w0**2*np.cos(w*t)])

# general recursive algorithm.
def general_recursion(function, tt, initial_value, w0, w, b, g):
    '''
        Implements a function recursively, according to its arguments.
        Saves time in later function definitions (or not).
        '''

    if isinstance(g, np.ndarray):
        return np.array([general_recursion(function, tt, initial_value, w0, w, b, _g) for _g in g])
    if isinstance(b, np.ndarray):
        return np.array([general_recursion(function, tt, initial_value, w0, w, _b, g) for _b in b])
    if isinstance(w, np.ndarray):
        return np.array([general_recursion(function, tt, initial_value, w0, _w, b, g) for _w in w])
    if isinstance(w0, np.ndarray):
        return np.array([general_recursion(function, tt, initial_value, _w0, w, b, g) for _w0 in w0])
    if isinstance(initial_value[0], np.ndarray):
        return np.array([general_recursion(function, tt, _initial_value, w0, w, b, g) for _initial_value in initial_value])

    return function(tt, initial_value, w0, w, b, g)

# general solver.
def solve(tt, initial_value, w0, w, b, g):
    '''
        Solves the DDP differential equation.
        
        Arguments: array arguments must be of type numpy.array.
        tt:             linear timespace
        initial_value:  an initial value state(s) of two values
        w0:             natural frequency(s)
        w:              driving frequency(s)
        b:              damping constant(s)
        g:              driving strength(s)

        Returns:
        Solution or array of solutions.
        '''

    def _solve(tt, initial_value, w0, w, b, g):

        def _equations_of_motion(t, x):
            return equations_of_motion(t, x, w0, w, b, g)

        solver = RungeKutta(_equations_of_motion).initial_value_solver(tt)
        return [s[0] for s in solver(initial_value)]

    return general_recursion(_solve, tt, initial_value, w0, w, b, g)

def difference(solution, log=True):

    m = len(solution)
    
    # multiple solutions, neighboring difference.
    if m > 2:
        return [difference(np.array([solution[i],solution[i-1]]), log=log) for i in range(1,m)]

    # difference between two solutions
    s1, s2 = solution
    n = len(s1)

    if log:
        d = [np.log(abs(s2[i] - s1[i])) for i in range(n)]
    else:
        d = [abs(s2[i] - s1[i]) for i in range(n)]

    return d

def extract(tt, period=1, displacement=0):
    '''
    Extract indices of timesteps close to a multiple of the assigned period.
    Allows us to get indices of corresponding solutions at timesteps.
    '''

    indices = list()
    current_value = displacement

    for i in range(len(tt)):
        if tt[i] >= current_value:
            current_value += period
            indices.append(i)

    return indices

# single period detector of period 1.
def repeated_points(tt, g, solution, period=1, displacement=0):

    ii = extract(tt, period=period, displacement=displacement)
    g_corresponding = [g for i in ii]
    solution_extract = [solution[i] for i in ii]

    return g_corresponding, solution_extract

def d(xx):
    n = len(xx)
    yy = [None for _ in range(n-1)]

    for i in range(1, n):
        yy[i-1] = xx[i] - xx[i-1]

    return yy

def state_space_trajectory(xx):
    _xx = xx[1:]
    yy = d(xx)
    return _xx, yy

def poincare_diagram(tt, solution, period=1, displacement=0):

    indices = extract(tt, period=period, displacement=displacement)
    n = len(indices)

    xx, yy = state_space_trajectory(solution)

    try:
        _xx = [xx[i] for i in indices[:n-1]]
    except IndexError:
        pass
    try:
        _yy = [yy[i] for i in indices[:n-1]]
    except IndexError:
        pass

    return _xx, _yy

# code broke.
def bifurcation(tt, initial_value, w0, w, b, gg, displacement=0):
    solution = solve(tt, initial_value, w0, w, b, gg)
    period = 2*np.pi/w

    ii = extract(tt, period=period, displacement=displacement)
    
    n_t = len(tt)
    n_g = len(gg)
    n_i = len(ii)

    g_extract = [None for _ in range(n_g*n_i)]
    s_extract = [None for _ in range(n_g*n_i)]
    k = 0

    for j in range(n_g):
        g = gg[j]

        for i in ii:
            g_extract[k] = g
            s_extract[k] = solution[j][i]
            k += 1

    return g_extract, s_extract

if __name__ == '__main__':

    tt = np.linspace(0, 1, 10)
    w = 2*np.pi
    w0 = 1.5*w
    b = w0/4

    vv = np.linspace(0, .1, 10)
    initial_values = [np.array([v, 0]) for v in vv]
    gg = np.linspace(0, .2, 10)

    bifurcation(tt, initial_values, w0, w, b, gg)

    exit(0)
