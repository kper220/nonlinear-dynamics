# RK4 (Runge-Kutta)
# djli

class RungeKutta:

    def __init__(self, f):
        self._f = f

    def next_step(self, h, t, y, **kwargs):
        k1 = h*self._f(t, y, **kwargs)
        k2 = h*self._f(t + h/2, y + k1/2, **kwargs)
        k3 = h*self._f(t + h/2, y + k2/2, **kwargs)
        k4 = h*self._f(t + h, y + k3, **kwargs)

        return y + (k1 + 2*k2 + 2*k3 + k4)/6

    def solve(self, initial_value, tt, **kwargs):
        n = len(tt)
        t_min, t_max = tt[0], tt[n-1]
        h = (t_max - t_min) / (n - 1)

        solution = [None for _ in range(n)]
        solution[0] = initial_value

        for i in range(1, n):
            solution[i] = self.next_step(h, tt[i-1], solution[i-1], **kwargs)

        return solution

    def initial_value_solver(self, tt):

        def _solve(initial_value, **kwargs):
            return self.solve(initial_value, tt, **kwargs)

        return _solve

if __name__ == '__main__':
    import numpy as np

    tt = np.linspace(0, 10, 1000)

    # first order differential equation
    f = lambda t,x,w=1:np.sin(w*x)
    solver = RungeKutta(f).initial_value_solver(tt)
    initial_value = 2
    solution = solver(initial_value, w=np.pi)
    print('first order differential equation: ', solution[-1])

    # second order differential equation
    def g(t,x):
        return np.array(x[1], -x[0])
    solver = RungeKutta(f).initial_value_solver(tt)
    initial_value = np.array([1, 0])
    solution = solver(initial_value)
    print('second order differential equation: ', solution[-1])

    exit(0)
