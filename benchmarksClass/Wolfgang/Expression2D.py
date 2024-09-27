import dolfin as dl

# user expresstion for 1 in the sepcified grid and 0 in the other domain
class Expression2D(dl.UserExpression):
    def __init__(self, degree, x_lim, y_lim):
        self.x_lim = x_lim
        self.y_lim = y_lim
        super().__init__(degree=degree)
    def eval(self, value, x):
        if x[0] >self.x_lim[0] and x[0] < self.x_lim[1] and x[1] > self.y_lim[0] and x[1] < self.y_lim[1]:
            value[0] = 1
        else:
            value[0] = 0