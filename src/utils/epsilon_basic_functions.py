import math

"""
these functions give the exploration ratio at each given iteration step they take as argument the state & the
iteration num, return a number between 0 and 1
"""
class EpsilonBasicFuncs:

    def constant_x(self, x):
        return lambda state, iteration_num: x

    def linear(self, start, end, in_x_iteration):
        def exploration_ratio_func(state, iteration_num):
            if iteration_num > in_x_iteration:
                return end
            else:
                return start + ((end - start) * iteration_num / in_x_iteration)

        return exploration_ratio_func

    def logarithmic(self, start, end, in_x_iteration):
        def exploration_ratio_func(state, iteration_num):
            if iteration_num > in_x_iteration:
                return end
            else:
                return start + ((end - start) * math.log(iteration_num + 1) / math.log(in_x_iteration))

        return exploration_ratio_func

