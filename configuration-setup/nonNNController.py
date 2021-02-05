
class nonNNPlant(object):
    def __init__(self, model, steps=1000):
        self.model = model
        self.steps = steps
        self.mode = 0

    def get_controller_input(self, state):
        if self.model == 'CartPoleLinControl':
            M = 1
            g = 9.81

            sigma = state[0]
            theta = state[3]

            control = -1.1 * M * g * theta - sigma
            return control
        elif self.model == 'CartPoleLinControl2':
            control = -10.0 * state[0] + 289.83 * state[1] - 19.53 * state[2] + 63.25 * state[3]
            return control
        else:
            return 1

    def get_vel(self, state):
        vel = state[1]
        print("start mode " + str(self.mode))
        if self.mode == 0 and state[0] <= 0.01 and state[1] <= 0:
            print("setting mode 1")
            self.mode = 1
            vel = -0.75 * state[1]
        print(" mode is " + str(self.mode))

        return vel

    def get_mode(self, state):
        mode = 0
        if state[1] >= 0:
            mode = 1
        return mode
