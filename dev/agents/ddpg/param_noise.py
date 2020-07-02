class ParamNoise(object):
    def __init__(self, initial_stddev = 0.1, desired_action_stddev = 0.1, adaption_coeff = 1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaption_coeff = adaption_coeff
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance <= self.desired_action_stddev:
            self.current_stddev = self.current_stddev * self.adaption_coeff
        else:
            self.current_stddev = self.current_stddev / self.adaption_coeff

    def get_stats(self):
        return self.current_stddev
