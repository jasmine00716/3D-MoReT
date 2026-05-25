import abc


class Writer:
    def __init__(self, output_file=None):
        self.output_file = output_file

    @abc.abstractmethod
    def write(self, config):
        return