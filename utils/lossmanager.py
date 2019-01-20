import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class LossManager:

    def __init__(self):
        self.losses :list = []

    def append(self, loss):
        self.losses.append(loss)
        return

    def draw_graph(self, outpath):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel = "Loss"
        plt.plot(self.losses)
        plt.savefig(outpath)
