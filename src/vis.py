import numpy
import matplotlib.pyplot as plt

class Visualization():

    def __init__(self, target, output, loss_data):

        self.target = target
        self.output = output
        self.loss_data


    def loss_plot(self, show=False):

        ax, fig = plt.subplots()
        ax.plot(self.loss_data)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")

        if show:
            plt.show()

    def PlotTrainRes(self):

        fig, ax = p




