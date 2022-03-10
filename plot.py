import pickle
import matplotlib.pyplot as plt

class PlotEAR: 

    def __init__(self, saveEAR):
        self.saveEAR = saveEAR

    def PlotEAR(self):
        unpickle = pickle.loads(self.saveEAR)


        plt.plot(unpickle)
        plt.xlim([0,100])
        plt.ylim([0,0.5])
        plt.autoscale = True
        plt.xlabel("Time")
        plt.ylabel("EAR")
        plt.title("Eye Avarage Ratio")
        plt.show()

