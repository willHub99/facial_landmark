import pickle
import matplotlib.pyplot as plt

class PlotEAR: 

    def PlotEAR(self):

        with open('files/ear.pkl', 'rb') as handle:
            data_ear = pickle.load(handle)

        with open('files/square.pkl', 'rb') as handle:
            data_square = pickle.load(handle)

        plt.plot(data_ear['ear'])
        plt.plot(data_square['square'], color='red')
        plt.xlim([0,100])
        plt.ylim([0,0.6])
        plt.autoscale = True
        plt.xlabel("Time")
        plt.ylabel("EAR")
        plt.title("Eye Avarage Ratio")
        plt.show()

