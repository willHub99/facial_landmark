import pickle
import matplotlib.pyplot as plt
import os

class Plot: 

    def Plot(self, date_format):

        #armazena o diretório base do projeto
        dir_base = os.getcwd()

        with open(f"{dir_base}/assets/files/ear/ear_{date_format}.pkl", 'rb') as handle:
            data_ear = pickle.load(handle)
        
        with open(f"{dir_base}/assets/files/square/square_{date_format}.pkl", 'rb') as handle:
            data_square = pickle.load(handle)

        plt.plot(data_ear['ear'])
        #plt.plot(data_square['square'], color='red')
        #plt.xlim([0,100])
        #plt.ylim([0,0.6])
        plt.autoscale = True
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel("Time")
        plt.ylabel("EAR")
        plt.title("Eye Avarage Ratio")
        plt.show()

