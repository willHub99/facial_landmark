import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

class Plot: 

    def Plot(self, date_format):

        #armazena o diretório base do projeto
        dir_base = os.getcwd()

        with open(f"{dir_base}/assets/files/ear/ear_{date_format}.pkl", 'rb') as handle:
            data_ear = pickle.load(handle)
        
        fig, ax = plt.subplots()
        sns.distplot(data_ear['ear'], bins=25, color="g", ax=ax)
        plt.show()

        ear = np.array(data_ear['ear'])
        ear = ear.reshape(-1,1)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(ear)
        plt.plot(kmeans.labels_)
        plt.show()

        plt.plot(data_ear['ear'])
        plt.autoscale = True
        plt.xlabel("Time")
        plt.ylabel("EAR")
        plt.title("Eye Avarage Ratio")
        plt.show()

        plt.style.use('_mpl-gallery')
        # plot
        fig, ax = plt.subplots()

        ax.stem(sorted(data_ear['ear']))

        plt.xlabel("Time")
        plt.ylabel("EAR")
        plt.title("Eye Avarage Ratio")

        plt.show()


