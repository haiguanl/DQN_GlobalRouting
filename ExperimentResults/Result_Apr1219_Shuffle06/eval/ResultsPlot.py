import pickle as pkl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":

    filename = 'solutionComboDRL'
    fileObject = open(filename,'rb')
    route = pkl.load(fileObject)
    fileObject.close()

    print('Route',route)