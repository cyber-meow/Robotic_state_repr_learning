
import matplotlib.pyplot as plt

# realstates and learnedstates must be two np arrays
def plot_states(realstates, learnedstates, xy, path=None):
    if xy == 'x':
        cs = realstates[:,0]
    elif xy == 'y': 
        cs = realstates[:,1]
    else:
        raise ValueError("xy must be 'x' or 'y'")
    fig, ax = plt.subplots(figsize=(16,12))
    plt.scatter(learnedstates[:,0], learnedstates[:,1], 
                s=3, lw=0, c=cs, vmin=0, vmax=45)
    plt.xlabel("State dimension 1")
    plt.ylabel("State dimension 2")
    cbar = plt.colorbar()
    cbar.set_label("{}-Coordinate of the robot".format(xy))
    if path is not None:
        plt.savefig(path)
    plt.show()