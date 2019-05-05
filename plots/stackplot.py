import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os

# function to smooth out a plot
def ma(x, N=4):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# setup plots
fig, axes = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False, figsize=(15,8))

# iterate over the three datasets
for idset, dset in enumerate(["mnist", "cifar", "imagenet"]):
    N = [18, 4, 1][idset]
    dataset = dset
    latents = [1, 2, 4] if dset == "imagenet" else [1, 2, 4, 8]
    xmax = [6500, 1200, 110][idset]
    ylim = [[0.87, 1.4], [1.4, 4.7], [2.25, 5.0]][idset]

    for idx, nz in enumerate(latents):
        indices = []
        if os.path.exists('{}{}/x.csv'.format(dataset, nz)):
            x = genfromtxt('{}{}/x.csv'.format(dataset, nz), delimiter=',')[1:,1:]
        else:
            i = 1
            xlist = []
            while os.path.exists('{}{}/x{}.csv'.format(dataset, nz, i)):
                x = genfromtxt('{}{}/x{}.csv'.format(dataset, nz, i), delimiter=',')[1:, 1:]
                if i > 1:
                    begins_at = x[0,0]
                    index = np.where(xlist[-1][:,0] == begins_at)[0][0]
                    indices.append(index)
                    xlist[-1] = xlist[-1][:index,:]
                xlist.append(x)
                i += 1
            x = np.concatenate(xlist, axis=0)

        max_epoch = x[-1,0]
        epochs = x[:,0]
        x = x[:,1]
        x = ma(x, N=N)
        stack = [x]

        rgb = [255, 255, 255]
        colors = ['#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])]
        labels = ["x"]
        for zi in range(1, nz + 1):
            if os.path.exists('{}{}/z{}.csv'.format(dataset, nz, zi)):
                z = genfromtxt('{}{}/z{}.csv'.format(dataset, nz, zi), delimiter=',')[1:,2:][:,0]
            else:
                i = 1
                zlist = []
                while os.path.exists('{}{}/z{}{}.csv'.format(dataset, nz, zi, i)):
                    z = genfromtxt('{}{}/z{}{}.csv'.format(dataset, nz, zi, i), delimiter=',')[1:,2:][:,0]
                    if i > 1:
                        zlist[-1] = zlist[-1][:indices[i-2]]
                    zlist.append(z)
                    i += 1
                z = np.concatenate(zlist, axis=0)
            z = ma(z, N=N)
            stack.append(z)
            rgb = [255 - round((zi/8)*255), 255 - round((zi/8)*255), 255]
            colors.append('#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2]))
            labels.append(r'$z_{}$'.format(zi))

        maxindex = np.min([item.shape[0] for item in stack])
        stack = [item[:maxindex] * ((32**2/28**2 if dataset == "mnist" else 1.)) for item in stack]
        y = np.vstack(stack)

        x = epochs[:maxindex]
        ax = axes[idset, idx]
        ax.stackplot(x, y, edgecolor='black', linewidth=0.5, colors=colors, labels=labels)
        ax.set_xlim([5 + (N-1)*5, xmax])
        ax.set_xscale('log')
        ax.set_ylim(ylim)
        ax.yaxis.tick_right()
        ax.set_title(f"{nz} latent layer{'' if idx == 0 else 's'}", fontdict=dict(weight='bold')) if idset == 0 else None
        ax.legend(loc='lower left', prop={'size': 7})

axes[-1,-1].axis('off')
axes[0,0].set_ylabel('MNIST', fontdict=dict(weight='bold'), labelpad=30)
axes[1,0].set_ylabel('CIFAR-10', fontdict=dict(weight='bold'), labelpad=30)
axes[2,0].set_ylabel('ImageNet (32x32)', fontdict=dict(weight='bold'), labelpad=30)
fig.text(0.5, 0.06, '# Epochs', ha='center', va='center')
fig.text(0.112, 0.5, 'Bits per dimension', ha='center', va='center', rotation='vertical')

plt.savefig(f'stackplot.pdf', bbox_inches='tight')
plt.show()