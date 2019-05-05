import numpy as np
import matplotlib.pyplot as plt

# function to plot a line + error bars
def plot_with_error_bars(ax, data, color):
    assert data.ndim == 2
    data_x = np.arange(data.shape[1])
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    ax.plot(data_x, data_mean, color=color)
    ax.fill_between(data_x, data_mean-data_std, data_mean+data_std, color=color, alpha=0.4)

# set-up plot
fig, axes = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(15,8))

# iterate over the three datasets and create plots and data for results tables
for idset, dataset in enumerate(["mnist", "cifar", "imagenet"]):
    bits = 10
    latents = [2, 4] if idset == 2 else [2, 4, 8]
    ylim = [[1.0, 3.0], [3.0, 8.0], [4.0, 8.0]][idset]

    for i, nz in enumerate(latents):
        nets = {}
        elbos = {}
        cmas = {}
        total = {}
        for scheme in ["bitswap", "bbans"] if nz > 1 else ["bbans"]:
            nets[scheme] = np.load("{}{}/{}_{}bits_{}.npy".format(dataset, nz, scheme, bits, "nets")) * (32**2/28**2 if dataset == "mnist" else 1.)
            elbos[scheme] = np.load("{}{}/{}_{}bits_{}.npy".format(dataset, nz, scheme, bits, "elbos")) * (32 ** 2 / 28 ** 2 if dataset == "mnist" else 1.)
            cmas[scheme] = np.load("{}{}/{}_{}bits_{}.npy".format(dataset, nz, scheme, bits, "cmas")) * (32**2/28**2 if dataset == "mnist" else 1.)
            total[scheme] = np.load("{}{}/{}_{}bits_{}.npy".format(dataset, nz, scheme, bits, "total"))
        timesteps = nets["bbans"].shape[1]
        print("")
        ax = axes[idset, i]
        for scheme in ["bitswap", "bbans"] if nz > 1 else ["bbans"]:
            swap = 1 if scheme == "bitswap" else 0
            if nz == 1:
                color = 'darkorange'
            elif nz > 1 and swap:
                color = 'g'
            else:
                color = 'r'
            plot_with_error_bars(ax, cmas[scheme], color)

        nets = np.concatenate([nets["bitswap"], nets["bbans"]], axis=0) if nz > 1 else nets["bbans"]
        ax.plot(np.arange(timesteps), nets.mean() * np.ones(timesteps), color='b', linestyle='-.')
        ax.fill_between(np.arange(timesteps),
                                        (nets.mean() - nets.std()) * np.ones(timesteps),
                                        (nets.mean() + nets.std()) * np.ones(timesteps),
                                        color='b', alpha=0.1)
        elbos = np.concatenate([elbos["bitswap"]], axis=0) if nz > 1 else elbos["bbans"]

        ax.set_ylim(*ylim)
        ax.set_xlim(0, timesteps)
        ax.yaxis.tick_right()
        ax.set_title(f"{nz} latent layer{'' if nz == 1 else 's'}", fontdict=dict(weight='bold')) if idset == 0 else None

        print(f"{dataset} - {nz} latents - {bits} bits")
        for scheme in ["bbans", "bitswap"] if nz > 1 else ["bbans"]:
            cma = cmas[scheme]
            print(f"${elbos.mean():.2f} $params ${nets.mean():.2f} \pm {nets.std():.2f}$ & {'BB-ANS' if scheme == 'bbans' else 'Bit-Swap'} & ${cma[:,0].mean():.2f} \pm {cma[:,0].std():.2f}$ & ${cma[:,49].mean():.2f} \pm {cma[:,49].std():.2f}$ & ${cma[:,99].mean():.2f} \pm {cma[:,99].std():.2f}$")

axes[-1,-1].axis('off')
axes[0,0].set_ylabel('MNIST', fontdict=dict(weight='bold'), labelpad=30)
axes[1,0].set_ylabel('CIFAR-10', fontdict=dict(weight='bold'), labelpad=30)
axes[2,0].set_ylabel('ImageNet (32x32)', fontdict=dict(weight='bold'), labelpad=30)
fig.text(0.5, 0.06, '# Datapoints compressed so far', ha='center', va='center')
fig.text(0.112, 0.5, 'Cumulative compression rate (bits/dim)', ha='center', va='center', rotation='vertical')

plt.savefig('cma.pdf', bbox_inches='tight')
plt.show()
