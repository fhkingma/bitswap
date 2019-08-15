from utils.torch.rand import *
from tqdm import tqdm
import os
from torchvision import datasets, transforms
from torch.utils.data import *
from sklearn.preprocessing import KBinsDiscretizer

# function that returns discretization bin endpoints and centres
def discretize(nz, quantbits, type, device, model, dataset):
    # number of samples per bin
    ppb = 30

    # total number of samples (ppb * number of bins)
    nsamples = ppb * (1 << quantbits)

    with torch.no_grad():
        # check if there does already exists a file with the discretization bins saved into it
        if not os.path.exists(f'bins/{dataset}_nz{nz}_zendpoints{quantbits}.pt'):
            # set up an empty tensor for all the bins (number of latent variables, total dimension of latent, number of bins)
            # note that we do not include the first and last endpoints, because those will always be -inf and inf
            zendpoints = np.zeros((nz, np.prod(model.zdim), (1 << quantbits) - 1))
            zcentres = np.zeros((nz, np.prod(model.zdim), (1 << quantbits)))

            # top latent is fixed, so we can calculate the discretization bins without samples
            zbins = Bins(torch.zeros((1, 1, np.prod(model.zdim))), torch.ones((1, 1, np.prod(model.zdim))), quantbits)
            zendpoints[nz - 1] = zbins.endpoints().numpy()
            zcentres[nz - 1] = zbins.centres().numpy()

            # create class that scales up the data to [0,255] if called
            class ToInt:
                def __call__(self, pic):
                    return pic * 255

            # get the train-sets of the corresponding datasets
            if dataset == "cifar":
                transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
                train_set = datasets.CIFAR10(root="model/data/cifar", train=True, transform=transform_ops, download=True)
            elif dataset == "imagenet":
                transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
                train_set = modules.ImageNet(root='model/data/imagenet/train', file='train.npy', transform=transform_ops)
            else:
                transform_ops = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), ToInt()])
                train_set = datasets.MNIST(root="model/data/mnist", train=True, transform=transform_ops, download=True)

            # set-up a batch-loader for the dataset
            train_loader = DataLoader(
                dataset=train_set,
                batch_size=128, shuffle=True, drop_last=True)
            datapoints = list(train_loader)

            # concatenate the dataset with itself if the length is not sufficient
            while len(datapoints) < nsamples:
                datapoints += datapoints

            bs = 128 # batch size to iterate over
            batches = nsamples // bs # number of batches

            # use 16-bit values to reduce memory usage
            gen_samples = np.zeros((nz, nsamples) + model.zdim, dtype=np.float16)
            gen_samples[-1] = logistic_eps((nsamples,) + model.zdim, device="cpu", bound=1e-30).numpy()
            inf_samples = np.zeros((nz, nsamples) + model.zdim, dtype=np.float16)

            # iterate over the latent variables
            for zi in reversed(range(1, nz)):

                # obtain samples from the generative model
                iterator = tqdm(range(batches), desc=f"sampling z{zi} from gen")
                for bi in iterator:
                    mu, scale = model.generate(zi)(given=torch.from_numpy(gen_samples[zi][bi * bs: bi * bs + bs]).to(device).float())
                    gen_samples[zi - 1][bi * bs: bi * bs + bs] = transform(logistic_eps(mu.shape, device=device, bound=1e-30), mu, scale).to("cpu")

                # obtain samples from the inference model (using the dataset)
                iterator = tqdm(range(batches), desc=f"sampling z{nz - zi} from inf")
                for bi in iterator:
                    datapoint = datapoints[bi] if dataset == "imagenet" else datapoints[bi][0]
                    given = (datapoint if nz - zi - 1 == 0 else torch.from_numpy(inf_samples[nz - zi - 2][bi * bs: bi * bs + bs])).to(device).float()
                    mu, scale = model.infer(nz - zi - 1)(given=given)
                    inf_samples[nz - zi - 1][bi * bs: bi * bs + bs] = transform(logistic_eps(mu.shape, device=device, bound=1e-30), mu, scale).cpu().numpy()

            # get the discretization bins
            for zi in range(nz - 1):
                samples = np.concatenate([gen_samples[zi], inf_samples[zi]], axis=0)
                zendpoints[zi], zcentres[zi] = discretize_kbins(model, samples, quantbits, strategy='uniform')

            # move the discretization bins to the GPU
            zendpoints = torch.from_numpy(zendpoints)
            zcentres = torch.from_numpy(zcentres)

            # save the bins for reproducibility and speed purposes
            torch.save(zendpoints, f'bins/{dataset}_nz{nz}_zendpoints{quantbits}.pt')
            torch.save(zcentres, f'bins/{dataset}_nz{nz}_zcentres{quantbits}.pt')
        else:
            zendpoints = torch.load(f'bins/{dataset}_nz{nz}_zendpoints{quantbits}.pt',
                                    map_location=lambda storage, location: storage)
            zcentres = torch.load(f'bins/{dataset}_nz{nz}_zcentres{quantbits}.pt',
                                  map_location=lambda storage, location: storage)

    # cast the bins to the appropriate type (in our experiments always float64)
    return zendpoints.type(type).to(device), zcentres.type(type).to(device)

# function that exploits the KBinsDiscretizer from scikit-learn
# two strategy are available
# 1. uniform: bins with equal width
# 2. quantile: bins with equal frequency
def discretize_kbins(model, samples, quantbits, strategy):
    # reshape samples
    samples = samples.reshape(-1, np.prod(model.zdim))

    # apply discretization
    est = KBinsDiscretizer(n_bins=1 << quantbits, strategy=strategy)
    est.fit(samples)

    # obtain the discretization bins endpoints
    endpoints = np.array([np.array(ar) for ar in est.bin_edges_]).transpose()
    centres = (endpoints[:-1,:] + endpoints[1:,:]) / 2
    endpoints = endpoints[1:-1]

    return endpoints.transpose(), centres.transpose()