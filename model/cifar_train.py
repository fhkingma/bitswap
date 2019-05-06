import torch
import torch.utils.data
from torch import nn, optim
from torchvision import *
import socket
import os
import time
from datetime import datetime
import numpy as np
import argparse

from tensorboardX import SummaryWriter

import utils.torch.modules as modules
import utils.torch.rand as random

class Model(nn.Module):
    def __init__(self, xs=(3, 32, 32), nz=1, zchannels=16, nprocessing=1, kernel_size=3, resdepth=2,
                 reswidth=256, dropout_p=0., tag='', root_process=True):
        super().__init__()
        # default: disable compressing mode
        # if activated, tensors will be flattened
        self.compressing = False

        # hyperparameters
        self.xs = xs
        self.nz = nz
        self.zchannels = zchannels
        self.nprocessing = nprocessing
        # latent height/width is always 16,
        # the number of channels depends on the dataset
        self.zdim = (self.zchannels, 16, 16)
        self.resdepth = resdepth
        self.reswidth = reswidth
        self.kernel_size = kernel_size

        # apply these two factors (i.e. on the ELBO) in sequence and it results in "bits/dim"
        # factor to convert "nats" to bits
        self.bitsscale = np.log2(np.e)
        # factor to divide by the data dimension
        self.perdimsscale = 1. / np.prod(self.xs)

        # calculate processing layers convolutions options
        # kernel/filter is 5, so in order to ensure same-size outputs, we have to pad by 2
        padding_proc = (5 - 1) / 2
        assert padding_proc.is_integer()
        padding_proc = int(padding_proc)

        # calculate other convolutions options
        padding = (self.kernel_size - 1) / 2
        assert padding.is_integer()
        padding = int(padding)

        # create loggers
        self.tag = tag
        if root_process:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs/cifar/', current_time + '_' + socket.gethostname() + tag)
            self.log_dir = log_dir
            self.logger = SummaryWriter(log_dir=self.log_dir)

        # set-up current "best elbo"
        self.best_elbo = np.inf

        # distribute ResNet blocks over latent layers
        resdepth = [0] * (self.nz)
        i = 0
        for _ in range(self.resdepth):
            i = 0 if i == (self.nz) else i
            resdepth[i] += 1
            i += 1

        # reduce initial variance of distributions corresponding
        # to latent layers if latent nz increases
        scale = 1.0 / (self.nz ** 0.5)

        # activations
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ELU()
        self.actresnet = nn.ELU()

        # Below we build up the main model architecture of the inference- and generative-models
        # All the architecure components are built up from different custom are existing PyTorch modules

        # <===== INFERENCE MODEL =====>
        # the bottom (zi=1) inference model
        self.infer_in = nn.Sequential(
            # shape: [1,32,32] -> [4,16,16]
            modules.Squeeze2d(factor=2),

            # shape: [4,16,16] -> [32,16,16]
            modules.WnConv2d(4 * xs[0],
                             self.reswidth,
                             5,
                             1,
                             padding_proc,
                             init_scale=1.0,
                             loggain=True),
            self.act
        )
        self.infer_res0 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                5,
                                1,
                                padding_proc,
                                self.nprocessing,
                                dropout_p,
                                self.actresnet),
            self.act
        ) if self.nprocessing > 0 else modules.Pass()

        self.infer_res1 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                self.kernel_size,
                                1,
                                padding,
                                resdepth[0],
                                dropout_p,
                                self.actresnet),
            self.act
        ) if resdepth[0] > 0 else modules.Pass()

        # shape: [32,16,16] -> [1,16,16]
        self.infer_mu = modules.WnConv2d(self.reswidth,
                                         self.zchannels,
                                         self.kernel_size,
                                         1,
                                         padding,
                                         init_scale=scale if self.nz > 1 else 2 ** 0.5 * scale)

        # shape: [32,16,16] -> [1,16,16]
        self.infer_scale = modules.WnConv2d(self.reswidth,
                                            self.zchannels,
                                            self.kernel_size,
                                            1,
                                            padding,
                                            init_scale=scale if self.nz > 1 else 2 ** 0.5 * scale)

        # <===== DEEP INFERENCE MODEL =====>
        # the deeper (zi > 1) inference models
        self.deepinfer_in = nn.ModuleList([
            # shape: [1,16,16] -> [32,16,16]
            nn.Sequential(
                modules.WnConv2d(self.zchannels,
                                 self.reswidth,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=1.0,
                                 loggain=True),
                self.act
            )
            for _ in range(self.nz - 1)])

        self.deepinfer_res = nn.ModuleList([
            # shape: [32,16,16] -> [32,16,16]
            nn.Sequential(
                modules.ResNetBlock(self.reswidth,
                                    self.reswidth,
                                    self.kernel_size,
                                    1,
                                    padding,
                                    resdepth[i + 1],
                                    dropout_p,
                                    self.actresnet),
                self.act
            ) if resdepth[i + 1] > 0 else modules.Pass()
            for i in range(self.nz - 1)])

        self.deepinfer_mu = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale if i < self.nz - 2 else 2 ** 0.5 * scale)
            )
            for i in range(self.nz - 1)])

        self.deepinfer_scale = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale if i < self.nz - 2 else 2 ** 0.5 * scale)
            )
            for i in range(self.nz - 1)])

        # <===== DEEP GENERATIVE MODEL =====>
        # the deeper (zi > 1) generative models
        self.deepgen_in = nn.ModuleList([
            # shape: [1,16,16] -> [32,16,16]
            nn.Sequential(
                modules.WnConv2d(self.zchannels,
                                 self.reswidth,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=1.0,
                                 loggain=True),
                self.act
            )
            for _ in range(self.nz - 1)])

        self.deepgen_res = nn.ModuleList([
            # shape: [32,16,16] -> [32,16,16]
            nn.Sequential(
                modules.ResNetBlock(self.reswidth,
                                    self.reswidth,
                                    self.kernel_size,
                                    1,
                                    padding,
                                    resdepth[i + 1],
                                    dropout_p,
                                    self.actresnet),
                self.act
            ) if resdepth[i + 1] > 0 else modules.Pass()
            for i in range(self.nz - 1)])

        self.deepgen_mu = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale)
            )
            for _ in range(self.nz - 1)])

        self.deepgen_scale = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding, init_scale=scale)
            )
            for _ in range(self.nz - 1)])

        # <===== GENERATIVE MODEL =====>
        # the bottom (zi = 1) inference model
        self.gen_in = nn.Sequential(
            # shape: [1,16,16] -> [32,16,16]
            modules.WnConv2d(self.zchannels,
                             self.reswidth,
                             self.kernel_size,
                             1,
                             padding,
                             init_scale=1.0,
                             loggain=True),
            self.act
        )

        self.gen_res1 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                self.kernel_size,
                                1,
                                padding,
                                resdepth[0],
                                dropout_p,
                                self.actresnet),
            self.act
        ) if resdepth[0] > 0 else modules.Pass()

        self.gen_res0 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                5,
                                1,
                                padding_proc,
                                self.nprocessing,
                                dropout_p,
                                self.actresnet),
            self.act
        ) if self.nprocessing > 0 else modules.Pass()

        self.gen_mu = nn.Sequential(
            # shape: [32,16,16] -> [4,16,16]
            modules.WnConv2d(self.reswidth,
                             4 * xs[0],
                             self.kernel_size,
                             1,
                             padding,
                             init_scale=0.1),
            # shape: [4,16,16] -> [1,32,23]
            modules.UnSqueeze2d(factor=2)
        )

        # the scale parameter of the bottom (zi = 1) generative model is modelled unconditional
        self.gen_scale = nn.Parameter(torch.Tensor(*self.xs))
        nn.init.zeros_(self.gen_scale)

    # function to set the model to compression mode
    def compress(self, compress=True):
        self.compressing = compress

    # function that only takes in the layer number and returns a distribution based on that
    def infer(self, i):
        # nested function that takes in the "given" value of the conditional Logistic distribution
        # and returns the mu and scale parameters of that distribution
        def distribution(given):
            h = given

            # if compressing, the input might not be float32, so we'll have to convert it first
            if self.compressing:
                type = h.type()
                h = h.float()

            # bottom latent layer
            if i == 0:
                # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
                if self.compressing:
                    h = h.view((-1,) + self.xs)
                # also, when NOT compressing, the input is not scaled from [0,255] to [-1,1]
                else:
                    h = (h - 127.5) / 127.5

                # input convolution
                h = self.infer_in(h)

                # processing ResNet blocks
                h = self.infer_res0(h)

                # other ResNet blocks
                h = self.infer_res1(h)

                # mu parameter of the conditional Logistic distribution
                mu = self.infer_mu(h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * self.sigmoid(self.infer_scale(h) + 2.)

            # deeper latent layers
            else:
                # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
                if self.compressing:
                    h = h.view((-1,) + self.zdim)

                # input convolution
                h = self.deepinfer_in[i - 1](h)

                # other ResNet blocks
                h = self.deepinfer_res[i - 1](h)

                # mu parameter of the conditional Logistic distribution
                mu = self.deepinfer_mu[i - 1](h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * self.sigmoid(self.deepinfer_scale[i - 1](h) + 2.)

            if self.compressing:
                # if compressing, the "batch-size" can only be 1
                assert mu.shape[0] == 1

                # flatten the Tensors back and convert back to the input datatype
                mu = mu.view(np.prod(self.zdim)).type(type)
                scale = scale.view(np.prod(self.zdim)).type(type)
            return mu, scale

        return distribution

    # function that only takes in the layer number and returns a distribution based on that
    def generate(self, i):
        # nested function that takes in the "given" value of the conditional Logistic distribution
        # and returns the mu and scale parameters of that distribution
        def distribution(given):
            h = given

            # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
            # also, the input might not be float32, so we'll have to convert it first
            if self.compressing:
                type = h.type()
                h = h.float()
                h = h.view((-1,) + self.zdim)

            # bottom latent layer
            if i == 0:
                # input convolution
                h = self.gen_in(h)

                # processing ResNet blocks
                h = self.gen_res1(h)

                # other ResNet blocks
                h = self.gen_res0(h)

                # mu parameter of the conditional Logistic distribution
                mu = self.gen_mu(h)

                # scale parameter of the conditional Logistic distribution
                # set a minimal value for the scale parameter of the bottom generative model
                scale = ((2. / 255.) / 8.) + modules.softplus(self.gen_scale)

            # deeper latent layers
            else:
                # input convolution
                h = self.deepgen_in[i - 1](h)

                # other ResNet blocks
                h = self.deepgen_res[i - 1](h)

                # mu parameter of the conditional Logistic distribution
                mu = self.deepgen_mu[i - 1](h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * modules.softplus(self.deepgen_scale[i - 1](h) + np.log(np.exp(1.) - 1.))


            if self.compressing:
                # if compressing, the "batch-size" can only be 1
                assert mu.shape[0] == 1

                # flatten the Tensors back and convert back to the input datatype
                mu = mu.view(np.prod(self.xs if i == 0 else self.zdim)).type(type)
                scale = scale.view(np.prod(self.xs if i == 0 else self.zdim)).type(type)
            return mu, scale

        return distribution

    # function that takes as input the data and outputs all the components of the ELBO + the latent samples
    def loss(self, x):
        # tensor to store inference model losses
        logenc = torch.zeros((self.nz, x.shape[0], self.zdim[0]), device=x.device)

        # tensor to store the generative model losses
        logdec = torch.zeros((self.nz, x.shape[0], self.zdim[0]), device=x.device)

        # tensor to store the latent samples
        zsamples = torch.zeros((self.nz, x.shape[0], np.prod(self.zdim)), device=x.device)

        for i in range(self.nz):
            # inference model
            # get the parameters of inference distribution i given x (if i == 0) or z (otherwise)
            mu, scale = self.infer(i)(given=x if i == 0 else z)

            # sample untransformed sample from Logistic distribution (mu=0, scale=1)
            eps = random.logistic_eps(mu.shape, device=mu.device)
            # reparameterization trick: transform using obtained parameters
            z_next = random.transform(eps, mu, scale)

            # store the inference model loss
            zsamples[i] = z_next.flatten(1)
            logq = torch.sum(random.logistic_logp(mu, scale, z_next), dim=2)
            logenc[i] += logq

            # generative model
            # get the parameters of inference distribution i given z
            mu, scale = self.generate(i)(given=z_next)

            # store the generative model loss
            if i == 0:
                # if bottom (zi = 1) generative model, evaluate loss using discretized Logistic distribution
                logp = torch.sum(random.discretized_logistic_logp(mu, scale, x), dim=1)
                logrecon = logp

            else:
                logp = torch.sum(random.logistic_logp(mu, scale, x if i == 0 else z), dim=2)
                logdec[i - 1] += logp

            z = z_next

        # store the prior loss
        logp = torch.sum(random.logistic_logp(torch.zeros(1, device=x.device), torch.ones(1, device=x.device), z), dim=2)
        logdec[self.nz - 1] += logp

        # convert from "nats" to bits
        logenc = torch.mean(logenc, dim=1) * self.bitsscale
        logdec = torch.mean(logdec, dim=1) * self.bitsscale
        logrecon = torch.mean(logrecon) * self.bitsscale
        return logrecon, logdec, logenc, zsamples

    # function to sample from the model (using the generative model)
    def sample(self, device, epoch, num=64):
        # sample "num" latent variables from the prior
        z = random.logistic_eps(((num,) + self.zdim), device=device)

        # sample from the generative distribution(s)
        for i in reversed(range(self.nz)):
            mu, scale = self.generate(i)(given=z)
            eps = random.logistic_eps(mu.shape, device=device)
            z_prev = random.transform(eps, mu, scale)
            z = z_prev

        # scale up from [-1,1] to [0,255]
        x_cont = (z * 127.5) + 127.5

        # ensure that [0,255]
        x = torch.clamp(x_cont, 0, 255)

        # scale from [0,255] to [0,1] and convert to right shape
        x_sample = x.float() / 255.
        x_sample = x_sample.view((num,) + self.xs)

        # make grid out of "num" samples
        x_grid = utils.make_grid(x_sample)

        # log
        self.logger.add_image('x_sample', x_grid, epoch)

    # function to sample a reconstruction of input data
    def reconstruct(self, x_orig, device, epoch):
        # take only first 32 datapoints of the input
        # otherwise the output image grid may be too big for visualization
        x_orig = x_orig[:32, :, :, :].to(device)

        # sample from the bottom (zi = 1) inference model
        mu, scale = self.infer(0)(given=x_orig)
        eps = random.logistic_eps(mu.shape, device=device)
        z = random.transform(eps, mu, scale)  # sample zs

        # sample from the bottom (zi = 1) generative model
        mu, scale = self.generate(0)(given=z)
        x_eps = random.logistic_eps(mu.shape, device=device)
        x_cont = random.transform(x_eps, mu, scale)

        # scale up from [-1.1] to [0,255]
        x_cont = (x_cont * 127.5) + 127.5

        # esnure that [0,255]
        x_sample = torch.clamp(x_cont, 0, 255)

        # scale from [0,255] to [0,1] and convert to right shape
        x_sample = x_sample.float() / 255.
        x_orig = x_orig.float() / 255.

        # concatenate the input data and the sampled reconstructions for comparison
        x_with_recon = torch.cat((x_orig, x_sample))

        # make a grid out of the original data and the reconstruction samples
        x_with_recon = x_with_recon.view((2 * x_orig.shape[0],) + self.xs)
        x_grid = utils.make_grid(x_with_recon)

        # log
        self.logger.add_image('x_reconstruct', x_grid, epoch)


def warmup(model, device, data_loader, warmup_batches, root_process):
    # convert model to evaluation mode (no Dropout etc.)
    model.eval()

    # prepare initialization batch
    for batch_idx, (image, _) in enumerate(data_loader):
        # stack image with to current stack
        warmup_images = torch.cat((warmup_images, image), dim=0) \
            if batch_idx != 0 else image

        # stop stacking batches if reaching limit
        if batch_idx + 1 == warmup_batches:
            break

    # set the stack to current device
    warmup_images = warmup_images.to(device)

    # do one 'special' forward pass to initialize parameters
    with modules.init_mode():
        logrecon, logdec, logenc, _ = model.loss(warmup_images)

    # log
    if root_process:
        logdec = torch.sum(logdec, dim=1)
        logenc = torch.sum(logenc, dim=1)

        elbo = -logrecon + torch.sum(-logdec + logenc)

        elbo = elbo.detach().cpu().numpy() * model.perdimsscale
        entrecon = -logrecon.detach().cpu().numpy() * model.perdimsscale
        entdec = -logdec.detach().cpu().numpy() * model.perdimsscale
        entenc = -logenc.detach().cpu().numpy() * model.perdimsscale

        kl = entdec - entenc

        print(f'====> Epoch: {0} Average loss: {elbo:.4f}')
        model.logger.add_text('architecture', f"{model}", 0)
        model.logger.add_scalar('elbo/train', elbo, 0)
        model.logger.add_scalar('x/reconstruction/train', entrecon, 0)
        for i in range(1, logdec.shape[0] + 1):
            model.logger.add_scalar(f'z{i}/encoder/train', entenc[i - 1], 0)
            model.logger.add_scalar(f'z{i}/decoder/train', entdec[i - 1], 0)
            model.logger.add_scalar(f'z{i}/KL/train', kl[i - 1], 0)


def train(model, device, epoch, data_loader, optimizer, ema, log_interval, root_process, schedule=True, decay=0.99995):
    # convert model to train mode (activate Dropout etc.)
    model.train()

    # get number of batches
    nbatches = data_loader.batch_sampler.sampler.num_samples // data_loader.batch_size

    # switch to parameters not affected by exponential moving average decay
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = ema.get_default(name)

    # setup training metrics
    if root_process:
        elbos = torch.zeros((nbatches), device=device)
        logrecons = torch.zeros((nbatches), device=device)
        logdecs = torch.zeros((nbatches, model.nz), device=device)
        logencs = torch.zeros((nbatches, model.nz), device=device)

    if root_process:
        start_time = time.time()

    # allocate memory for data
    data = torch.zeros((data_loader.batch_size,) + model.xs, device=device)

    # enumerate over the batches
    for batch_idx, (batch, _) in enumerate(data_loader):
        # keep track of the global step
        global_step = (epoch - 1) * len(data_loader) + (batch_idx + 1)

        # update the learning rate according to schedule
        if schedule:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                lr = lr_step(global_step, lr, decay=decay)
                param_group['lr'] = lr

        # empty all the gradients stored
        optimizer.zero_grad()

        # copy the mini-batch in the pre-allocated data-variable
        data.copy_(batch)

        # evaluate the data under the model and calculate ELBO components
        logrecon, logdec, logenc, zsamples = model.loss(data)

        # free bits technique, in order to prevent posterior collapse
        bits_pc = 1.
        kl = torch.sum(torch.max(-logdec + logenc, bits_pc * torch.ones((model.nz, model.zdim[0]), device=device)))

        # compute the inference- and generative-model loss
        logdec = torch.sum(logdec, dim=1)
        logenc = torch.sum(logenc, dim=1)

        # construct ELBO
        elbo = -logrecon + kl

        # scale by image dimensions to get "bits/dim"
        elbo *= model.perdimsscale
        logrecon *= model.perdimsscale
        logdec *= model.perdimsscale
        logenc *= model.perdimsscale

        # calculate gradients
        elbo.backward()

        # take gradient step
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1., norm_type=2)
        optimizer.step()

        # log gradient norm
        if root_process:
            model.logger.add_scalar('gnorm', total_norm, global_step)

        # do ema update on parameters used for evaluation
        if root_process:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        ema(name, param.data)

        # log
        if root_process:
            elbos[batch_idx] += elbo
            logrecons[batch_idx] += logrecon
            logdecs[batch_idx] += logdec
            logencs[batch_idx] += logenc

        # log and save parameters
        if root_process and batch_idx % log_interval == 0 and log_interval < nbatches:
            # print metrics to console
            print(f'Train Epoch: {epoch} [{batch_idx}/{nbatches} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {elbo.item():.6f}\tGnorm: {total_norm:.2f}\tSteps/sec: {(time.time() - start_time) / (batch_idx + 1):.3f}')


            model.logger.add_scalar('step-sec', (time.time() - start_time) / (batch_idx + 1), global_step)
            entrecon = -logrecon
            entdec = -logdec
            entenc = -logenc
            kl = entdec - entenc

            # log
            model.logger.add_scalar('elbo/train', elbo, global_step)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            model.logger.add_scalar('lr', lr, global_step)

            model.logger.add_scalar('x/reconstruction/train', entrecon, global_step)
            for i in range(1, logdec.shape[0] + 1):
                model.logger.add_scalar(f'z{i}/encoder/train', entenc[i - 1], global_step)
                model.logger.add_scalar(f'z{i}/decoder/train', entdec[i - 1], global_step)
                model.logger.add_scalar(f'z{i}/KL/train', kl[i - 1], global_step)

    # save training params, to be able to return to these values after evaluation
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register_default(name, param.data)

    # print the average loss of the epoch to the console
    if root_process:
        elbo = torch.mean(elbos).detach().cpu().numpy()
        print(f'====> Epoch: {epoch} Average loss: {elbo:.4f}')


def test(model, device, epoch, ema, data_loader, tag, root_process):
    # convert model to evaluation mode (no Dropout etc.)
    model.eval()

    # setup the reconstruction dataset
    recon_dataset = None
    nbatches = data_loader.batch_sampler.sampler.num_samples // data_loader.batch_size
    recon_batch_idx = int(torch.Tensor(1).random_(0, nbatches - 1))

    # setup testing metrics
    if root_process:
        logrecons = torch.zeros((nbatches), device=device)
        logdecs = torch.zeros((nbatches, model.nz), device=device)
        logencs = torch.zeros((nbatches, model.nz), device=device)

    elbos = []

    # switch to EMA parameters for evaluation
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = ema.get_ema(name)

    # allocate memory for the input data
    data = torch.zeros((data_loader.batch_size,) + model.xs, device=device)

    # enumerate over the batches
    for batch_idx, (batch, _) in enumerate(data_loader):
        # save batch for reconstruction
        if batch_idx == recon_batch_idx:
            recon_dataset = data

        # copy the mini-batch in the pre-allocated data-variable
        data.copy_(batch)

        with torch.no_grad():
            # evaluate the data under the model and calculate ELBO components
            logrecon, logdec, logenc, _ = model.loss(data)

            # construct the ELBO
            elbo = -logrecon + torch.sum(-logdec + logenc)

            # compute the inference- and generative-model loss
            logdec = torch.sum(logdec, dim=1)
            logenc = torch.sum(logenc, dim=1)

        if root_process:
            # scale by image dimensions to get "bits/dim"
            elbo *= model.perdimsscale
            logrecon *= model.perdimsscale
            logdec *= model.perdimsscale
            logenc *= model.perdimsscale

            elbos.append(elbo.item())

            # log
            logrecons[batch_idx] += logrecon
            logdecs[batch_idx] += logdec
            logencs[batch_idx] += logenc

    if root_process:
        elbo = np.mean(elbos)

        entrecon = -torch.mean(logrecons).detach().cpu().numpy()
        entdec = -torch.mean(logdecs, dim=0).detach().cpu().numpy()
        entenc = -torch.mean(logencs, dim=0).detach().cpu().numpy()
        kl = entdec - entenc

        # print metrics to console and Tensorboard
        print(f'\nEpoch: {epoch}\tTest loss: {elbo:.6f}')
        model.logger.add_scalar('elbo/test', elbo, epoch)

        # log to Tensorboard
        model.logger.add_scalar('x/reconstruction/test', entrecon, epoch)
        for i in range(1, logdec.shape[0] + 1):
            model.logger.add_scalar(f'z{i}/encoder/test', entenc[i - 1], epoch)
            model.logger.add_scalar(f'z{i}/decoder/test', entdec[i - 1], epoch)
            model.logger.add_scalar(f'z{i}/KL/test', kl[i - 1], epoch)

        # if the current ELBO is better than the ELBO's before, save parameters
        if elbo < model.best_elbo and not np.isnan(elbo):
            model.logger.add_scalar('elbo/besttest', elbo, epoch)
            if not os.path.exists(f'params/cifar/'):
                os.makedirs(f'params/cifar/')
            torch.save(model.state_dict(), f'params/cifar/{tag}')
            if epoch % 25 == 0:
                torch.save(model.state_dict(), f'params/cifar/epoch{epoch}_{tag}')
            print("saved params\n")
            model.best_elbo = elbo

            model.sample(device, epoch)
            model.reconstruct(recon_dataset, device, epoch)
        else:
            print("loss did not improve\n")

# learning rate schedule
def lr_step(step, curr_lr, decay=0.99995, min_lr=5e-4):
    # only decay after certain point
    # and decay down until minimal value
    if curr_lr > min_lr:
        curr_lr *= decay
        return curr_lr
    return curr_lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=99, type=int, help="seed for experiment reproducibility")
    parser.add_argument('--nz', default=8, type=int, help="number of latent variables, greater or equal to 1")
    parser.add_argument('--zchannels', default=8, type=int, help="number of channels for the latent variables")
    parser.add_argument('--nprocessing', default=4, type=int, help='number of processing layers')
    parser.add_argument('--gpu', default=0, type=int, help="number of gpu's to distribute optimization over")
    parser.add_argument('--interval', default=100, type=int, help="interval for logging/printing of relevant values")
    parser.add_argument('--epochs', default=10000000000, type=int, help="number of sweeps over the dataset (epochs)")
    parser.add_argument('--blocks', default=8, type=int, help="number of ResNet blocks")
    parser.add_argument('--width', default=256, type=int, help="number of channels in the convolutions in the ResNet blocks")
    parser.add_argument('--dropout', default=0.3, type=float, help="dropout rate of the hidden units")
    parser.add_argument('--kernel', default=3, type=int, help="size of the convolutional filter (kernel) in the ResNet blocks")
    parser.add_argument('--batch', default=16, type=int, help="size of the mini-batch for gradient descent")
    parser.add_argument('--dist', default=0, type=int, help="distribute (1) over different gpu's and use Horovod to do so, or not (0)")
    parser.add_argument('--lr', default=2e-3, type=float, help="learning rate gradient descent")
    parser.add_argument('--schedule', default=1, type=float, help="learning rate schedule: yes (1) or no (0)")
    parser.add_argument('--decay', default=0.999995, type=float, help="decay of the learning rate when using learning rate schedule")

    args = parser.parse_args()
    print(args)  # print all the hyperparameters

    # store hyperparameters in variables
    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch
    nz = args.nz
    zchannels = args.zchannels
    nprocessing = args.nprocessing
    gpu = args.gpu
    blocks = args.blocks
    width = args.width
    log_interval = args.interval
    dropout = args.dropout
    kernel = args.kernel
    distributed = args.dist
    lr = args.lr
    schedule = True if args.schedule == 1 else False
    decay = args.decay
    assert nz > 0

    # setup seeds to maintain experiment reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Distributed: set up horovod over multiple gpu's
    if distributed:
        import horovod.torch as hvd

        # initialize horovod
        hvd.init()

        # pin gpu to "local rank" (see Horovod documentation)
        torch.cuda.set_device(hvd.local_rank())
        print(f"My local rank is {hvd.local_rank()}")

        # distribute mini-batches over the different gpu's
        batch_size //= hvd.size()

    # string-tag for logging
    tag = f'nz{nz}'

    # define the "root process": only one of the gpu's has to log relevant values
    # set only one gpu as root process
    root_process = True
    if distributed and not hvd.rank() == 0:
        root_process = False

    # set GPU/CPU options
    use_cuda = torch.cuda.is_available()
    cudastring = "cuda" if distributed else f"cuda:{gpu}"
    device = torch.device(cudastring if use_cuda else "cpu")

    # set number of workers and pin the memory if we distribute over multiple gpu's
    # (see Dataloader docs of PyTorch)
    kwargs = {'num_workers': 8, 'pin_memory': True} if distributed else {}

    # create class that scales up the data to [0,255] if called
    class ToInt:
        def __call__(self, pic):
            return pic * 255

    # set data pre-processing transforms
    transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])

    # store CIFAR data shape
    xs = (3, 32, 32)

    if root_process:
        print("Load model")

    # build model from hyperparameters
    model = Model(xs=xs,
                  kernel_size=kernel,
                  nprocessing=nprocessing,
                  nz=nz,
                  zchannels=zchannels,
                  resdepth=blocks,
                  reswidth=width,
                  dropout_p=dropout,
                  tag=tag,
                  root_process=root_process).to(device)

    # set up Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # if we distribute over multiple GPU's, set up Horovod's distributed optimizer wrapper around the optimizer
    if distributed:
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),
                                             compression=hvd.Compression.fp16)

    # print and log amount of parameters
    if root_process:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_parameters = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Number of trainable parameters in model: {num_parameters}')
        model.logger.add_text(f'hyperparams', '{num_parameters}', 0)

    if root_process:
        print("Load data")

    # get dataset for training and testing of the model
    if root_process:
        train_set = datasets.CIFAR10(root="data/cifar", train=True, transform=transform_ops, download=True)
        test_set = datasets.CIFAR10(root="data/cifar", train=False, transform=transform_ops, download=True)

    # if distributed over multiple GPU's, set-up barrier a barrier ensuring that all the processes have loaded the data
    if distributed:
        hvd.allreduce_(torch.Tensor(0), name='barrier')

    # get dataset for training and testing of the model
    if not root_process:
        train_set = datasets.CIFAR10(root="data/cifar", train=True, transform=transform_ops, download=True)
        test_set = datasets.CIFAR10(root="data/cifar", train=False, transform=transform_ops, download=True)

    # setup data sampler
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=hvd.size(), rank=hvd.rank())
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_set, num_replicas=hvd.size(), rank=hvd.rank())

    # setup mini-batch enumerator for both train-set and test-set
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, sampler=train_sampler if distributed else None,
        batch_size=batch_size, shuffle=False if distributed else True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, sampler=test_sampler if distributed else None,
        batch_size=batch_size, shuffle=False if distributed else True, drop_last=True, **kwargs)

    # Distributed: broadcast parameters to all the processes
    if distributed:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    print("Data Dependent Initialization") if root_process else print("Data Dependent Initialization with ya!")
    # data-dependent initialization
    warmup(model, device, train_loader, 25, root_process)

    # if distributed over multiple GPU's, set-up a barrier ensuring that all the processes have initialized the models
    if distributed:
        hvd.allreduce_(torch.Tensor(0), name='barrier')

    # setup exponential moving average decay (EMA) for the parameters.
    # This basically means maintaining two sets of parameters during training/testing:
    # 1. parameters that are the result of EMA
    # 2. parameters not affected by EMA
    # The (1)st parameters are only active during test-time.
    ema = modules.EMA(0.999)
    with torch.no_grad():
        for name, param in model.named_parameters():
            # only parameters optimized using gradient-descent are relevant here
            if param.requires_grad:
                # register (1) parameters
                ema.register_ema(name, param.data)
                # register (2) parameters
                ema.register_default(name, param.data)

    # initial test loss
    test(model, device, 0, ema, test_loader, tag, root_process)

    # do the training loop and run over the test-set 1/5 epochs.
    print("Training") if root_process else print("Training with ya!")
    for epoch in range(1, epochs + 1):
        train(model, device, epoch, train_loader, optimizer, ema, log_interval, root_process, schedule, decay)
        if epoch % 5 == 0:
            test(model, device, epoch, ema, test_loader, tag, root_process)
