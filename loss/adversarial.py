from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args['type'] == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args['type'] == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args['type'] == 'ADAMax':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args['type'] == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': 1e-08}

    kwargs['lr'] = args['lr']
    kwargs['weight_decay'] = args['weight_decay']

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args['decay_type'] == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args['lr_decay'],
            gamma=args['gamma']
        )
    elif args['decay_type'].find('step') >= 0:
        milestones = args['decay_type'].split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args['gamma']
        )
    elif args['decay_type'] == 'plateau':
        scheduler = lrs.ReduceLROnPlateau(
            my_optimizer,
            mode='max',
            factor=args['gamma'],
            patience=args['patience'],
            threshold=0.01, # metric to be used is psnr
            threshold_mode='abs',
            verbose=True
        )

    return scheduler


class Adversarial(nn.Module):
    def __init__(self, PatchSize, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = 1
        if gan_type == 'T_WGAN_GP':
            self.discriminator = discriminator.Temporal_Discriminator(PatchSize)
        elif gan_type == 'FI_GAN':
            self.discriminator = discriminator.FI_Discriminator(PatchSize)
        elif gan_type == 'FI_Cond_GAN':
            self.discriminator = discriminator.FI_Cond_Discriminator(PatchSize)
        elif gan_type == 'STGAN':
            self.discriminator = discriminator.ST_Discriminator(PatchSize)
        else:
            self.discriminator = discriminator.Discriminator(PatchSize, gan_type)
        
        # optimizer
        if gan_type != 'WGAN_GP' and gan_type != 'T_WGAN_GP':
            self.optimizer = make_optimizer(args={'type': 'ADAMax', 'lr': 0.001, 'weight_decay': 0}, my_model=self.discriminator)
        else:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=1e-5
            )
        self.scheduler = make_scheduler(args={'decay_type': 'plateau', 'gamma': 0.5, 'patience': 5}, my_optimizer=self.optimizer)

    def forward(self, fake, real, input_frames=None):
        # if len(input_frames) == 4:
        #     input_frames = input_frames[1:3]
        fake_detach = fake.detach()

        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            # discriminator forward pass
            if self.gan_type in ['T_WGAN_GP', 'FI_Cond_GAN', 'STGAN']:
                d_fake = self.discriminator(input_frames[:, 0], fake_detach, input_frames[:, 1])
                d_real = self.discriminator(input_frames[:, 0], real, input_frames[:, 1])
            elif self.gan_type == 'FI_GAN':
                d_01 = self.discriminator(input_frames[:, 0], fake_detach)
                d_12 = self.discriminator(fake_detach, input_frames[:, 1])
            else:
                d_fake = self.discriminator(fake_detach)
                d_real = self.discriminator(real)

            # compute discriminator loss
            if self.gan_type in ['GAN', 'FI_Cond_GAN', 'STGAN']:
                label_fake = torch.zeros_like(d_fake)
                label_real = torch.ones_like(d_real)
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type == 'FI_GAN':
                label_01 = torch.zeros_like(d_01)
                label_12 = torch.ones_like(d_12)
                loss_d = F.binary_cross_entropy_with_logits(d_01, label_01) + F.binary_cross_entropy_with_logits(d_12, label_12)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward()
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        if self.gan_type == 'GAN':
            d_fake_for_g = self.discriminator(fake)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g, label_real)

        elif self.gan_type == 'FI_GAN':
            d_01_for_g = F.sigmoid(self.discriminator(input_frames[:, 0], fake))
            d_12_for_g = F.sigmoid(self.discriminator(fake, input_frames[:, 1]))
            loss_g = d_01_for_g * torch.log(d_01_for_g + 1e-12) + d_12_for_g * torch.log(d_12_for_g + 1e-12)
            loss_g = loss_g.mean()

        elif self.gan_type.find('WGAN') >= 0:
            d_fake_for_g = self.discriminator(fake)
            loss_g = -d_fake_for_g.mean()
        
        elif self.gan_type in ['FI_Cond_GAN', 'STGAN']:
            d_fake_for_g = self.discriminator(input_frames[:, 0], fake, input_frames[:, 1])
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g, label_real)

        # Generator loss
        return loss_g
