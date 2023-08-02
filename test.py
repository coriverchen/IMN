import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import test_datasets
import dwt_iwt
from options import arguments
opt=arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer')


net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=opt.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=opt.lr, betas=opt.betas, eps=1e-6, weight_decay=opt.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, opt.weight_step, gamma=opt.gamma)
load(opt.model_path)
net.eval()
dwt = dwt_iwt.DWT()
iwt = dwt_iwt.IWT()


with torch.no_grad():
    for i, data in enumerate(test_datasets.testloader):
        data = data.to(device)
        mask_face = data[data.shape[0] // 2:, :, :, :]
        protect_face = data[:data.shape[0] // 2, :, :, :]
        mask_face_input = dwt(mask_face)
        protect_face_input = dwt(protect_face)
        input_img = torch.cat((mask_face_input, protect_face_input), 1)

        # forward:

        output = net(input_img)
        output_masked_face = output.narrow(1, 0, 4 * opt.channels_in)
        output_lost_matrix_m = output.narrow(1, 4 * opt.channels_in, output.shape[1] - 4 * opt.channels_in)
        masked_face = iwt(output_masked_face)
        ramdom_martrix_input = gauss_noise(output_lost_matrix_m.shape)
        lost_matrix_m = iwt(output_lost_matrix_m)
        ramdom_martrix = iwt(ramdom_martrix_input)

        # backward

        output_rev = torch.cat((output_masked_face, ramdom_martrix_input), 1)
        backward_img = net(output_rev, rev=True)
        protect_face_rev = backward_img.narrow(1, 4 * opt.channels_in, backward_img.shape[1] - 4 * opt.channels_in)
        protect_facet_rev = iwt(protect_face_rev)
        mask_face_rev = backward_img.narrow(1, 0, 4 * opt.channels_in)
        mask_face_rev = iwt(mask_face_rev)
        resi_mask_and_masked = (masked_face - mask_face_rev) * 20  #magnify 20 times
        resi_protected_and_recovered = (protect_face - protect_facet_rev) * 20 # magnify 20 times

        torchvision.utils.save_image(mask_face, opt.test_mask_face_path + '%.5d.png' % i)
        torchvision.utils.save_image(protect_face, opt.test_protect_face_path + '%.5d.png' % i)
        torchvision.utils.save_image(masked_face, opt.test_masked_face_path + '%.5d.png' % i)
        torchvision.utils.save_image(protect_facet_rev, opt.test_protect_facet_rev_path + '%.5d.png' % i)
        torchvision.utils.save_image(mask_face_rev, opt.test_mask_face_rev_path + '%.5d.png' % i)
        torchvision.utils.save_image(resi_protected_and_recovered, opt.test_resi_protected_and_recovered_path + '%.5d.png' % i)
        torchvision.utils.save_image(resi_mask_and_masked, opt.test_resi_mask_and_masked_path + '%.5d.png' % i)
        torchvision.utils.save_image(ramdom_martrix, opt.test_ramdom_matrix_path + '%.5d.png' % i)
        torchvision.utils.save_image(lost_matrix_m, opt.test_lost_matrix_m_path + '%.5d.png' % i)




