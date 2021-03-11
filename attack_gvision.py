import argparse
import random
import torchvision.models as models
import torch
import os
import json
import DataLoader
import gvision_wrapper
from utils import *
from FCN import *
from Normalize import Normalize, Permute
from imagenet_model.Resnet import resnet152_denoise, resnet101_denoise

from matplotlib import pyplot as plt


def print_desc_scores(descs, scores):
    print("\n")
    print("-"*30)
    for desc, score in zip(descs, scores):
        print(f"{desc}: {score}")

    print("-"*30)
    print("\n")



# whether label is cointained in label_set
def is_label_in_labelset(label, label_set):
    for l in label_set:
        if l.lower() in label.lower():
            return True
    return False


# True if at least one of labels returned by Gvision is in label_set
def is_correctly_classified(cls_labels, label_set):
    for cls_l in cls_labels:
        if is_label_in_labelset(cls_l, label_set):
            return True
    return False

def compute_loss(descs, scores, label_set, threshold=0):
    dic = dict(zip(descs, scores))

    true_scores = []
    other_scores = []

    true_labels = []
    other_labels = []

    for d, s in zip(descs, scores):
        s -= threshold
        if is_label_in_labelset(d, label_set):
            true_labels.append(d)
            true_scores.append(s)
        else:
            other_scores.append(s)
            other_labels.append(d)

    print("-----------")
    print("true labels:", true_labels)
    print("true scores:", true_scores)
    print("-----------")
    print("other labels:", other_labels)
    print("other scores:", other_scores)
    print("-----------\n")

    if len(true_scores) == 0:
        true_scores = [0]

    if len(other_scores) == 0:
        # don't let the loss  jump wildly 
        other_scores = [min(true_scores)]

    # losses of the original model are in the range 2-10
    # so I multiply my loss by coeficient by some constant factor bigger than 1
    #
    # We want to minize the score of true labels and maximize the score of all the others

    
    if config["loss"] == "sum":
        return (sum(true_scores) - sum(other_scores)) / len(scores) * 10
    elif config["loss"] == "max":
        # return (max(true_scores) - max(other_scores)) * 70
        return (max(true_scores) - dic["Plant"]) * 70


c = 0
def save_img(img):
    global c
    c += 1
    img = img.transpose(1,2,0)
    img = (img * 255).astype(np.uint8)
    fn = "output/cat" + str(c) + ".png"
    plt.imsave(fn, img)
    print("img saved at ", fn)

    # plt.show()
    input("Press enter to continue")


def l2_norm(tensor):
    return np.sum((tensor)**2)**0.5


def EmbedBAGVision(gvision, encoder, decoder, image, true_labels, latent=None):
    device = image.device

    if latent is None:
        latent = encoder(image.unsqueeze(0)).squeeze().view(-1)
    
    # latent.shape == [1568]
    momentum = torch.zeros_like(latent)
    dimension = len(latent)
    noise = torch.empty((dimension, config['sample_size']), device=device)
    origin_image = image.clone()
    last_loss = []
    lr = config['lr']


    last_loss = []
    last_img = [image.detach().numpy()]
    for iter in range(config['num_iters']+1):
        print("+"*30)
        print("iter:", iter)
        print("+"*30)

        perturbation = torch.clamp(decoder(latent.unsqueeze(0)).squeeze(0)*config['epsilon'], -config['epsilon'], config['epsilon'])

        pertubed_img = torch.clamp(image+perturbation, 0, 1).detach().numpy()
        last_img.append(pertubed_img)
        print("L2 difference between 2 last tries:", l2_norm(last_img[-1] - last_img[-2]))

        descriptions, scores = gvision(pertubed_img)

        print_desc_scores(descriptions, scores)

        last_loss.append(compute_loss(descriptions, scores, true_labels))
        print(f"loss: {last_loss[-1]}, l2_deviation {torch.norm(perturbation)}")
        print(f"lr: {lr}")
        print(f"sigma: {config['sigma']}")

        save_img(pertubed_img)


        # success = descriptions[0] != label
        if not is_correctly_classified(descriptions, true_labels):
            print("Success!")
            return True, pertubed_img


        nn.init.normal_(noise)

        # make the noise symmetrical
        noise[:, config['sample_size']//2:] = -noise[:, :config['sample_size']//2]

        latents = latent.repeat(config['sample_size'], 1) + noise.transpose(0, 1)*config['sigma']
        perturbations = torch.clamp(decoder(latents)*config['epsilon'], -config['epsilon'], config['epsilon'])
        samples = torch.clamp(image.expand_as(perturbations) + perturbations, 0, 1)

        losses = np.zeros(config["sample_size"], dtype=np.float32)
        for i in range(config["sample_size"]):
            sample_img = samples[i].detach().numpy()
            descriptions, scores = gvision(sample_img)
            # print_desc_scores(descriptions, scores)
            losses[i] = compute_loss(descriptions, scores, true_labels)
        
        losses = torch.tensor(losses)
        grad = torch.mean(losses.expand_as(noise) * noise, dim=1)
        print(torch.norm(grad))



        momentum = config['momentum'] * momentum + (1-config['momentum'])*grad
        latent = (latent - lr * momentum)

        # last_loss = last_loss[-config['plateau_length']:]
        # if (last_loss[-1] > last_loss[0]+config['plateau_overhead'] or last_loss[-1] > last_loss[0] and last_loss[-1]<0.6) and len(last_loss) == config['plateau_length']:
            # if lr > config['lr_min']:
                # lr = max(lr / config['lr_decay'], config['lr_min'])
                # last_loss = []

        lr = max(lr / config['lr_decay'], config['lr_min'])
        config['sigma'] = max(config['sigma'] / config['sigma_decay'],  config['sigma_min'])

    return False, origin_image


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/attack_untargeted_gvision.json', help='config file')
# parser.add_argument('--config', default='config/test.json', help='config file')
parser.add_argument('--device', default='cpu', help='Device for Attack')
parser.add_argument('--save_prefix', default=None, help='override save_prefix in config file')
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)
    

if args.save_prefix is not None:
    config['save_prefix'] = args.save_prefix



device = torch.device(args.device if torch.cuda.is_available() else "cpu")
weight = torch.load(os.path.join("G_weight", config['generator_name']+".pytorch"), map_location=device)

encoder_weight = {}
decoder_weight = {}
for key, val in weight.items():
    if key.startswith('0.'):
        encoder_weight[key[2:]] = val
    elif key.startswith('1.'):
        decoder_weight[key[2:]] = val

test_loader, nlabels, labels, mean, std = DataLoader.gvision(config)

if 'OSP' in config:
    if config['source_model_name'] == 'Adv_Denoise_Resnet152':
        s_model = resnet152_denoise()
        loaded_state_dict = torch.load(os.path.join('weight', config['source_model_name']+".pytorch"))
        s_model.load_state_dict(loaded_state_dict)
    if 'defense' in config and config['defense']:
        source_model = nn.Sequential(
            Normalize(mean, std),
            Permute([2,1,0]),
            s_model
        )
    else:
        source_model = nn.Sequential(
            Normalize(mean, std),
            s_model
        )

encoder = Imagenet_Encoder()
decoder = Imagenet_Decoder()
encoder.load_state_dict(encoder_weight)
decoder.load_state_dict(decoder_weight)

gvision = gvision_wrapper.GvisionWrapper()

encoder.to(device)
encoder.eval()
decoder.to(device)
decoder.eval()

if 'OSP' in config:
    source_model.to(device)
    source_model.eval()

count_success = 0
count_total = 0
if not os.path.exists(config['save_path']):
    os.mkdir(config['save_path'])



# All labels that we want to minimize
shark_label_set = ["Shark", "Fin", "Water", "Jaw", "Fish", "Carcharhiniformes", "Lamnidae", "Lamniformes"]

cat_label_set = ["Cat", "Felidae", "Whiskers"]

for i, (images, labels) in enumerate(test_loader):
    # bs=1 
    image = images[0]

    # label = label_to_str[int(labels[0].numpy())]

    # images = images.to(device)
    # glabels, scores = gvision(image.numpy())

    # If gvision top-1 label is correct, start the attack
    # correct = glabels[0] == label
    # if correct:
    
    # Always run the attack
    if True:
        success, adv = EmbedBAGVision(gvision, encoder, decoder, image, cat_label_set)


# success_rate = float(count_success) / float(count_total)
# if state['target']:
    # np.save(os.path.join(state['save_path'], '{}_class_{}.npy'.format(state['save_prefix'], state['target_class'])), np.array(F.counts))
# else:
    # np.save(os.path.join(state['save_path'], '{}.npy'.format(state['save_prefix'])), np.array(F.counts))
# print("success rate {}".format(success_rate))
# print("average eval count {}".format(F.get_average()))
