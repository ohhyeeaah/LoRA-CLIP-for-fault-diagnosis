import os
import random
import argparse
import yaml
from tqdm import tqdm
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import LoRA_Clip
import loralib as lora
import clip   #openai clip
from utils import *
# os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of LoRA-CLIP in yaml format')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, val_features_raw, test_features, test_labels, test_features_raw, clip_weights, clip_weights_raw, cache_keys_raw, cache_values_raw):
    
    # print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits_raw = 100. * val_features_raw @ clip_weights_raw
    acc_raw = cls_acc(clip_logits_raw, val_labels)
    # print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc_raw))
    
    # Zero-shot Finetuned_LoRA_Clip
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    # print("\n**** Zero-shot Finetuned_LoRA_Clip's val accuracy: {:.2f}. ****\n".format(acc))
    
    # Tip-Adapter-raw
    beta_raw, alpha_raw = cfg['init_beta'], cfg['init_alpha']
    
    affinity_raw = val_features_raw @ cache_keys_raw
    cache_logits_raw = ((-1) * (beta_raw - beta_raw * affinity_raw)).exp() @ cache_values_raw
    # cache_logits_raw = affinity_raw @ cache_values_raw
    
    tip_logits_raw = clip_logits_raw + cache_logits_raw * alpha_raw
    # tip_logits_raw = clip_logits_raw + cache_logits_raw
    acc = cls_acc(tip_logits_raw, val_labels)
    # print("**** Tip-Adapter_raw's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters-raw
    best_beta_raw, best_alpha_raw = search_hp(cfg, cache_keys_raw, cache_values_raw, val_features_raw, val_labels, clip_weights_raw)

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    # cache_logits = affinity @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    # tip_logits = clip_logits + cache_logits
    acc = cls_acc(tip_logits, val_labels)
    # print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits_raw = 100. * test_features_raw @ clip_weights_raw
    acc_raw = cls_acc(clip_logits_raw, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy on {}: {:.2f} ****\n".format(cfg['current_wc'],acc_raw))
    
    # Zero-shot Finetuned_LoRA_Clip
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot Finetuned_LoRA_Clip's test accuracy: {:.2f} ****\n".format(acc))
    
    # Tip-Adapter-raw
    affinity_raw = test_features_raw @ cache_keys_raw
    cache_logits_raw = ((-1) * (best_beta_raw - best_beta_raw * affinity_raw)) @ cache_values_raw
    
    tip_logits_raw = clip_logits_raw + cache_logits_raw * best_alpha_raw
    acc = cls_acc(tip_logits_raw, test_labels)
    print("**** Tip-Adapter_raw's test accuracy on {}: {:.2f} ****\n".format(cfg['current_wc'],acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f} ****\n".format(acc))
    
    # Search Hyperparameters again
    # best_beta_a, best_alpha_a = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F, choice):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    # cache_keys_copy = cache_keys.clone()
    # train_loader_F_copy = copy.deepcopy(train_loader_F)
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    # optimizer = torch.optim.SGD(adapter.parameters(), lr=cfg['lr'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    # beta = nn.Parameter(torch.tensor(float(cfg['init_beta'])))
    # alpha = nn.Parameter(torch.tensor(float(cfg['init_alpha'])))
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        # print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(train_loader_F):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            # for name, param in clip_model.named_parameters():
            #     print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            
            # # Initialize with equal weights
            # class_weights = torch.ones(tip_logits.shape[1]).cuda()  
            # penalty_list = [6, 8]
            # # Set a higher weight for the specific class
            # for i in penalty_list:
            #     class_weights[i] *= 1.1
            # loss = F.cross_entropy(tip_logits, target, weight=class_weights)

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        # print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(val_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * val_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, val_labels)

        # print("**** Tip-Adapter-F's val accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    # print(f"**** After fine-tuning, Tip-Adapter-F's best val accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)
    
    # best_beta, best_alpha = beta, alpha

    print("\n-------- Evaluating F on the test set. --------")
   
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    clip_logits = 100. * test_features @ clip_weights
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    # acc_class = cls_acc_each_class(tip_logits, test_labels)
    # for c, accuracy in enumerate(acc_class):
    #     print('Accuracy of class {} : {:.2f}%'.format(c, accuracy))
    if choice == 0:
        print("**** Tip-Adapter-F's test accuracy on {}: {:.2f} ****\n".format(cfg['current_wc'],acc))
    else:
        print("**** Tip-Adapter-F_LoRA_CLIP 's test accuracy on {}: {:.2f} ****\n".format(cfg['current_wc'],acc))


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    shots_number = 0

    for shots in cfg['shots_list']:
        cfg['shots'] = shots
        cfg['train_epoch'] = cfg['train_epoch_list'][shots_number]
        shots_number += 1
        for working_condition in cfg['working_condition']:
            cfg['current_wc'] = working_condition
            cache_dir = os.path.join('./caches', cfg['dataset'], cfg['current_wc'])
            os.makedirs(cache_dir, exist_ok=True)
            cfg['cache_dir'] = cache_dir

            print("\nRunning configs.")
            # print(cfg, "\n")
            
            # CLIP
            clip_raw_model, preprocess = clip.load(cfg['backbone'])
            clip_raw_model.eval()

            # LoRA_CLIP
            clip_model, _ = LoRA_Clip.load(cfg['backbone'],lora_r=cfg['lora_r'])
            
            # Then load the LoRA checkpoint
            clip_model.load_state_dict(torch.load('./finetune_model_chek\\fintune_meta' + '/best_machine_lora_model.pt'), strict=False)
            lora.mark_only_lora_as_trainable(clip_model, bias='all')
            clip_model.eval()

            # Prepare dataset
            random.seed(1)
            torch.manual_seed(1)
            # torch.use_deterministic_algorithms(True)
            
            print("Preparing dataset.")
            dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'], cfg['current_wc'])

            val_loader = build_data_loader(data_source=dataset.val, batch_size=128, is_train=False, tfm=preprocess, shuffle=False)
            test_loader = build_data_loader(data_source=dataset.test, batch_size=128, is_train=False, tfm=preprocess, shuffle=False)

            train_tranform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
            # train_tranform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])

            train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=16, tfm=train_tranform, is_train=True, shuffle=False)
            train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=16, tfm=train_tranform, is_train=True, shuffle=True)

            # Textual features
            # print("\nGetting textual features as CLIP's classifier.")
            clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
            clip_weights_raw = clip_classifier_raw(dataset.classnames, dataset.template, clip_raw_model)

            # Construct the cache model by few-shot training set
            # print("\nConstructing cache model by few-shot visual features and labels.")
            cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache, lora_r=cfg['lora_r'])
            cache_keys_raw, cache_values_raw  = build_cache_model(cfg, clip_raw_model, train_loader_cache, lora_r=0)

            # Pre-load val features
            # print("\nLoading visual features and labels from val set.")
            val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
            val_features_raw = pre_load_features_raw(cfg, "val", clip_raw_model, val_loader)

            # Pre-load test features
            # print("\nLoading visual features and labels from test set.")
            test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
            test_features_raw = pre_load_features_raw(cfg, "test", clip_raw_model, test_loader)
            
            i, j = 0, 1

            # ------------------------------------------ Tip-Adapter ------------------------------------------
            run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, val_features_raw, test_features, test_labels, test_features_raw, clip_weights, clip_weights_raw, cache_keys_raw, cache_values_raw)
            # ------------------------------------------ Tip-Adapter-F ------------------------------------------
            run_tip_adapter_F(cfg, cache_keys_raw, cache_values_raw, val_features_raw, val_labels, test_features_raw, test_labels, clip_weights_raw, clip_raw_model, train_loader_F, i)
            # ------------------------------------------ Tip-Adapter-F_LoRA_CLIP ------------------------------------------
            run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F, j)
           

if __name__ == '__main__':
    main()