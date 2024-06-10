import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import LoRA_Clip
from utils import *
import loralib as lora

from datasets import build_dataset
from datasets.utils import build_data_loader

def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of LoRA-CLIP in yaml format')
    args = parser.parse_args()

    return args


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    flag = 0
    
    print("**** Start ****")
    for round in range(cfg['Round']):
        for source in cfg['source_working_condition']:
            flag += 1
            print("Training on {} , at round {}".format(source, round))
            cache_dir = os.path.join('./finetune_model_chek', cfg['finetune_method'], cfg['dataset'], source)
            os.makedirs(cache_dir, exist_ok=True)
            cfg['cache_dir'] = cache_dir

            print("\nRunning configs.")
            # print(cfg, "\n")
            
            # LoRA_CLIP
            clip_model, preprocess = LoRA_Clip.load(cfg['backbone'],lora_r=cfg['lora_r'])
            clip_model.load_state_dict(torch.load('./finetune_model_chek\\fintune_meta' + '/best_machine_lora_model.pt'), strict=False)
            lora.mark_only_lora_as_trainable(clip_model, bias='all')
            
            # Calculate the number of trainable parameters of the lora_clip_model
            # print_trainable_parameters(clip_model)

            # Finetune dataset
            random.seed(1)
            torch.manual_seed(1)
            
            train_tranform = transforms.Compose([
                        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                    ])
            
            dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'], source)

            val_loader = build_data_loader(data_source=dataset.val, batch_size=128, is_train=False, tfm=preprocess, shuffle=False)
            test_loader = build_data_loader(data_source=dataset.test, batch_size=128, is_train=False, tfm=preprocess, shuffle=False)
            
            train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=16, tfm=train_tranform, is_train=True, shuffle=False)
            train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=16, tfm=train_tranform, is_train=True, shuffle=True)

            # Textual features
            # print("\nGetting textual features as CLIP's classifier.")
            clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

            # Construct the cache model by few-shot training set
            # print("\nConstructing cache model by few-shot visual features and labels.")
            cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache, lora_r=cfg['lora_r'])

            # Pre-load val features
            # print("\nLoading visual features and labels from val set.")
            val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
            
            # Pre-load test features
            # print("\nLoading visual features and labels from test set.")
            test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
            
            # Enable the cached keys to be learnable
            if flag == 1:
                adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
                adapter.weight = nn.Parameter(cache_keys.t())
            
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
            # optimizer = torch.optim.SGD(adapter.parameters(), lr=cfg['lr'], momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
            
            beta, alpha = cfg['init_beta'], cfg['init_alpha']
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
                print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

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

            print("\n-------- Evaluating F on the test set. --------")
        
            affinity = adapter(test_features)
            cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
            clip_logits = 100. * test_features @ clip_weights
            tip_logits = clip_logits + cache_logits * best_alpha
            acc = cls_acc(tip_logits, test_labels)
            print("**** Tip-Adapter-F_LoRA_CLIP 's test accuracy on {} at round {} : {:.2f} ****\n".format(source,round,acc))
            
    print("**** Test ****")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'], cfg['target_working_condition'])

    test_loader = build_data_loader(data_source=dataset.test, batch_size=128, is_train=False, tfm=preprocess, shuffle=False)

    # Textual features
    # print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    # print("\nConstructing cache model by few-shot visual features and labels.")
    _, cache_values = build_cache_model(cfg, clip_model, train_loader_cache, lora_r=cfg['lora_r'])
    
    # Pre-load test features
    # print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
   
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, adapter=adapter)
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    clip_logits = 100. * test_features @ clip_weights
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter-F_LoRA_CLIP 's domain generalization test accuracy on {}: {:.2f} ****\n".format(cfg['target_working_condition'],acc))
    
    

if __name__ == '__main__':
    main()        

