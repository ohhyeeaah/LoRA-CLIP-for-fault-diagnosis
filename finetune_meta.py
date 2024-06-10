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


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of LoRA-CLIP in yaml format')
    args = parser.parse_args()

    return args

# def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
#     probs = F.softmax(logits, dim=-1)
#     log_probs = F.log_softmax(logits, dim=-1)

#     targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()

#     pt = torch.sum(probs * targets_one_hot, dim=-1)
#     pt = pt.unsqueeze(1) 
    
#     focal_weights = (alpha * targets_one_hot + (1 - alpha) * (1 - targets_one_hot)) * (1 - pt).pow(gamma)

#     focal_loss = -torch.sum(focal_weights * log_probs, dim=-1)
#     return focal_loss.mean()

def load_cache_meta(clip_model, loader):
    
    cache_keys = []
    cache_values = []
    train_features = []
    with torch.no_grad():
        for _, (images, target) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()
            image_features = clip_model.encode_image(images)
            # for name, param in clip_model.named_parameters():
            #     print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")
            train_features.append(image_features)
            cache_values.append(target)
    
    cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
        
    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).float()
    
    return cache_keys, cache_values

def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./finetune_model_chek', cfg['finetune_method'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['finetune_cache_dir'] = cache_dir

    print("\nRunning configs.")
    # print(cfg, "\n")
    
    # LoRA_CLIP
    clip_model, preprocess = LoRA_Clip.load(cfg['backbone'],lora_r=cfg['lora_r'])
    lora.mark_only_lora_as_trainable(clip_model, bias='all')
    
    # Calculate the number of trainable parameters of the lora_clip_model
    print_trainable_parameters(clip_model)

    # Finetune dataset
    random.seed(1)
    torch.manual_seed(1)
    
    best_acc = 0.0
    correct_samples, all_samples = 0, 0
    loss_list = []
    alpha = 0.1
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['meta_epoch'])
    
    # optimizer = torch.optim.SGD(clip_model.parameters(),
    #             lr=0.1 * 16/256,
    #             momentum=0.9,
    #             weight_decay=1e-4)

    # # learning 
    # milestones = [30, 60, 90]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_beta = nn.Parameter(torch.tensor(1.0))
    best_alpha = nn.Parameter(torch.tensor(3.0))
    
    train_tranform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        
    print("**** Start Meta_training ****")
    for epoch in range(cfg['meta_epoch']):
        random.shuffle(cfg['dataset_list'])
        for sub_dataset in cfg['dataset_list']:
            working_condition_list = cfg[sub_dataset+'_working_condition']
            for random_wc in working_condition_list:
                # random_wc = random.choice(working_condition_list)
                # print("Training on {} with condition {}, at epoch {}".format(sub_dataset, random_wc, epoch))
                dataset = build_dataset(sub_dataset, cfg['root_path'], cfg['shots'], random_wc)
                test_meta = dataset.generate_fewshot_dataset(dataset.test, num_shots=cfg['query_shots'])

                meta_train_loader = build_data_loader(dataset.train_x, batch_size=16, tfm=train_tranform, is_train=True, shuffle=False)
                meta_test_loader = build_data_loader(test_meta, batch_size=16, tfm=preprocess, is_train=False, shuffle=False)
                
                features, labels = [], []

                # Support set generate
                clip_model.train()
                cache_keys, cache_values = load_cache_meta(clip_model, meta_train_loader)
                clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
                
                # Text encoder frozen with removing "cache_keys.requires_grad_()"
                clip_weights.requires_grad_(), cache_keys.requires_grad_()
                
                # Query set test
                clip_model.eval()
                with torch.no_grad():
                    for _, (images, target) in enumerate(meta_test_loader):
                        images, target = images.cuda(), target.cuda()
                        image_features = clip_model.encode_image(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        features.append(image_features)
                        labels.append(target)

                features, labels = torch.cat(features), torch.cat(labels)
                clip_model.train()
                
                affinity = features @ cache_keys
                cache_logits = ((-1) * (best_beta - best_alpha * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * best_alpha
                
                # clip_logits = 100. * features @ clip_weights
                # cosine_similarity = features @ cache_keys @ cache_values
                # tip_logits = clip_logits + cosine_similarity

                loss = F.cross_entropy(tip_logits, labels)
                # ent_loss = torch.mean(torch.sum(-F.softmax(tip_logits, dim=-1) * F.log_softmax(tip_logits, dim=-1), dim=-1))
                # loss += alpha * ent_loss
                loss_list.append(loss)
                
                # Focal Loss
                # loss = focal_loss(tip_logits, labels)
                # ent_loss = torch.mean(torch.sum(-F.softmax(tip_logits, dim=-1) * F.log_softmax(tip_logits, dim=-1), dim=-1))
                # loss += alpha * ent_loss
                # loss_list.append(loss)

                acc = cls_acc(tip_logits, labels)
                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
            

        optimizer.zero_grad()
        total_loss = torch.sum(torch.stack(loss_list)) 
        total_loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        acc = correct_samples / all_samples
        
        if acc > best_acc:
            best_acc = acc
            torch.save(clip_model.state_dict(), cfg['finetune_cache_dir'] + "/best_machine_lora_model.pt")
        
        current_lr = scheduler.get_last_lr()[0]
        print('Meta training stage ---- LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, acc, correct_samples, all_samples, total_loss))
        
        loss_list = []
        correct_samples, all_samples = 0, 0
    
    print("**** LoRA-Clip's best meta training accuracy : {:.4f}. ****\n".format(best_acc))
    
    print("**** Start Meta_testing ****")
    # Load the LoRA-Clip checkpoint
    clip_model.load_state_dict(torch.load(cfg['finetune_cache_dir'] + "/best_machine_lora_model.pt"), strict=False)
    clip_model.eval()

    test_working_condition_list = cfg[cfg['meta_test_dataset']+'_working_condition']
    test_random_wc = random.choice(test_working_condition_list)
    dataset = build_dataset(cfg['meta_test_dataset'], cfg['root_path'], cfg['test_shots'], test_random_wc)
    test_meta = dataset.generate_fewshot_dataset(dataset.test, num_shots=cfg['test_shots'])

    meta_train_loader = build_data_loader(dataset.train_x, batch_size=16, tfm=preprocess, is_train=True, shuffle=False)
    meta_test_loader = build_data_loader(test_meta, batch_size=16, tfm=preprocess, is_train=False, shuffle=False)
    
    cache_keys, cache_values = load_cache_meta(clip_model, meta_train_loader)
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
    features, labels = [], []
    
    with torch.no_grad():
        for _, (images, target) in enumerate(tqdm(meta_test_loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)
    
    affinity = features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_alpha * affinity)).exp() @ cache_values
    clip_logits = 100. * features @ clip_weights
    tip_logits = clip_logits + cache_logits * best_alpha
            
    # clip_logits = 100. * features @ clip_weights
    # cosine_similarity = features @ cache_keys @ cache_values
    # tip_logits = clip_logits + cosine_similarity
    
    acc = cls_acc(tip_logits, labels)
    print("**** LoRA-Clip's meta testing accuracy on {} with condition {}: {:.4f}. ****\n".format(cfg['meta_test_dataset'], test_random_wc, acc))
    

if __name__ == '__main__':
    main()        

