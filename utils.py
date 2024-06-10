from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import LoRA_Clip
import clip


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

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def cls_acc_each_class(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    class_correct = [0] * (target.max().item() + 1)
    class_total = [0] * (target.max().item() + 1)

    for c in range(target.max().item() + 1):
        class_correct[c] = correct[:, target == c].sum().item()
        class_total[c] = (target == c).sum().item()

    class_accuracy = [100 * class_correct[c] / class_total[c] if class_total[c] != 0 else 0 for c in range(target.max().item() + 1)]
    return class_accuracy

def clip_classifier(classnames, template, clip_model):
    
    with torch.no_grad():
        clip_weights = []
        
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            
            # prompt ensemble for ImageNet
            text_lora_clip = LoRA_Clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(text_lora_clip)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
    return clip_weights

def clip_classifier_raw(classnames, template, clip_raw_model):
    
    with torch.no_grad():
        clip_weights_raw = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            
            texts_raw = clip.tokenize(texts).cuda()
            class_embeddings_raw = clip_raw_model.encode_text(texts_raw)
            class_embeddings_raw /= class_embeddings_raw.norm(dim=-1, keepdim=True)
            class_embeddings_raw = class_embeddings_raw.mean(dim=0)
            class_embeddings_raw /= class_embeddings_raw.norm()
            clip_weights_raw.append(class_embeddings_raw)

        clip_weights_raw = torch.stack(clip_weights_raw, dim=1).cuda()
        
    return clip_weights_raw

def build_cache_model(cfg, clip_model, train_loader_cache, lora_r):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                # print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(train_loader_cache):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        
        if lora_r == 0:
            cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
            torch.save(cache_keys, cfg['cache_dir'] + '/keys_raw_' + str(cfg['shots']) + "shots.pt")
            torch.save(cache_values, cfg['cache_dir'] + '/values_raw_' + str(cfg['shots']) + "shots.pt")
        else:
            cache_values = F.one_hot(torch.cat(cache_values, dim=0)).float()
            torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
            torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
            
    else:
        if lora_r == 0:
            cache_keys = torch.load(cfg['cache_dir'] + '/keys_raw_' + str(cfg['shots']) + "shots.pt")
            cache_values = torch.load(cfg['cache_dir'] + '/values_raw_' + str(cfg['shots']) + "shots.pt")
        else:
            cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
            cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        
    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels

def pre_load_features_raw(cfg, split, clip_model_raw, loader):

    if cfg['load_pre_feat_raw'] == False:
        features = []

        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model_raw.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)

        features = torch.cat(features)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f_raw.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f_raw.pt")
    
    return features


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    # print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        # print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
        
    else:
        best_beta, best_alpha = cfg['init_beta'], cfg['init_alpha']

    return best_beta, best_alpha
