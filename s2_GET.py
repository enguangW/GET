import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment,distill_crit
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root,get_dir
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups

from clip import clip
import torch.nn.functional as F

class TES(nn.Module):
    def __init__(self,
                 device = None,
                 num_words = 7,  # context len
                 word_dim = 512,  # word embedding dim
                 words_drop_ratio = 0.2, 
                 ):
        super(TES, self).__init__()
        self.num_words = num_words
        self.word_dim = word_dim
        self.device = device
        
        self.projector = nn.Linear(in_features=512, out_features=num_words*word_dim).to(device)  # transfer img embedding to text tokens
        self.tes_model,_ = clip.load_TES_CLIP("ViT-B/16", torch.device(device),n_ctx=self.num_words+2) #  num_words + sot + eot
        self.vlm_model,_ = clip.load_TES_CLIP("ViT-B/16", torch.device(device),n_ctx=77)
        for params in self.vlm_model.parameters():
            params.requires_grad = False  
              
        self.words_drop_ratio = words_drop_ratio
        
        for params in self.tes_model.parameters():
            params.requires_grad = False

    def text_encoder(self,text_descriptions):
        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        text_features = self.vlm_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def genPf(self,pseudo_words,drop_or_not):
        if pseudo_words.shape[0] == 0: 
            raise ValueError("pseudo_words.shape[0] == 0")
        valid_mask = self._drop_word(pseudo_words)
        pseudo_text, end_token_ids = self.tes_model.prepare_pseudo_text_tensor(
            pseudo_words,valid_mask=valid_mask,w_dropout=drop_or_not) 
        pseudo_feats = self.tes_model.encode_pseudo_text(pseudo_text, end_token_ids, text_pe=True)
        
        return pseudo_feats
        
    def _drop_word(self, pseudo_words):
        p = self.words_drop_ratio
        num_preds, num_words, _ = pseudo_words.shape
        mask = F.dropout(pseudo_words.new_ones(num_preds, num_words),
                         p=p,
                         training=self.training)
        start_end_mask = torch.ones_like(mask[:, :1])
        # check empty
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0       # TODO add random on this
        mask[mask > 0.0] = 1.0
        # add start and end token mask
        valid_mask = torch.cat([start_end_mask, mask, start_end_mask], dim=-1)

        return valid_mask
    
    
    def forward(self,x,drop_or_not=False):
        # [bs,512]
        img_feasts = self.vlm_model.encode_image(x).float()
        # [bs,7,512]
        pseudo_words = self.projector(img_feasts).view(-1, self.num_words, self.word_dim)
        # [bs,512]
        pseudo_text_features = self.genPf(pseudo_words, drop_or_not)
        pseudo_text_features = pseudo_text_features.float()

        
        return pseudo_text_features
    

def train_dual(backbone, projector, model_tes,ln_t,train_loader, test_loader, unlabelled_train_loader, args,device):
    # params_groups = get_params_groups(student)
    # optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = SGD(list(ln_t.parameters())+list(projector.parameters())+list(backbone.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0
    for epoch in range(args.epochs):
            
        loss_record = AverageMeter()
        backbone.train()
        projector.train()
        ln_t.train()
        model_tes.eval()
        
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True,device=device), mask_lab.cuda(non_blocking=True,device=device).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True,device=device)


            with torch.cuda.amp.autocast(fp16_scaler is not None):
                image_features = backbone.encode_image(images).float()
                text_features = model_tes(images)
                text_features_ln = ln_t(text_features)
        
                ## -----text_branch-----
                student_proj_text, student_out_text = projector(text_features_ln)
                teacher_out_text = student_out_text.detach()

                # clustering, sup
                sup_logits_text = torch.cat([f[mask_lab] for f in (student_out_text / 0.1).chunk(2)], dim=0)
                sup_labels_text = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss_text = nn.CrossEntropyLoss()(sup_logits_text, sup_labels_text)

                # clustering, unsup
                cluster_loss_text = cluster_criterion(student_out_text, teacher_out_text, epoch)
                avg_probs_text = (student_out_text / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss_text = - torch.sum(torch.log(avg_probs_text**(-avg_probs_text))) + math.log(float(len(avg_probs_text)))  # mean entropy reg for text
                cluster_loss_text += args.memax_weight * me_max_loss_text

                # represent learning, unsup
                contrastive_logits_text, contrastive_labels_text = info_nce_logits(features=student_proj_text,device=device)
                contrastive_loss_text = torch.nn.CrossEntropyLoss()(contrastive_logits_text, contrastive_labels_text)

                # representation learning, sup
                student_proj_text = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj_text.chunk(2)], dim=1)
                student_proj_text = torch.nn.functional.normalize(student_proj_text, dim=-1)
                sup_con_labels_text = class_labels[mask_lab]
                sup_con_loss_text = SupConLoss()(student_proj_text, labels=sup_con_labels_text,device=device)
              
                loss_t = 0
                loss_t += (1 - args.sup_weight) * cluster_loss_text + args.sup_weight * cls_loss_text
                loss_t += (1 - args.sup_weight) * contrastive_loss_text + args.sup_weight * sup_con_loss_text
                
                
                ## -----visual_branch-----
                student_proj, student_out = projector(image_features)
                teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs))) # mean entropy reg for visual
                cluster_loss += args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj,device=device)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels,device=device)
              
                loss_v = 0
                loss_v += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss_v += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
              
              
              
                ## -----CICO-----
                lab_idx = torch.unique(class_labels[mask_lab])
                class_labels_2 = torch.concat([class_labels,class_labels])
                mask_lab_2 = torch.concat([mask_lab,mask_lab])
                anchor_img = [image_features[mask_lab_2][torch.where(class_labels_2[mask_lab_2] == i)[0]].mean(0) \
                    for i in lab_idx]
                anchor_img = torch.vstack(anchor_img).detach()
                
                
                sim_img = F.softmax(F.normalize(image_features,dim=-1) @ F.normalize(anchor_img,dim=-1).t() / 0.4, dim=1)

                sim_img_teacher = sim_img.detach()
                
                anchor_text = [text_features_ln[mask_lab_2][torch.where(class_labels_2[mask_lab_2] == i)[0]].mean(0) \
                    for i in lab_idx]
                anchor_text = torch.vstack(anchor_text).detach()
                
                sim_text = F.softmax(F.normalize(text_features_ln,dim=-1) @ F.normalize(anchor_text,dim=-1).t() / 0.4, dim=1)

                sim_text_teacher = sim_text.detach()

                kl_img = torch.mean(torch.sum(-torch.log(sim_img) * sim_text_teacher, dim=-1))
                kl_text = torch.mean(torch.sum(-torch.log(sim_text) * sim_img_teacher, dim=-1))
                
                loss_cico = kl_text + kl_img
                
              
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                pstr += f'cls_loss_text: {cls_loss_text.item():.4f} '
                pstr += f'cluster_loss_text: {cluster_loss_text.item():.4f} '
                pstr += f'sup_con_loss_text: {sup_con_loss_text.item():.4f} '
                pstr += f'contrastive_loss_text: {contrastive_loss_text.item():.4f} '
                pstr += f'kl_img: {kl_img.item():.4f} '
                pstr += f'kl_text: {kl_text.item():.4f} '
                
                    
                loss = loss_v + loss_t + args.lamC * loss_cico
  

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
                

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc_img, old_acc_img, new_acc_img,all_acc_text, old_acc_text, new_acc_text,\
            all_acc_mix, old_acc_mix, new_acc_mix = test(backbone, projector, model_tes,ln_t, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args,device=device)


        args.logger.info('Train Accuracies(img): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_img, old_acc_img, new_acc_img))
        args.logger.info('Train Accuracies(text): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_text, old_acc_text, new_acc_text))
        args.logger.info('Train Accuracies(sum): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_mix, old_acc_mix, new_acc_mix))
        
        
        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'backbone': backbone.state_dict(),
            'model_tes_projector': model_tes.projector.state_dict(),
            'ln_t': ln_t.state_dict(),
            'proj': projector.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("full model saved to {}.".format(args.model_path))

        # if old_acc_test > best_test_acc_lab:
        #     
        #     args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
        #     args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        #     
        #     torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
        #     args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
        #     
        #     # inductive
        #     best_test_acc_lab = old_acc_test
        #     # transductive            
        #     best_train_acc_lab = old_acc
        #     best_train_acc_ubl = new_acc
        #     best_train_acc_all = all_acc
        # 
        # args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(backbone, projector, model_tes,ln_t, test_loader, epoch, save_name, args,device):
    backbone.eval()
    projector.eval()
    model_tes.eval()
    ln_t.eval()

    preds_img, preds_text, targets = [], [],[]
    preds_mix = []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True,device=device)
        with torch.no_grad():
            image_features = backbone.encode_image(images).float()
            text_features = model_tes(images)
            text_features_ln = ln_t(text_features)
            _, logits_text = projector(text_features_ln)
            _, logits_img = projector(image_features)
            logits_mix = (logits_img.softmax(-1) + logits_text.softmax(-1))/2
            
            preds_img.append(logits_img.argmax(1).cpu().numpy())
            
            preds_text.append(logits_text.argmax(1).cpu().numpy())
            preds_mix.append(logits_mix.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds_img = np.concatenate(preds_img)
    preds_text = np.concatenate(preds_text)
    preds_mix = np.concatenate(preds_mix)
    
    targets = np.concatenate(targets)
    all_acc_img, old_acc_img, new_acc_img = log_accs_from_preds(y_true=targets, y_pred=preds_img, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)
    all_acc_text, old_acc_text, new_acc_text = log_accs_from_preds(y_true=targets, y_pred=preds_text, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)
    all_acc_mix, old_acc_mix, new_acc_mix = log_accs_from_preds(y_true=targets, y_pred=preds_mix, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc_img, old_acc_img, new_acc_img,all_acc_text, old_acc_text, new_acc_text,all_acc_mix, old_acc_mix, new_acc_mix

def set_random_seed(seed: int) -> None:
    import random,os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, imagenet_100, cub, scars, aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='debug', type=str)
    parser.add_argument('--exp_id', default='debug', type=str)
    
    parser.add_argument('--num_words', default=7, type=int)
    parser.add_argument('--words_drop_ratio', default=0.2, type=float)
    parser.add_argument('--word_dim', default=512, type=int)   
    parser.add_argument('--lamC', default=1.0, type=float)
    parser.add_argument('--TES_model_dir', type=str, default=None)
    
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    
    
    
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    
    if args.seed is None:
        torch.backends.cudnn.benchmark = True
    else: set_random_seed(args.seed)
    
    device = torch.device(f"cuda:{args.cuda_dev}" if torch.cuda.is_available() else "cpu")

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    
    
    init_experiment(args, runner_name=['GET'], exp_id=args.exp_id)
    
    args.interpolation = 3
    args.crop_pct = 0.875

    args.image_size = 224
    args.feat_dim = 512
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    
    
    args.logger.info('\n--------------stage2--------------'*10)
    
    args.logger.info(f'argsparser for stage 2:{args}')
    
    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    # train_transform, test_transform = preprocess,preprocess
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets,train_examples_test = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # train_test_loader = DataLoader(train_examples_test, num_workers=args.num_workers,
    #                                     batch_size=256, shuffle=False, pin_memory=False)
    
    
    # load model

    model_tes = TES(device=device, num_words=args.num_words, 
                              word_dim=args.word_dim, words_drop_ratio=args.words_drop_ratio)
    
    if args.TES_model_dir is not None:
        args.logger.info(f'Loading weights from {args.TES_model_dir}')
        model_tes.projector.load_state_dict(torch.load(args.TES_model_dir, map_location="cpu")['model_tes_projector'])
    else:    
        model_tes.projector.load_state_dict(torch.load(args.tes_model_path, map_location="cpu")['model_tes_projector'])
    
    model_clip, preprocess = clip.load("ViT-B/16",device)

    for m in model_tes.parameters():
        m.requires_grad = False
        
    for m in model_clip.parameters():
        m.requires_grad = False

    for name, m in model_clip.named_parameters():
        # if 'visual.proj' in name:
        #     m.requires_grad =True
        if 'visual.transformer.resblocks' in name:
            block_num = int(name.split('.')[3])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    
    model_clip.visual.use_proj = False  # True for imagenet1k
    
    args.logger.info("stage2-model_tes:")        
    for name, m in model_tes.named_parameters():
        if m.requires_grad ==True:
            args.logger.info(f"{name} requires_grad")
    
    args.logger.info("stage2-model_clip:")
    for name, m in model_clip.named_parameters():
        if m.requires_grad ==True:
            args.logger.info(f"{name} requires_grad")
    
    args.logger.info('stage2 model build')
    
    
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=768, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers).to(device)
    
    
    # make pseudo text embedding match the dim of visual embedding (except 512 to 512 for imagenet1k)
    ln_t = nn.Linear(in_features=512,out_features=768).to(device)  
    
    train_dual(model_clip, projector, model_tes, ln_t, train_loader, None, test_loader_unlabelled, args,device)
    
    
    
        
    
    
  
