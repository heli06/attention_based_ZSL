import os 
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
from .kNN import kNNClassify
from .util import *
from sklearn.metrics import accuracy_score

def train2(image_model, att_model, relation_net, train_loader, test_seen_loader, test_unseen_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir
    save_model_dir = os.path.join(exp_dir,'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)
       
    epoch = 0
    
    if epoch != 0:        
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        att_model.load_state_dict(torch.load("%s/models/att_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    
    image_model = image_model.to(device)
    att_model = att_model.to(device)
    relation_net = relation_net.to(device)
    # Set up the optimizer
    image_trainables = [p for p in image_model.parameters() if p.requires_grad] # if p.requires_grad
    att_trainables = [p for p in att_model.parameters() if p.requires_grad]
    rel_trainables = [p for p in relation_net.parameters() if p.requires_grad]

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(att_trainables, args.lr_A,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizer_img = torch.optim.SGD(image_trainables, args.lr_I,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizer_rel = torch.optim.SGD(rel_trainables, args.lr_R,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(att_trainables, args.lr_A,
                                    weight_decay=args.weight_decay,
                                    betas=(0.95, 0.999))
        optimizer_img = torch.optim.Adam(image_trainables, args.lr_I,
                                    weight_decay=args.weight_decay,
                                    betas=(0.95, 0.999))
        optimizer_rel = torch.optim.Adam(rel_trainables, args.lr_R,
                                         weight_decay=args.weight_decay,
                                         betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)
    
    """
    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)
    """
    
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    image_model.train()
    att_model.train()
    relation_net.train()
    criterion_hinge = nn.TripletMarginLoss(margin=1.0,p=2)
    criterion_e = nn.MSELoss()
    criterion_s = nn.CosineSimilarity()
    criterion_c = nn.CrossEntropyLoss()    
    criterion_k = nn.KLDivLoss()
    criterion_m = nn.MSELoss()
    
    # 载入ZSL和GZSL评估所需要的id和attr文件
    print('载入id和attr文件')
    test_attr_file = os.path.join(args.data_path, args.test_class_attr)
    test_att = np.loadtxt(test_attr_file)
   
    all_attr_file = os.path.join(args.data_path, args.all_class_attr)
    all_att = np.loadtxt(all_attr_file)

    train_id_file = os.path.join(args.data_path, args.train_class_id)
    train_id = np.loadtxt(train_id_file, dtype=int)
    
    test_id_file = os.path.join(args.data_path, args.test_class_id)
    test_id = np.loadtxt(test_id_file, dtype=int)
    
    all_id_file = os.path.join(args.data_path, args.all_class_id)
    all_id = np.loadtxt(all_id_file, dtype=int)
    
    pre_acc = -1
    while epoch<=500:
        epoch += 1
        adjust_learning_rate(args.lr_A, args.lr_decay, optimizer, epoch)
        adjust_learning_rate(args.lr_I, args.lr_decay, optimizer_img, epoch)
        adjust_learning_rate(args.lr_R, args.lr_decay, optimizer_rel, epoch)
        end_time = time.time()
        image_model.train()
        att_model.train()
        relation_net.train()

        for i, (image_input, att_input, neg_att_input, cls_id, key, label) in enumerate(train_loader):
            att_input = att_input.float().to(device)   
            B = att_input.size(0)
            neg_att_input = neg_att_input.float().to(device)
            label = label.long().to(device)
            image_input = image_input.float().to(device)
            image_input = image_input.squeeze(1)            
    
            optimizer.zero_grad()
            optimizer_img.zero_grad()
            optimizer_rel.zero_grad()

            final_image_output = None
            final_att_output = None

            # 输出图像特征向量和属性向量
            image_output = image_model(image_input)
            att_output = att_model(att_input)
            neg_att_output = att_model(neg_att_input)

            pos_output = relation_net(torch.cat((image_output, att_output), dim=-1))
            neg_output = relation_net(torch.cat((image_output, neg_att_output), dim=-1))
            output = torch.cat((pos_output, neg_output), dim=0)
            y1 = np.ones((args.batch_size,), dtype=np.int)
            y1 = torch.from_numpy(y1).to(device)
            y0 = np.zeros((args.batch_size,), dtype=np.int)
            y0 = torch.from_numpy(y0).to(device)
            y = torch.cat((y1, y0), dim=-1)

                # print('--------------------------------------------------------------------')
                # print('Best_zsl:', pre_acc)
                # print('According_gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (pre_seen, pre_useen, pre_H))
                

            #print(image_input.size())
            #print(att_input.size())
            #print(image_output.size())
            #print(att_output.size())
            #torch.Size([20, 3, 244, 244])
            #torch.Size([20, 312])
            #torch.Size([20, 2048])
            #torch.Size([20, 2048])
            
            # pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            # nframes.div_(pooling_ratio)

            # loss = sampled_margin_rank_loss(image_output, audio_output,
            #     nframes, margin=args.margin, simtype=args.simtype)
            
            # loss = torch.pow(image_output - audio_output, 2).sum()
            # loss = criterion(image_output , audio_output)
            
            
            # image_output = normalizeFeature(image_output)
            # audio_output = normalizeFeature(audio_output)
            # neg_samples = normalizeFeature(neg_samples)
            # loss_t = criterion(image_output,audio_output,neg_samples)
            loss = 0
            if args.Loss_CE:
                loss = criterion_c(output, y)
            if args.Loss_cont:
                loss = criterion_e(final_image_output, final_att_output) * args.gamma_cont
            if args.Loss_batch:
                lossb1,lossb2 = batch_loss(final_image_output, final_att_output,cls_id,args)
                loss_batch = lossb1 + lossb2
                loss += loss_batch*args.gamma_batch        
            
            if args.Loss_dist:
                loss_dist1, loss_dist2  = distribute_loss(final_image_output, final_att_output)
                loss_dist = loss_dist1 + loss_dist2
                loss += loss_dist*args.gamma_dist            
            
            if args.Loss_hinge:
                neg_pair_audio,neg_pair_image = hardest_negative_mining_pair(image_output,audio_output,cls_id)
                # neg_single_audio = hardest_negative_mining_single(audio_output,cls_id)
                # neg_single_image = hardest_negative_mining_single(image_output,cls_id)
                hinge_IAA = criterion_hinge(image_output,att_output,neg_pair_att)
                hinge_AII = criterion_hinge(att_output,image_output,neg_pair_image)
                hinge_loss = hinge_AII + hinge_IAA
                loss += hinge_loss*args.gamma_hinge


            loss.backward()
            optimizer.step()
            optimizer_img.step()
            optimizer_rel.step()
            
            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            
            if i % 5 == 0:
                print('epoch: %d | iteration = %d | loss = %f'
                      %(epoch, i, loss))
                
        if epoch % 2 == 0:
            # train: (8821,)
            # trainval: (7057,)
            # test_seen: (1764,)
            # test_unseen: (2967,)
            image_model.eval()
            att_model.eval()
            relation_net.eval()
            acc_zsl = compute_accuracy(image_model, att_model, relation_net,
                                       test_unseen_loader, test_att, test_id, args)
            acc_seen_gzsl = compute_accuracy(image_model, att_model, relation_net,
                                             test_seen_loader, all_att, all_id, args)
            acc_unseen_gzsl = compute_accuracy(image_model, att_model, relation_net,
                                               test_unseen_loader, all_att, all_id, args)
            H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)
            if acc_zsl > pre_acc:
                pre_acc = acc_zsl
                pre_seen = acc_seen_gzsl
                pre_unseen = acc_unseen_gzsl
                pre_epoch = epoch
                pre_H = H

            print('epoch: %d | itr: %d | zsl: ACC=%.4f | gzsl: seen=%.4f, unseen=%.4f, h=%.4f'
                  % (epoch, i, acc_zsl, acc_seen_gzsl, acc_unseen_gzsl, H))
            print('max acc: epoch: %d | itr: %d | zsl: ACC=%.4f | gzsl: seen=%.4f, unseen=%.4f, h=%.4f'
                  % (pre_epoch, i, pre_acc, pre_seen, pre_unseen, pre_H))
                

def compute_accuracy(image_model, att_model, relation_net, test_loader, test_att, test_id, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpred = []
    test_label = []
    
    test_att = torch.from_numpy(test_att)
    test_att = test_att.float().to(device)
    test_att_output = att_model(test_att)
    test_size = test_att_output.size()[0]

    test_att_output = test_att_output / test_att_output.norm(dim=1,keepdim=True)
    
    for i, (image_input, att_input, cls_id, key, label) in enumerate(test_loader):
        att_input = att_input.float().to(device)
        label = list(label.numpy())
        
        image_input = image_input.float().to(device)
        image_input = image_input.squeeze(1)

        image_output = image_model(image_input)
        att_output = att_model(att_input)

        image_output = image_output / image_output.norm(dim=1, keepdim=True)
        att_output = att_output / att_output.norm(dim=1, keepdim=True)
        
        for j in range(len(label)):
            output_repeat = image_output[j, :].repeat(test_size, 1)
            output = F.softmax(relation_net(torch.cat((output_repeat, test_att_output), dim=-1)))
            index = int(torch.max(output[:, 1], -1)[1])
            outputLabel = test_id[index]
            outpred.append(outputLabel)
            test_label.append(label[j])
    
    outpred = np.array(outpred)
    test_label = np.array(test_label)
    unique_labels = np.unique(test_label)
    
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(test_label == l)[0]
        acc += accuracy_score(test_label[idx], outpred[idx])
        
    acc = acc / unique_labels.shape[0]
    # acc = np.equal(outpred, test_label).mean()
    return acc
