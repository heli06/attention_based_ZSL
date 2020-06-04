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

def train4(image_model, AttModels, relation_net, Attention, train_loader, test_seen_loader, test_unseen_loader, args):
    att_proj = AttModels.AttProj(args)
    att_model = AttModels.AttEncoder(args)
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

    if torch.cuda.device_count() > 1:
        image_model = nn.DataParallel(image_model)
        att_model = nn.DataParallel(att_model)
        relation_net = nn.DataParallel(relation_net)
    
    image_model = image_model.to(device)
    att_model = att_model.to(device)
    relation_net = relation_net.to(device)
    att_proj = att_proj.to(device)
    # Set up the optimizer
    image_trainables = [p for p in image_model.parameters() if p.requires_grad] # if p.requires_grad
    att_trainables = [p for p in att_model.parameters() if p.requires_grad]
    rel_trainables = [p for p in relation_net.parameters() if p.requires_grad]
    proj_trainables = [p for p in att_proj.parameters() if p.requires_grad]
    trainables = image_trainables + att_trainables + rel_trainables + proj_trainables
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(trainables, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
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
    att_proj.train()
    criterion_hinge = nn.TripletMarginLoss(margin=1.0,p=2)
    criterion_e = nn.MSELoss()
    criterion_s = nn.CosineSimilarity()
    criterion_c = nn.CrossEntropyLoss()    
    criterion_k = nn.KLDivLoss()
    criterion_m = nn.MSELoss()
    criterion_b = nn.BCELoss()
    
    # 载入ZSL和GZSL评估所需要的id和attr文件
    print('载入id和attr文件')
    test_attr_file = os.path.join(args.data_path, args.test_class_attr)
    test_att = np.loadtxt(test_attr_file)
   
    all_attr_file = os.path.join(args.data_path, args.all_class_attr)
    all_att = np.loadtxt(all_attr_file)

    train_attr_file = os.path.join(args.data_path, args.train_class_attr)
    train_att = np.loadtxt(train_attr_file)

    train_id_file = os.path.join(args.data_path, args.train_class_id)
    train_id = np.loadtxt(train_id_file, dtype=int)
    
    test_id_file = os.path.join(args.data_path, args.test_class_id)
    test_id = np.loadtxt(test_id_file, dtype=int)
    
    all_id_file = os.path.join(args.data_path, args.all_class_id)
    all_id = np.loadtxt(all_id_file, dtype=int)
    
    pre_acc = -1
    train_att = torch.from_numpy(train_att)
    train_att = train_att.float().to(device)
    att_num = train_att.size()[0]
    nor_train = normalizeFeature(train_att) # 200, 312

    while epoch<=500:
        epoch += 1
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        image_model.train()
        att_model.train()
        relation_net.train()
        att_proj.train()

        for i, (image_input, att_input, neg_att_input, cls_id, key, label) in enumerate(train_loader):
            att_input = att_input.float().to(device)   
            B = att_input.size(0)
            neg_att_input = neg_att_input.float().to(device)
            label = label.long().to(device)
            image_input = image_input.float().to(device)
            image_input = image_input.squeeze(1)            
    
            optimizer.zero_grad()
            final_image_output = None
            final_att_output = None

            # 输出属性向量
            nor_att = normalizeFeature(att_input) # batch_size, 312
            train_att_out = att_model(train_att) #  150, 1024
            att_output = att_proj(att_input)

            sim_mat = nor_att.mm(nor_train.t()) # batch_size, 150
            # conti_label = F.softmax(sim_mat)
            new_label = sim_mat.argmax(dim=-1).long().to(device) # batch_size 0~199

            # 输出图像特征向量
            image_output, attn_map = image_model(image_input, att_output)  # batch_size, 1024
            output_repeat = image_output.repeat_interleave(att_num, 0) # batch_size * 150, 1024

            train_att_out_repeat = train_att_out.repeat(B, 1) # batch_size * 150, 1024
            output = relation_net(torch.cat((output_repeat, train_att_out_repeat), dim=-1)) # batch_size * 150
            pred = output.view(B,-1)*args.smooth_gamma_r # batch_size, 150
            """
            for j in range(len(label)):
                output_repeat = image_output[j, :].repeat(att_num, 1)
                output = relation_net(torch.cat((output_repeat, all_att_out), dim=-1))
                output = output.view(1,att_num)
                outputs.append(output)
            pred = torch.cat(outputs) 
            """

            loss = criterion_c(pred, new_label)
            # pred = pred*50.0
            # p0 = F.softmax(pred, dim=-1)
            # loss = torch.sum(p0 * (F.log_softmax(pred, dim=-1) - F.log_softmax(conti_label, dim=-1)), 1).sum()

            """
            if args.Loss_BCE:
                loss = criterion_b(output, y)
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
            """

            loss.backward()
            optimizer.step()
           
            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            
            if i % 5 == 0:
                print('epoch: %d | iteration = %d | loss = %f'
                      %(epoch, i, loss))
                
        if epoch % 5 == 0:
            # train: (8821,)
            # trainval: (7057,)
            # test_seen: (1764,)
            # test_unseen: (2967,)
            image_model.eval()
            att_model.eval()
            relation_net.eval()
            att_proj.eval()
            acc_zsl = compute_accuracy(image_model, att_model, relation_net, att_proj,
                                       test_unseen_loader, test_att, test_id, args)            
                       
            info = ' Epoch: [{0}] | Loss {loss_:.4f} | ACC {acc:.4f} \n'.format(epoch,loss_=loss,acc=acc_zsl)
            print(info)
            
            save_path = os.path.join('outputs',args.save_file)
            with open(save_path, "a") as file:
                file.write(info)
            
            """
            if acc_zsl > pre_acc:      
                pre_acc = acc_zsl 
                if epoch>20:
                    acc_seen_gzsl = compute_accuracy(image_model, att_model, relation_net,
                                             test_seen_loader, all_att, all_id, args)
                    acc_unseen_gzsl = compute_accuracy(image_model, att_model, relation_net,
                                                    test_unseen_loader, all_att, all_id, args)
                    H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)               
                    
                    pre_seen = acc_seen_gzsl
                    pre_unseen = acc_unseen_gzsl
                    pre_epoch = epoch
                    pre_H = H                
                    print('max acc: epoch: %d | itr: %d | zsl: ACC=%.4f | gzsl: seen=%.4f, unseen=%.4f, h=%.4f'
                    % (pre_epoch, i, pre_acc, pre_seen, pre_unseen, pre_H))
            """

def compute_accuracy(image_model, att_model, relation_net, att_proj, test_loader, test_att, test_id, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpred = []
    test_label = []
    
    test_att = torch.from_numpy(test_att)
    test_att = test_att.float().to(device)
    test_att_output = att_model(test_att)
    test_size = test_att_output.size()[0]
    
    for i, (image_input, att_input, cls_id, key, label) in enumerate(test_loader):
        att_input = att_input.float().to(device)
        label = list(label.numpy())
        image_input = image_input.float().to(device)
        image_input = image_input.squeeze(1)

        att_output = att_proj(att_input)
        image_output, attn_map = image_model(image_input, att_output)
        
        for j in range(len(label)):
            output_repeat = image_output[j, :].repeat(test_size, 1)
            output = relation_net(torch.cat((output_repeat, test_att_output), dim=-1))
            index = int(torch.max(output[:, 0], -1)[1])
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
