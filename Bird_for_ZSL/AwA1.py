#对视觉类原型和att分别进行交替更新

from scipy import io
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
import kNN
import random
import scipy.io as sio
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class MLP_att(nn.Module):
	def __init__(self,inp_dim,hid_dim,out_dim):
		super(MLP_att, self).__init__()
		self.fc1 = nn.Linear(inp_dim,hid_dim)
		self.fc2 = nn.Linear(hid_dim,out_dim)
		self.relu = nn.ReLU()
		# self.fc1.weight.data.normal_(0,0.02)
		# self.fc2.weight.data.normal_(0,0.02)
		nn.init.normal_(self.fc1.weight,0,0.02)
		nn.init.normal_(self.fc2.weight,0,0.02)
		nn.init.constant_(self.fc1.bias,0)
		nn.init.constant_(self.fc2.bias,0)

	def forward(self,x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		return x

class MLP_img(nn.Module):
	def __init__(self,inp_dim,hid_dim,out_dim):
		super(MLP_img, self).__init__()
		self.fc1 = nn.Linear(inp_dim,hid_dim)
		self.fc2 = nn.Linear(hid_dim,out_dim)
		self.relu = nn.ReLU()
		nn.init.eye_(self.fc1.weight)
		nn.init.eye_(self.fc2.weight)
		nn.init.constant_(self.fc1.bias,0)
		nn.init.constant_(self.fc2.bias,0)

	def forward(self,x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		return x


def pdist(x1,x2):
    x1_square = x1.pow(2).sum(1).reshape(-1,1)
    x2_square = x2.pow(2).sum(1).reshape(1,-1)
    dist2 = torch.clamp((x1_square-2*torch.mm(x1,x2.t())+x2_square),min=0,max=1e6)
    dist = torch.sqrt(dist2)
    # dist = torch.pow(x1-x2.t(),2).sum()
    # dist/=dist.size(0)
    return dist

def distance(feature,centers):
	f2 = torch.pow(feature,2).sum(1)
	c2 = torch.pow(centers,2).sum(1)
	dist = f2.unsqueeze(-1).repeat(1,c2.shape[0])-torch.mm(feature,centers.t())+c2.unsqueeze(-1).t().repeat(f2.shape[0],1)
	return dist

def compute_accuracy(test_att, test_visual, test_id, test_label):     
	outpre = [0]*test_visual.shape[0]  # CUB 2933
	test_att = Variable(torch.from_numpy(test_att).float().cuda())
	test_visual = Variable(torch.from_numpy(test_visual).float().cuda())
	
	att_pre = forward(test_att)
	att_pre = att_pre / att_pre.norm(dim=1,keepdim=True)
	
	# test_visual = forward_img(test_visual)
	test_visual = test_visual / test_visual.norm(dim=1,keepdim=True)
	att_pre = att_pre.cpu().data.numpy()
	test_visual = test_visual.cpu().data.numpy()
	
	test_id = np.squeeze(np.asarray(test_id))

	test_label = np.squeeze(np.asarray(test_label))
	test_label = test_label.astype("float32")
	
	for i in range(test_visual.shape[0]):  # CUB 2933
		outputLabel = kNN.kNNClassify(test_visual[i,:], att_pre, test_id, 1)
		outpre[i] = outputLabel
	#compute averaged per class accuracy
	outpre = np.array(outpre, dtype='int')
	unique_labels = np.unique(test_label)
	acc = 0
	for l in unique_labels:
		idx = np.nonzero(test_label == l)[0]
		acc += accuracy_score(test_label[idx], outpre[idx])
	acc = acc / unique_labels.shape[0]
	return acc


def sample_postive(labels, lab_indxs):        
	img_feats_b = []
	for i in range(len(labels)):
		label = labels[i]
		anchor_indx = lab_indxs[i]
		img_indx = anchor_indx
		indxs = np.where(labels==label)[0]
		while img_indx==anchor_indx:
			img_indx = np.random.choice(indxs,1,replace=False)
		
		img_feats_b.append(train_x[img_indx])
	img_feats_b = np.concatenate(img_feats_b,axis=0)
	
	return img_feats_b

def data_iterator():
	""" A simple data iterator """
	batch_idx = 0
	while True:
		# shuffle labels and features
		idxs = np.arange(0, len(train_x))
		np.random.shuffle(idxs)
		shuf_visual = train_x[idxs]
		shuf_att = train_att[idxs]
		# shuf_proto = train_protos[idxs]
		shuf_lab = train_label[idxs]
		shuf_lab_seq = train_label_seq[idxs]
		batch_size = 100   #batch = 100   64好于100

		for batch_idx in range(0, len(train_x), batch_size):
			visual_batch = shuf_visual[batch_idx:batch_idx + batch_size]
			visual_batch = visual_batch.astype("float32")
			labels = shuf_lab[batch_idx:batch_idx + batch_size]
			labels_seq = shuf_lab_seq[batch_idx:batch_idx + batch_size]
			labels_seq = Variable(torch.from_numpy(labels_seq).long().cuda())
			att_batch = shuf_att[batch_idx:batch_idx + batch_size]
			# proto_batch = shuf_proto[batch_idx:batch_idx + batch_size]
			indxs = np.arange(batch_idx,batch_idx + batch_size)
			att_batch = Variable(torch.from_numpy(att_batch).float().cuda())
			# proto_batch = Variable(torch.from_numpy(proto_batch).float().cuda())
			visual_batch = Variable(torch.from_numpy(visual_batch).float().cuda())	
			# visual_batch = forward_img(visual_batch)		
			img_feats_p = forward_img(visual_batch)
			att_pred = forward(att_batch)
			
			"""
			#获取图像正样本
			img_feats_b = []
			for i in range(len(labels)):
				label = labels[i]
				anchor_indx = i + batch_idx*batch_size
				img_indx = anchor_indx
				inds = np.where(shuf_lab == label)[0]
				while img_indx==anchor_indx:
					img_indx = np.random.choice(inds,1,replace=False)
				
				img_feats_b.append(shuf_visual[img_indx])
			img_feats_b = np.concatenate(img_feats_b,axis=0)
			pos_img_feats = Variable(torch.from_numpy(img_feats_b).float().cuda())
			train_att_clss = Variable(torch.from_numpy(train_att_single).float().cuda())
			train_att_clss_p = forward(train_att_clss)  ######################语义负样本
			

			# img_feats_p = visual_batch
			att_dist = pdist(att_batch,att_batch).cpu().data.numpy()
			neg_mask = (att_dist.astype(bool)).astype(int)
			
			img_dist = pdist(img_feats_p,img_feats_p).cpu().data.numpy()
			att_dist = pdist(att_pred,att_pred).cpu().data.numpy()
			neg_img_dist = img_dist*neg_mask
			neg_att_dist = att_dist*neg_mask
			neg_img_indxs = []
			#挖掘图像的负样本
			for i in range(neg_img_dist.shape[0]):
				feat_row = neg_img_dist[i,:]
				dists = np.unique(feat_row)
				dists = sorted(list(dists))          
				mini_dis = dists[1]
				neg_img_indx = np.where(feat_row == mini_dis)[0][0]
				neg_img_indxs.append(neg_img_indx)
			neg_img_indxs = np.array(neg_img_indxs)
			neg_img_feats = visual_batch[neg_img_indxs]
			
			#挖掘语义的负样本
			att_dist_n = pdist(train_att_clss_p,att_pred).cpu().data.numpy()
			neg_att_indxs = []
			for i in range(att_pred.shape[0]):				
				att_dist_n_col = att_dist_n[:,i]
				att_dist_n_sort = sorted(att_dist_n_col)
				att_n_indx = np.where(att_dist_n_col==att_dist_n_sort[1])[0]
				# n_att_feat = train_att_clss[att_n_indx]
				# anchor_label = labels[i]
				# sample_label = train_id[att_n_indx]
				neg_att_indxs.append(att_n_indx[0])
			neg_att_indxs = np.array(neg_att_indxs)
			neg_att_indxs = neg_att_indxs.reshape(-1,1)
			
			neg_att_indxs = neg_att_indxs.squeeze(1)
			n_att_feats = train_att_clss[neg_att_indxs]     
			                   
			"""

			yield  labels_seq, visual_batch #, proto_batch#att_batch, visual_batch, neg_img_feats,pos_img_feats, n_att_feats


dataset = 'E:\\Shawn\\code\\zero_shot\\structure-preserve\\PP\CUB\\AwA1_data'
image_embedding = 'res101' 
class_embedding = 'original_att_splits'

matcontent = sio.loadmat(dataset + "/" + image_embedding + ".mat")
feature = matcontent['features'].T
label_all = matcontent['labels'].astype(int).squeeze() - 1
matcontent = sio.loadmat(dataset + "/" + class_embedding + ".mat")
# numpy array index starts from 0, matlab starts from 1
trainval_loc = matcontent['trainval_loc'].squeeze() - 1
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

attribute = matcontent['att'].T   #att

x = feature[trainval_loc] # train_features (7057,2048)
train_label = label_all[trainval_loc].astype(int)  # train_label(7057,)
att = attribute[train_label] # train attributes(7057,312)

x_test = feature[test_unseen_loc]  # test_feature (2967, 2048)
test_label = label_all[test_unseen_loc].astype(int) # test_label (2967,)
x_test_seen = feature[test_seen_loc]  #test_seen_feature (1764, 2048)
test_label_seen = label_all[test_seen_loc].astype(int) # test_seen_label (1764,)
train_id = np.unique(train_label)
test_id = np.unique(test_label)   # test_id  (50,)
att_pro = attribute[test_id]      # test_attribute (50, 312)
train_att_single = attribute[train_id]
label_id = np.unique(label_all)
class_num = label_id.shape[0]

seq_labels_for_train = np.zeros(label_id.shape[0])
for i in range(40):	
	seq_labels_for_train[train_id[i]] = i

train_label_seq = seq_labels_for_train[train_label]


train_x = x
train_att = att

test_att=att_pro 
test_x = x_test
test_att2label  = test_id
test_x2label=test_label

unique_labels = np.unique(label_all)
features_protos = []
i = 0
for label in unique_labels:
	one_class_indexs = np.where(label_all==label)[0]
	one_class_features = feature[one_class_indexs]
	features_proto = one_class_features.mean(0).reshape(1,-1)
	if i == 0:
		features_protos = features_proto
		i = 1
	else:
		features_protos = np.vstack((features_protos,features_proto))

train_proto_s = features_protos[train_id]
train_protos = features_protos[train_label]

train_labels = np.arange(0,40,1)


# f = io.loadmat('CUB_data/train_attr.mat')
# train_att = np.array(f['train_attr'])
# print('train attr:', train_att.shape)

# f = io.loadmat('CUB_data/train_cub_googlenet_bn.mat')
# train_x = np.array(f['train_cub_googlenet_bn'])
# print('train x:', train_x.shape)

# f = io.loadmat('CUB_data/test_cub_googlenet_bn.mat')
# test_x = np.array(f['test_cub_googlenet_bn'])
# print('test x:', test_x.shape)

# f = io.loadmat('CUB_data/test_proto.mat')
# test_att = np.array(f['test_proto'])
# print('test att:', test_att.shape)

# f = io.loadmat('CUB_data/test_labels_cub.mat')
# test_x2label = np.squeeze(np.array(f['test_labels_cub']))
# print('test x2label:', test_x2label)

# f = io.loadmat('CUB_data/testclasses_id.mat')
# test_att2label = np.squeeze(np.array(f['testclasses_id']))
# print('test att2label:', test_att2label)

# att_weights = np.load('att_weights.npy')
# w1 = att_weights[0]
# b1 = att_weights[1]
# w2 = att_weights[2]
# b2 = att_weights[3]

# w1 = Variable(torch.from_numpy(w1).cuda(),requires_grad = False)
# b1 = Variable(torch.from_numpy(b1).cuda(),requires_grad = False)
# w2 = Variable(torch.from_numpy(w2).cuda(),requires_grad = False)
# b2 = Variable(torch.from_numpy(b2).cuda(),requires_grad = False)

# forward = MLP_att(85,800,2048).cuda()
# forward_img = MLP_img(2048,2048,2048).cuda()

w1 = Variable(torch.FloatTensor(85, 800).cuda(), requires_grad=True)  #1000
b1 = Variable(torch.FloatTensor(800).cuda(), requires_grad=True)
w2 = Variable(torch.FloatTensor(800, 2048).cuda(), requires_grad=True)
b2 = Variable(torch.FloatTensor(2048).cuda(), requires_grad=True)
w = np.eye(2048,k=0,dtype='float32')
w3 = Variable(torch.from_numpy(w).cuda(),requires_grad = True)
w4 = Variable(torch.from_numpy(w).cuda(),requires_grad = True)
b3 = Variable(torch.FloatTensor(2048).cuda(), requires_grad=True)
b4 = Variable(torch.FloatTensor(2048).cuda(), requires_grad=True)

# train_proto_single = Variable(torch.FloatTensor(class_num,2048).cuda(),requires_grad=True)
# train_proto_single.data.normal_(0,1)
train_proto_single = Variable(torch.from_numpy(train_proto_s).float().cuda(),requires_grad=True)

# must initialize!
w1.data.normal_(0, 0.02)
w2.data.normal_(0, 0.02)
# w3.data.normal_(0, 0.02)
# w4.data.normal_(0, 0.02)
b1.data.fill_(0)
b2.data.fill_(0)
b3.data.fill_(0)
b4.data.fill_(0)





def forward(att):
	a1 = F.relu(torch.mm(att, w1) + b1)
	a2 = F.relu(torch.mm(a1, w2) + b2)

	return a2

def forward_img(img):
	d1 = F.relu(torch.mm(img, w3) + b3)
	d2 = F.relu(torch.mm(d1, w4) + b4)
	return d2

def getloss(pred, x, nx, px, n_pred): #
	# norm_pred = pred / pred.norm(dim=1,keepdim=True)
	# norm_x = x / x.norm(dim=1,keepdim=True)
	# norm_nx = nx /nx.norm(dim=1,keepdim=True)
	# norm_px = px /px.norm(dim=1,keepdim=True)
	
	C_p_loss = torch.pow(x - pred, 2).sum()
	C_p_loss /= x.size(0)
	
	# C_n_loss = torch.pow(pred - nx, 2).sum()  #原来是负例图像和att之间
	
	C_n_loss = torch.pow(n_pred - x, 2).sum()
	# C_n_loss = torch.pow(n_pred - pred, 2).sum()
	C_n_loss /= x.size(0)
	margin1 = 0.6*(C_p_loss+C_n_loss)

	C_loss = F.relu(margin1 + C_p_loss - C_n_loss)
	
	# if C_loss<=0:
	# 	C_loss = C_p_loss
	
	# Bi loss
	B_p_loss = torch.pow(pred - px, 2).sum()
	B_p_loss /= x.size(0)
	
	B_n_loss = torch.pow(pred - nx, 2).sum()
	B_n_loss /= x.size(0)
	margin_B = 0.03*(B_p_loss+B_n_loss)
	B_loss = F.relu(margin_B + B_p_loss - B_n_loss)
	
	
	#keep structure
	K_p_loss = torch.pow(x - px, 2).sum()
	K_p_loss /= x.size(0)
	
	K_n_loss = torch.pow(x - nx, 2).sum()
	K_n_loss /= x.size(0)
	
	margin = 0.01*(K_p_loss+K_n_loss)
	# margin = 0.06*K_p_loss
	# K_loss = F.relu(K_p_loss - 0.9*K_n_loss)
	K_loss = F.relu(margin + K_p_loss - K_n_loss)
	# if K_loss == 0:
	# 	loss = C_loss
	# else:
	loss = C_loss + 5*K_loss  # + 0.05*B_loss
	return loss 

# optimizer = torch.optim.Adam([w3, b3, w4, b4], lr=1e-5, weight_decay=1e-2)  #,w3, b3, w4, b4
# params_att = filter(lambda p: p.requires_grad, forward.parameters())
# params_img = filter(lambda p: p.requires_grad, forward_img.parameters())

# optimizer1 = torch.optim.Adam(forward.parameters(), lr=1e-5, weight_decay=1e-2)
# optimizer2 = torch.optim.Adam(forward_img.parameters(), lr=1e-7, weight_decay=1e-2)
optimizer1 = torch.optim.Adam([w1,w2,b1,b2,train_proto_single], lr=1e-5, weight_decay=1e-3)
optimizer2 = torch.optim.Adam([train_proto_single], lr=1e-6, weight_decay=1e-2) #w3,w4,b3,b4
scheduler1 = StepLR(optimizer1,step_size=5000,gamma=1)
scheduler2 = StepLR(optimizer2,step_size=5000,gamma=1)
# # Run
iter_ = data_iterator()
pre_acc = 0
train_att_single = Variable(torch.from_numpy(train_att_single).float().cuda())
# train_proto_single = Variable(torch.from_numpy(train_proto_single).float().cuda())
train_labels = Variable(torch.from_numpy(train_labels).long().cuda())
# train_labels = torch.LongTensor(train_labels)
for i in range(1000000):
	labels_seq, visual_batch_val= next(iter_)  #, proto_batch
	"""
	att_batch_val, visual_batch_val,neg_visual_batch,pos_visual_batch,neg_att_batch= next(iter_)  # 

	pred = forward(att_batch_val)	
	pred_visual = forward_img(visual_batch_val)	
	pred_neg_visual = forward_img(neg_visual_batch)
	pred_pos_visual = forward_img(pos_visual_batch)
	pred_neg_att = forward(neg_att_batch)   #############负例语义
	# pred_visual = visual_batch_val
	# pred_neg_visual = neg_visual_batch
	
	
	# loss = getloss(pred, visual_batch_val)
	loss = getloss(pred,pred_visual,pred_neg_visual,pred_pos_visual,pred_neg_att) #
	"""	
	
	# with torch.no_grad():
	# train_proto_single = forward_img(train_proto_single)
	# simlarity = torch.mm(visual_batch_val,train_proto_single.t())	
	# simlarity2 = torch.mm(embed_att,train_proto_single_no_grad.t())
	# simlarity = -distance(visual_batch_val,train_proto_single)
	
	# simlarity2 = -distance(embed_att,train_proto_single_no_grad)
	# one_hot_label = Variable(torch.zeros(40, 40).scatter_(1, train_labels.view(-1,1), 1)).cuda()
	# mse = nn.MSELoss().cuda()
	# struct_loss = torch.pow(visual_batch_val - proto_batch, 2).sum()
	# loss =  F.cross_entropy(simlarity2,train_labels)#+ struct_loss# +F.cross_entropy(simlarity,labels_seq) +
	
	train_proto_single.requires_grad = True
	# visual_batch_val = forward_img(visual_batch_val)
	optimizer2.zero_grad()
	# loss2 = torch.pow(visual_batch_val - , 2).sum()
	simlarity = torch.mm(visual_batch_val,train_proto_single.t())
	loss2 = F.cross_entropy(simlarity,labels_seq)
	scheduler2.step()
	loss2.backward()
# gradient clip makes it converge much faster!
# torch.nn.utils.clip_grad_norm(forward.parameters(), 1)  #,w3, b3, w4, b4
# torch.nn.utils.clip_grad_norm(forward_img.parameters(), 1)  #,w3, b3, w4, b4
	torch.nn.utils.clip_grad_norm([w3,b3,w4,b4], 1)	
	optimizer2.step()
	
	if i % 500 ==0:
		for j in range(1000):
			train_proto_single.requires_grad = False
			with torch.no_grad():
				train_proto_single_no_grad = train_proto_single		
			embed_att = forward(train_att_single)
			loss1 = torch.pow(embed_att - train_proto_single_no_grad, 2).sum()
		# loss = mse(simlarity,one_hot_label)			
			optimizer1.zero_grad()			
			scheduler1.step()
			loss1.backward()
		# gradient clip makes it converge much faster!
		# torch.nn.utils.clip_grad_norm(forward.parameters(), 1)  #,w3, b3, w4, b4
		# torch.nn.utils.clip_grad_norm(forward_img.parameters(), 1)  #,w3, b3, w4, b4
			torch.nn.utils.clip_grad_norm([w1,b1,w2,b2], 1)			
			optimizer1.step()
		# if i % 10==0:
		# 	print ('iter%d'%i)
	

	if i %1000 == 0:
		
		acc_zsl = compute_accuracy(test_att, test_x, test_att2label, test_x2label)
		acc_seen_gzsl = compute_accuracy(attribute, x_test_seen, label_id, test_label_seen)
		acc_unseen_gzsl = compute_accuracy(attribute, test_x, label_id, test_x2label)
		H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)
		if acc_zsl>pre_acc:	
			pre_acc = acc_zsl
			pre_seen = acc_seen_gzsl
			pre_useen = acc_unseen_gzsl
			pre_H = H
			test_visual = Variable(torch.from_numpy(test_x).float().cuda())
			processed_visal = forward_img(test_visual).cpu().detach().numpy()

		print('itr: %d | zsl: ACC=%.4f | gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (i,acc_zsl,acc_seen_gzsl, acc_unseen_gzsl, H))
		# print('--------------------------------------------------------------------')
		# print('Best_zsl:', pre_acc)
		# print('According_gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (pre_seen, pre_useen, pre_H))
			
