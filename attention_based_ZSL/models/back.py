class AttentionLoss(nn.Module):
    def __init__(self, gamma1, gamma2, gamma3):
        super(AttentionLoss, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        return

    # image_features batch_size*150, 128, 256
    # att_features batch_size*150, 312, 128
    # nef 128
    def forward(self, img_features, att_features, labels, args):
        '''
        # batch_size*150, 312, 256
        # equation 7
        attn = torch.bmm(att_features, img_features)
        #batch_size*150, 256, 312
        attn = torch. transpose(attn, 1, 2)

        # special normalize
        # equation 8
        batch_size, img_dim, att_dim = attn.size()
        attn = attn.contiguous().view(-1, att_dim)
        attn = nn.Softmax()(attn)
        attn = attn.view(batch_size, img_dim, att_dim)

        # equation 9
        gamma1 = self.gamma1
        attn = torch.exp(gamma1 * attn)
        # batch_size*150, 312, 256
        divisor = torch.sum(attn, 2)
        alpha = attn.permute(2, 0, 1) / divisor
        # batch_size*150, 312, 256
        alpha = alpha.permute(1, 2, 0)

        # equation 10
        # batch_size*150, 128, 312
        context = torch.matmul(img_features, alpha.permute(0, 2, 1)).permute(0, 2, 1)
        '''

        # --> batch x ndf x queryL
        weiContext, attn = self.func_attention(att_features, img_features)

        weiContext = weiContext.transpose(1, 2).contiguous()
        gamma2 = self.gamma2
        # batch_size*150, 312
        cos_similarity = F.cosine_similarity(att_features, weiContext)
        cos_similarity = torch.exp(gamma2 * cos_similarity)
        # batch_size*150
        # 此处和论文相符，但和源码不符
        cos_similarity = torch.log(
            torch.pow(torch.sum(cos_similarity, 1), 1 / gamma2))

        # equation 11
        gamma3 = self.gamma3
        cos_similarity = gamma3 * cos_similarity
        pred = cos_similarity.view(args.batch_size, -1)
        loss = F.cross_entropy(pred, labels)
        return loss

    def func_attention(self, query, context):
        """
        query: batch x ndf x queryL
        context: batch x ndf x ih x iw (sourceL=ihxiw)
        mask: batch_size x sourceL
        """
        batch_size, queryL = query.size(0), query.size(1)
        sourceL = context.size(2)

        # --> batch x sourceL x ndf
        context = context.view(batch_size, -1, sourceL)
        contextT = torch.transpose(context, 1, 2).contiguous()

        # Get attention
        # (batch x sourceL x ndf)(batch x ndf x queryL)
        # -->batch x sourceL x queryL
        attn = torch.bmm(contextT, query.transpose(1, 2))  # Eq. (7) in AttnGAN paper
        # --> batch*sourceL x queryL
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)  # Eq. (8)

        # --> batch x sourceL x queryL
        attn = attn.view(batch_size, sourceL, queryL)
        # --> batch*queryL x sourceL
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = attn.view(batch_size * queryL, sourceL)
        #  Eq. (9)
        attn = attn * self.gamma1
        attn = nn.Softmax()(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # (batch x ndf x sourceL)(batch x sourceL x queryL)
        # --> batch x ndf x queryL
        weightedContext = torch.bmm(context, attnT)

        return weightedContext, attn

'''
# image_features: batch_size*200. 1024, 256
# att_features: batch_size*200, 312, 1024
def attention_loss(img_features, att_features, labels, criterion, args):
    # batch_size*200, 312, 256
    similarity = torch.matmul(att_features, img_features)

    # special normalize
    similarity = torch.exp(similarity)
    divisor = torch.sum(similarity, 1)
    similarity = similarity.permute(1, 0, 2)
    similarity = similarity / divisor
    similarity = similarity.permute(1, 0, 2)

    gamma1 = 5
    similarity = torch.exp(gamma1 * similarity)
    # batch_size*200, 312, 256
    divisor = torch.sum(similarity, 2)
    alpha = similarity.permute(2, 0, 1) / divisor
    # batch_size*200, 312, 256
    alpha = alpha.permute(1, 2, 0)

    # batch_size*200, 1024, 312
    context = torch.matmul(img_features, alpha.permute(0,2,1))

    gamma2 = 5
    # batch_size*200, 312
    cos_similarity = F.cosine_similarity(att_features, context)
    cos_similarity = torch.exp(gamma2 * cos_similarity)
    # batch_size*200
    cos_similarity = torch.log(
        torch.pow(torch.sum(cos_similarity, 1), 1/gamma2))

    gamma3 = 10
    cos_similarity = gamma3 * cos_similarity
    pred = cos_similarity.view(args.batch_size, -1)
    loss = criterion(pred, labels)
    return loss
'''