import torch
import torch.nn as nn
import torch.nn.functional as F


# YEAR: 21.01.29
# ORTHOGONAL EMBEDDED MEMORY: TESTING
class OrthogonalEmbeddedMemory(nn.Module):
    def __init__(self, memory_shape, isinitialize=False):
        super(OrthogonalEmbeddedMemory, self).__init__()

        # Check type of memory size: torch.Size
        assert isinstance(memory_shape, torch.Size)

        # Memory configuration
        self.isinitialize = isinitialize
        self.memory_size = 70
        self.memory_shape = memory_shape
        self.memory_capacity = memory_shape.numel()

        # Single Memory parameters
        # initialize parameters
        self.memory = nn.Parameter(torch.zeros([self.memory_size, self.memory_capacity]), requires_grad=True)
        nn.init.orthogonal_(self.memory)

    def cosine_similarity(self, pool_feat_adv):
        # feat_dot_product_memory  : (Batch, self.memory_size, self.memory_capacity)
        # feat_inner_product_memory: (Batch, self.memory_size)
        # feat_memory_magnitude    : (Batch, self.memory_size)
        feat_dot_product_memory = pool_feat_adv.unsqueeze(dim=1) * self.memory
        feat_inner_product_memory = feat_dot_product_memory.sum(dim=2)

        feat_memory_magnitude = torch.sqrt(pool_feat_adv.square().sum(dim=1)).unsqueeze(dim=1)\
                                            *torch.sqrt(self.memory.square().sum(dim=1)).unsqueeze(dim=0)
        return feat_inner_product_memory / (feat_memory_magnitude+1e-10)

    def inner_product(self, pool_feat_adv):
        # feat_dot_product_memory  : (Batch, self.memory_size, self.memory_capacity)
        # feat_inner_product_memory: (Batch, self.memory_size)
        # feat_magnitude    : (Batch, self.memory_size)
        feat_dot_product_memory = pool_feat_adv.unsqueeze(dim=1) * self.memory
        feat_inner_product_memory = feat_dot_product_memory.sum(dim=2)
        feat_magnitude = torch.sqrt(pool_feat_adv.square().sum(dim=1)).unsqueeze(dim=1)

        return feat_inner_product_memory / feat_magnitude

    def forward(self, feat_adv, train=False):


        # self.memory   : (self.memory_size, self.memory_capacity)
        # feat_adv      : (Batch, Channel, Height, Width)
        # pool_feat_adv : (Batch, self.memory_capacity=Channel*Height*Width)
        # cs_memory     : (Batch, self.memory_size)
        # memory_matrix : (Batch, self.memory_size, self.memory_capacity)
        # addressing_memory            : (Batch, self.memory_size, self.memory_capacity)
        # orthogonal_memory_matrix     : (Batch, self.memory_size, self.memory_capacity)
        # symmetric_martix             : (Batch, self.memory_size, self.memory_size)

        pool_feat_adv = feat_adv.view(-1, self.memory_capacity)
        if not self.isinitialize:
            print("Memory Initialize")
            self.memory.data   = pool_feat_adv[:self.memory_size,:].data
            self.isinitialize=True
        cs_memory = self.inner_product(pool_feat_adv)
        memory_matrix = cs_memory.unsqueeze(dim=2) * self.memory.unsqueeze(dim=0)
        feat_memory = memory_matrix.sum(dim=1).view(feat_adv.shape)

        if not train:
            return feat_memory

        orthogonal_memory_matrix = memory_matrix / torch.sqrt(memory_matrix.square().sum(dim=2)).unsqueeze(dim=2)
        symmetric_martix = orthogonal_memory_matrix.matmul(orthogonal_memory_matrix.transpose(2, 1))
        memory_loss = (symmetric_martix-torch.eye(self.memory_size).cuda()).square().sum()

        return feat_memory, memory_loss