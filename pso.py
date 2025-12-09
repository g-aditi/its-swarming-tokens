import torch
import random

class PSOEngine:
    def __init__(self, emb_dim, w=0.5, c1=1.5, c2=1.5):
        self.emb_dim = emb_dim
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update_particle(self, pos, vel, p_best, g_best):
        r1 = torch.rand_like(pos)
        r2 = torch.rand_like(pos)

        vel = self.w * vel + self.c1 * r1 * (p_best - pos) + self.c2 * r2 * (g_best - pos)
        pos = pos + vel
        return pos, vel
