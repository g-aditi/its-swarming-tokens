import random

class GAEngine:
    def __init__(self, token_mapping):
        self.tm = token_mapping

    def elitism(self, ranked_ids, elite_ratio, pop_size):
        elite_n = max(1, int(elite_ratio * pop_size))
        return ranked_ids[:elite_n]

    def crossover(self, parent_a, parent_b):
        # cross by taking neighbor from either parent
        if random.random() < 0.5:
            neigh = self.tm.nearest_neighbors(parent_a, 8)
            return random.choice(neigh) if neigh else parent_a
        else:
            neigh = self.tm.nearest_neighbors(parent_b, 8)
            return random.choice(neigh) if neigh else parent_b

    def mutate(self, tid):
        if random.random() < 0.08:
            neigh = self.tm.nearest_neighbors(tid, 16)
            return random.choice(neigh) if neigh else tid

        if random.random() < 0.02:
            return random.choice(self.tm.allowed_ids)

        return tid
