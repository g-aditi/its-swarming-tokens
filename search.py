import torch
import random
from ga import GAEngine
from pso import PSOEngine

class HybridSearch:
    def __init__(self, config, surrogate, evaluator, token_mapping):
        self.cfg = config
        self.sur = surrogate
        self.eval = evaluator
        self.tm = token_mapping

        self.pop_size = config.population_size
        self.generations = config.generations

        self.ga = GAEngine(token_mapping)
        self.pso = PSOEngine(token_mapping.emb_dim)

        self.stat_store = {}
        self.discovered = set()

    def initialize_population(self):
        pops = []
        for _ in range(self.pop_size):
            tid = random.choice(self.tm.allowed_ids)
            pops.append(tid)
        return pops

    def run(self, verbose=True):
        population = self.initialize_population()

        # init PSO embedding states
        pos = [self.tm.embedding[tid].clone() for tid in population]
        vel = [0.01 * torch.randn_like(p) for p in pos]

        personal_best = pos.copy()
        personal_best_score = [-1e9] * len(pos)

        g_best = personal_best[0]
        g_best_score = -1e9

        for gen in range(self.generations):
            if verbose:
                print(f"[Gen {gen+1}/{self.generations}]")

            ############################################################
            # 1. SURROGATE SCORING
            ############################################################
            surrogate_candidates = []  # list of (token_id, approx_score)

            for tid in population:
                ids, scores, _ = self.sur.surrogate(tid, self.tm)
                # keep only top 20 surrogate neighbors for logging clarity
                for cid, sc in zip(ids[:20], scores[:20]):
                    surrogate_candidates.append((cid, float(sc)))

            # dedupe <-- best approx score kept
            surrogate_map = {}
            for cid, sc in surrogate_candidates:
                if cid not in surrogate_map or sc > surrogate_map[cid]:
                    surrogate_map[cid] = sc

            # sorted list by approx score
            sorted_surrogate = sorted(
                surrogate_map.items(), key=lambda x: x[1], reverse=True
            )

            # choose exact evaluation set
            eval_ids = [cid for cid, _ in sorted_surrogate[: self.cfg.batch_eval_size * 4]]

            ############################################################
            # 2. EXACT EVALUATION
            ############################################################
            exact_res = self.eval.evaluate(eval_ids, self.tm)

            # merge exact stats into global stat store
            for tid, r in exact_res.items():
                self.stat_store[tid] = r

            ############################################################
            # 3. GLITCH CHECK (STRICT)
            ############################################################
            glitch_results = self.eval.glitch_check(eval_ids, self.tm)
            for tid, is_g in glitch_results.items():
                if is_g:
                    self.discovered.add(tid)

            ############################################################
            # ----------- GENERATION REPORTING ------------------------
            ############################################################
            if verbose:
                print(f"\n=== GENERATION {gen+1} REPORT ===")

                # Print surrogate candidates
                print("\nSurrogate candidates (top 25):")
                for cid, sc in sorted_surrogate[:25]:
                    tok = self.tm.tf.decode([cid])
                    print(f"  {cid:6d}  {repr(tok):20s}  approx_score={sc:.4f}")

                # Print exact evaluated
                print("\nExact evaluated candidates:")
                exact_sorted = sorted(exact_res.items(), key=lambda x: x[1]["mean"], reverse=True)
                for tid, r in exact_sorted:
                    tok = self.tm.tf.decode([tid])
                    print(
                        f"  {tid:6d}  {repr(tok):20s}  entropy_mean={r['mean']:.6f}  std={r['std']:.6f}  n={r['n']}"
                    )

                # Print glitch results
                print("\nGlitch results this generation:")
                for tid, is_g in glitch_results.items():
                    tok = self.tm.tf.decode([tid])
                    print(
                        f"  {tid:6d}  {repr(tok):20s}  GLITCH={is_g}"
                    )

            ############################################################
            # 4. UCB SCORE FOR NEXT-GEN SELECTION
            ############################################################
            ucb_scores = []
            for tid, r in self.stat_store.items():
                ucb = r["mean"] + self.cfg.ucb_beta * r["std"]
                ucb_scores.append((tid, ucb))

            ranked = sorted(ucb_scores, key=lambda x: x[1], reverse=True)
            population = [tid for tid, _ in ranked[: self.pop_size]]

            ############################################################
            # 5. GENETIC ALGORITHM UPDATE
            ############################################################
            elites = self.ga.elitism([tid for tid, _ in ranked], 0.1, self.pop_size)
            new_pop = elites.copy()

            while len(new_pop) < self.pop_size:
                parent_a = random.choice(population)
                parent_b = random.choice(population)
                child_tid = self.ga.crossover(parent_a, parent_b)
                child_tid = self.ga.mutate(child_tid)
                new_pop.append(child_tid)

            population = new_pop

            ############################################################
            # 6. PSO UPDATE
            ############################################################
            for i in range(self.pop_size):
                pos[i], vel[i] = self.pso.update_particle(
                    pos[i], vel[i], personal_best[i], g_best
                )
                mapped_tid = self.tm.embedding_to_token(pos[i])

                # score for mapped_tid
                info = self.stat_store.get(mapped_tid, {"mean": -1e9})
                score = info["mean"]

                # personal best update
                if score > personal_best_score[i]:
                    personal_best_score[i] = score
                    personal_best[i] = pos[i].clone()

                # global best update
                if score > g_best_score:
                    g_best_score = score
                    g_best = pos[i].clone()

        ############################################################
        # END OF ALL GENERATIONS
        ############################################################
        return list(self.discovered), self.stat_store
