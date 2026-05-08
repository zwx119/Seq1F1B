from sympy import symbols, Eq, solve

def round_down(x, tp_size):
    if tp_size <= 1:
        return x
    return x // tp_size * tp_size
class solver:
    def __init__(self, total_seqlen, config, causal=True):
        self.total_seqlen = total_seqlen 
        self.config = config
        self.total_tflops = config.get_seq_tflops(total_seqlen, causal)
        

    def solve_partition(self, num_splits, tp_size=1):
        if tp_size > 1 and self.total_seqlen % tp_size == 0:
            return self.solve_aligned_partition(num_splits, tp_size)

        res = []
        prefix = self.total_seqlen
        for i in range(1, num_splits):
            seqlen = symbols('seqlen')
            tflops = self.config.get_prefix_tflops(seqlen, prefix)
            eq = Eq(tflops, self.total_tflops / num_splits)
            sol = solve(eq, seqlen)
            sol = round_down(int(sol[0]), tp_size)
            if sol <= 0:
                sol = tp_size
            res.insert(0, int(sol))
            prefix -= int(sol)
        res.insert(0, prefix)
        return res

    def solve_aligned_partition(self, num_splits, align):
        units = self.total_seqlen // align
        if units < num_splits:
            raise ValueError(
                f"Cannot split sequence length {self.total_seqlen} into "
                f"{num_splits} positive chunks aligned to {align}."
            )

        target = self.total_tflops / num_splits
        inf = float("inf")
        dp = [[inf] * (units + 1) for _ in range(num_splits + 1)]
        prev = [[-1] * (units + 1) for _ in range(num_splits + 1)]
        dp[0][0] = 0.0

        for k in range(1, num_splits + 1):
            min_end = k
            max_end = units - (num_splits - k)
            for end in range(min_end, max_end + 1):
                for start in range(k - 1, end):
                    if dp[k - 1][start] == inf:
                        continue
                    seqlen = (end - start) * align
                    prefix = end * align
                    cost = self.config.get_prefix_tflops(seqlen, prefix)
                    score = max(dp[k - 1][start], abs(cost - target))
                    if score < dp[k][end]:
                        dp[k][end] = score
                        prev[k][end] = start

        splits = []
        end = units
        for k in range(num_splits, 0, -1):
            start = prev[k][end]
            if start < 0:
                raise RuntimeError("Failed to solve aligned sequence partition.")
            splits.insert(0, (end - start) * align)
            end = start
        return splits
        

if __name__ == "__main__":
    from sp_utils import SeqTFlops
    kw = {
        "num_layers": 24,
        "hidden_size": 4096,
        "ffn_size": 16384,
        "num_heads": 32,
        "dim_head": 128,
        "vocab_size": 32000
    }
    config = SeqTFlops(**kw)
    s = solver(16384, config)
    s.solve_partition(4, 2)
        
        
        
        
    
    
    
    
