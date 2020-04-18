class LowPass():
    def __init__(self, alpha, p_last=0):
        assert(alpha >= 0 and alpha <= 1)
        self.alpha = alpha
        self.p_last = p_last

    def __call__(self, value):
        # p[n] = alpha*p[n-1] + (1-alpha)*pi[n]
        p_current = alpha * self.p_last + (1 - alpha) * value
        self.p_last = p_current
        return p_current
