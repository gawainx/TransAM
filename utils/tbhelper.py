from torch.utils.tensorboard import SummaryWriter
from metrics import KGLPMetrics


class TBHelper:
    """TBHelper is Tensorboard Kit"""

    def __init__(self, outdir: str):
        self.writer = SummaryWriter(outdir)

    def log_test_metrics(self, metrics: KGLPMetrics, epoch: int):
        self.writer.add_scalar('T-MRR', metrics.mrr, epoch)
        self.writer.add_scalar('T-HITS@10', metrics.hits_at_10, epoch)
        self.writer.add_scalar('T-HITS@1', metrics.hits_at_1, epoch)
        self.writer.add_scalar('T-HITS@5', metrics.hits_at_5, epoch)

    def log_valid_metrics(self, metrics: KGLPMetrics, epoch: int):
        self.writer.add_scalar('V-MRR', metrics.mrr, epoch)
        self.writer.add_scalar('V-HITS@10', metrics.hits_at_10, epoch)
        self.writer.add_scalar('V-HITS@1', metrics.hits_at_1, epoch)
        self.writer.add_scalar('V-HITS@5', metrics.hits_at_5, epoch)

    def log_loss(self, tag: str, value: float, epoch: int):
        self.writer.add_scalar(tag, value, epoch)
