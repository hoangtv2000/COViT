from abc import abstractmethod
import torch


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, config, checkpoint_dir, logger,  metric_ftns=None):

        self.config = config
        if self.config.cuda:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")
        self.logger = logger
        self.epochs = self.config.epochs
        self.logger = logger

        self.checkpoint_dir = checkpoint_dir

        self.epochs = config.epochs
        self.log_step = config.log_interval


    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        Args:
            epoch (int): id of epoch
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch, mode, loader):
        """
        Args:
            epoch ():
            mode ():
            loader ():
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """
        Full training logic
        """
        raise NotImplementedError
