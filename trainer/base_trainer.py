from abc import abstractmethod
import torch


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, config,  data_loader, checkpoint_dir, logger,
                 valid_data_loader=None,
                 test_data_loader=None, metric_ftns=None):

        self.config = config
        if self.config.cuda:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")
        self.logger = logger
        self.train_data_loader = data_loader
        self.len_epoch = len(self.train_data_loader)
        self.epochs = self.config.epochs

        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        # setup GPU device if available, move model into configured device
        # self.device, device_ids = self._prepare_device(config['n_gpu'])
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

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids
