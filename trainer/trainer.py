import os, json
import torch
import numpy as np
from tqdm import tqdm

from dataloader_n_aug.dataloader import get_balance_train_data, get_val_data
from .base_trainer import BaseTrainer
from utils.util import one_hot_enc
from metrics.metric import MetricTracker, sensitivity, positive_predictive_value
from utils.early_stopping import EarlyStopping


class Trainer(BaseTrainer):
    """
    Trainer and Tracking metrics.
    """

    def __init__(self, config, model, optimizer, name_optimizer, checkpoint_dir, logger,
                 lr_scheduler, start_epoch=None, metric_ftns=None):
        super(Trainer, self).__init__(config, checkpoint_dir, logger,
                                      metric_ftns=metric_ftns)

        if (self.config.cuda):
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")

        self.config = config
        self.logger = logger

        self.start_epoch = 0
        if (config.load_for_training):
            self.start_epoch = start_epoch

        # DataLoader
        self.valid_data_loader = get_val_data(self.config)

        # Epochs
        self.epochs = self.config.epochs-1

        # Step to write info to log
        self.log_step = self.config.log_interval

        # Model
        self.model = model
        self.num_classes = config.dataset.num_classes

        # Optimizer
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.name_optimizer = name_optimizer

        self.early_stopping = EarlyStopping(self.config, self.logger)

        # Loss func
        self.criterion = torch.nn.CrossEntropyLoss()

        self.checkpoint_dir = checkpoint_dir

        self.metric_ftns = ['loss', 'acc']
        self.train_metrics = MetricTracker(*[m for m in self.metric_ftns], mode='train')
        self.valid_metrics = MetricTracker(*[m for m in self.metric_ftns], mode='validation')


        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)


    def _optimizer_coordinating(self, data, output, gr_truth):
        """For the option of the primary or complex optimizer.
        """
        if self.name_optimizer == 'AdamW':
            loss = self.criterion(output, gr_truth)
            loss.backward()
            self.optimizer.step()

        if self.name_optimizer == 'SAM_SGD' or self.name_optimizer == 'SAM_AdamW':
            # first forward-backward pass
            loss = self.criterion(output, gr_truth)
            loss.backward(retain_graph=True)
            self.optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            output_2 = self.model(data)
            loss_2 = self.criterion(output_2, gr_truth)
            loss_2.backward()
            self.optimizer.second_step(zero_grad=True)

        return loss



    def _train_epoch(self, epoch):
        """
        Training and cal statistics for one ep.
        Args:
            epoch (int): current training epoch.
        """

        self.model.train()
        self.confusion_matrix = 0* self.confusion_matrix
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            # data = data.to(self.device)
            # gr_truth = one_hot_enc(target, self.config.dataloader.train.batch_size,\
            #                 self.config.dataset.num_classes).to(self.device)
            # target = torch.tensor(target, dtype=torch.float32).to(self.device)

            data = data.to(self.device)
            gr_truth = torch.tensor(target, dtype=torch.long).to(self.device)
            output = self.model(data)

            if (output.size(0) != gr_truth.size(0)):
                print(f'Failed to load batch:{batch_idx}')
                continue

            # Performing optimizers
            loss = self._optimizer_coordinating(data, output, gr_truth)

            prediction = torch.tensor(torch.argmax(output, dim=1), dtype=torch.float32)
            accuracy = np.sum(prediction.cpu().numpy() == target.cpu().numpy())

            writer_step = (epoch - 1) * self.len_epoch + batch_idx

            self.train_metrics.update(key='loss', value=loss.item(), n=1, writer_step=writer_step)
            self.train_metrics.update(key='acc',
                                          value= accuracy,
                                          n=target.size(0), writer_step=writer_step)

            for tar, pred in zip(target.cpu().view(-1), prediction.cpu().view(-1)):
                self.confusion_matrix[tar.long(), pred.long()] += 1
            self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train')

        self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train', print_summary=True)

        train_loss, train_acc = self.train_metrics.avg('loss'), self.train_metrics.avg('acc')

        return train_loss, train_acc




    def _valid_epoch(self, epoch, mode, loader):
        """
        Calculate acc confusion matrix, sensitivity for valid_set after ending of ep.
        Args:
            epoch (int): current epoch
            mode (string): 'validation' or 'test'
            loader (dataloader):
        Returns: validation loss
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.confusion_matrix = 0* self.confusion_matrix

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                # data = data.to(self.device)
                # gr_truth = one_hot_enc(target, self.config.dataloader.train.batch_size,\
                #                 self.config.dataset.num_classes).to(self.device)
                # target = torch.tensor(target, dtype=torch.float32).to(self.device)
                data = data.to(self.device)
                gr_truth = torch.tensor(target, dtype=torch.long).to(self.device)
                output = self.model(data)

                if  (output.size(0) != gr_truth.size(0)):
                    print(f'Failed to load batch:{batch_idx}')
                    continue

                loss = self.criterion(output, gr_truth)

                prediction = torch.tensor(torch.argmax(output, dim=1), dtype=torch.float32)
                accuracy = np.sum(prediction.cpu().numpy() == target.cpu().numpy())

                writer_step = (epoch - 1) * len(loader) + batch_idx
                self.valid_metrics.update(key='loss', value=loss.item(), n=1, writer_step=writer_step)
                self.valid_metrics.update(key='acc',
                                          value = accuracy,
                                          n=target.size(0), writer_step=writer_step)

                for tar, pred in zip(target.cpu().view(-1), prediction.cpu().view(-1)):
                    self.confusion_matrix[tar.long(), pred.long()] += 1

        self._progress(batch_idx, epoch, metrics=self.valid_metrics, mode=mode, print_summary=True)

        s = sensitivity(self.confusion_matrix.numpy())
        ppv = positive_predictive_value(self.confusion_matrix.numpy())
        print(f" sensitivity: {s} ,positive_predictive_value: {ppv}")

        val_loss, val_acc = self.valid_metrics.avg('loss'), self.valid_metrics.avg('acc')

        return val_loss, val_acc




    def train(self):
        """
        Train the model
        """
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc':[]}
        for epoch in range(self.start_epoch, self.epochs):
            torch.manual_seed(self.config.seed)
            torch.autograd.set_detect_anomaly(True)

            self.train_data_loader = get_balance_train_data(self.config) # randomly sample data to balance
            self.len_epoch = self.config.dataloader.train.batch_size * len(self.train_data_loader)

            train_loss, train_acc = self._train_epoch(epoch)

            self.logger.info(f"{'!' * 10}    VALIDATION   {'!' * 10}")
            val_loss, val_acc = self._valid_epoch(epoch, 'validation', self.valid_data_loader)

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            self.early_stopping(self.checkpoint_dir, model=self.model, optimizer=self.optimizer,\
                    scheduler=self.lr_scheduler, val_loss=val_loss, epoch=epoch, name='model_best')

            self.lr_scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if self.early_stopping.early_stop == True:
                self.logger.info(f'Training session ends at epoch: {epoch}')
                break

            writer = os.path.join(self.checkpoint_dir, 'writer.txt')
            f = open(writer, "w")
            f.write(json.dumps(history))



    def _progress(self, batch_idx, epoch, metrics, mode='', print_summary=False):
        metrics_string = metrics.calc_all_metrics()
        iter = batch_idx * self.config.dataloader.train.batch_size
        if (iter % self.log_step == 0 and iter !=0):

            if metrics_string == None:
                self.logger.warning(f" No metrics")
            else:
                self.logger.info(
                    f"{mode} Epoch: [{epoch:2d}/{self.epochs:2d}]\t Sample [{batch_idx * self.config.dataloader.train.batch_size:5d}/{self.len_epoch:5d}]\t {metrics_string}")
        elif print_summary:
            self.logger.info(
                f'{mode} summary  Epoch: [{epoch}/{self.epochs}]\t {metrics_string}')
