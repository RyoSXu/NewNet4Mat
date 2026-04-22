import torch


class Metrics(object):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(Metrics, self).__init__()
        self.epsilon = epsilon

    def MSE(self, pred, gt):
        return torch.mean((pred - gt) ** 2).item()

    def RMSE(self, pred, gt):
        return torch.sqrt(torch.mean((pred - gt) ** 2)).item()

    def MAE(self, pred, gt):
        return torch.mean(torch.abs(pred - gt)).item()


class MetricsRecorder(object):
    def __init__(self, metrics_list, epsilon=1e-7, **kwargs):
        super(MetricsRecorder, self).__init__()
        self.epsilon = epsilon
        self.metrics = Metrics(epsilon=epsilon)
        self.metrics_list = []
        for metric in metrics_list:
            try:
                metric_func = getattr(self.metrics, metric)
                self.metrics_list.append([metric, metric_func])
            except Exception:
                raise NotImplementedError('Invalid metric type: {}.'.format(metric))

    def evaluate_batch(self, data_dict):
        pred = data_dict['pred']
        gt = data_dict['gt']
        losses = {}
        for metric_name, metric_func in self.metrics_list:
            loss = metric_func(pred, gt)
            if isinstance(loss, torch.Tensor):
                for i in range(len(loss)):
                    losses[metric_name + str(i)] = loss[i].item()
            else:
                losses[metric_name] = loss
        return losses
