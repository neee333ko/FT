import torch 
import sys
import os
import time
import torch.nn.functional as F
import yaml
import neuron_dict
from torch import nn


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', )))

# import spikingjelly
from spikingjelly.spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.spikingjelly.activation_based.model import parametric_lif_net, train_classify, tv_ref_classify

# print(spikingjelly.__file__)


class Cifar10NetSaturateTrainer(train_classify.Trainer_step):
    def load_data(self, args):
        return super().load_CIFAR10(args)

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        parser.add_argument('--p_force', default=0.0, type=float, help="set the rate of the saturate")
        parser.add_argument('--config', default='./config/saturate_train.yaml', type=str, help="set the neruons")
        return parser

    def get_tb_logdir_name(self, args):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        neuron_names = cfg["spiking_neurons"]
        dir = super().get_tb_logdir_name(args) + f'_saturate_training_{args.p_force}_{neuron_names}'
        
        return dir
        

    def train_one_epoch(self, model, criterion, optimizer, data_loader, device, epoch, args, num_classes, model_ema=None, scaler=None):
        model.train()
        metric_logger = tv_ref_classify.utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", tv_ref_classify.utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", tv_ref_classify.utils.SmoothedValue(window_size=10, fmt="{value}"))
        

        header = f"Epoch: [{epoch}]"
        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, -1, header)):
            start_time = time.time()
            image, target = image.to(device), target.to(device)
            target_onehot = F.one_hot(target.long(), num_classes).float()

            output_fr = 0.
            
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                for t in range(args.time_step):
                    output_fr += model(image)

                output_fr = output_fr / args.time_step
                loss = criterion(output_fr, target_onehot)
                
            functional.mask_on(model)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
            functional.reset_net(model)

            if model_ema and i % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)

            acc1, acc5 = self.cal_acc1_acc5(output_fr, target_onehot)
            batch_size = target.shape[0]
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        train_loss, train_acc1, train_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
        print(f'Train: train_acc1={train_acc1:.3f}, train_acc5={train_acc5:.3f}, train_loss={train_loss:.6f}, samples/s={metric_logger.meters["img/s"]}')
        return train_loss, train_acc1, train_acc5



    def load_model(self, args, num_classes):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        neuron_names = cfg["spiking_neurons"]

        spiking_neurons = [neuron_dict.NEURON_DICT[name] for name in neuron_names]
    
        if args.model in parametric_lif_net.__all__:
            model = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True, p_force = args.p_force)
            functional.set_step_mode(model, step_mode='s')
            # if args.cupy:
            #     functional.set_backend(model, 'cupy', neuron.IFNode)

            return model
        else:
            raise ValueError(f"args.model should be one of {parametric_lif_net.__all__}")
        
        
if __name__ == "__main__":
    # nohup python train_saturate_conv_2.py --p-force 0.03 --data-path ~/workspace/dataset --model CIFAR10Net --device cuda:5 -b 50 --epochs 50 -j 4 --lr 1e-3 --time-step 20 --opt adamw > ./logs/train_saturate_conv_2_0.03_log.log 2>&1 &
    
    trainer = Cifar10NetSaturateTrainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)