import torch 
import sys
import os
import time
import warnings
import torch.nn.functional as F
import neuron_dict
import yaml
from torch.utils.tensorboard import SummaryWriter

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spikingjelly')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spikingjelly')))

# import spikingjelly
from spikingjelly.activation_based import surrogate, neuron, functional, monitor
from spikingjelly.activation_based.model import parametric_lif_net, train_classify, tv_ref_classify

# print(spikingjelly.__file__)


class Cifar10NetSpikeCounter(train_classify.Trainer_step):
    def load_data(self, args):
        return super().load_CIFAR10(args)
    

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        parser.add_argument('--config', default='./config/neurons_of_spikecount.py.yaml', type=str, help="neurons")
        parser.add_argument('--neuron_type', default='' ,type=str, help="set the neuron type to monitor")
        parser.add_argument('--tag', default='' ,type=str, help="used for extra mark")
        
        return parser
    
    
    def get_tb_logdir_name(self, args):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        neuron_names = cfg["spiking_neurons"]
        
        tb_dir = 'test/'+ f'{args.model}_spikecount_{neuron_names}_TAG:{args.tag}' 
        return tb_dir


    def evaluate(self, args, model, criterion, data_loader, tb_writer, device, num_classes, log_suffix=""):
        model.eval()
        
        header = f"Test: {log_suffix}"
        
        # tb_dir_2 = self.get_tb_logdir_name(args) + "_2"
        # tb_dir_2 = os.path.join(args.output_dir, tb_dir_2)
        
        # os.makedirs(tb_dir_2, exist_ok=True)
        
        # tb_writer_2 = SummaryWriter(tb_dir_2, purge_step=args.start_epoch)
        

        spike_monitor = monitor.OutputMonitor(model, neuron_dict.NEURON_DICT[args.neuron_type])

        spikes = []
        metric_logger = tv_ref_classify.utils.MetricLogger(delimiter="  ")
        
        num_processed_samples = 0
        start_time = time.time()
        with torch.inference_mode():
            for image, target in metric_logger.log_every(data_loader, -1, header):
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                target_onehot = F.one_hot(target.long(), num_classes).float()

                output_fr = 0.
                
                for t in range(args.time_step):
                    output_fr += model(image)
                    
                    spike = []
                    for layer in spike_monitor.monitored_layers:
                        spike.append((spike_monitor[layer][0] == 1.).sum().item())
                        
                    spikes.append(spike)
                    spike_monitor.clear_recorded_data()
                    
                    
                output_fr = output_fr / args.time_step

                loss_IF = criterion(output_fr, target_onehot)

                acc1_IF, acc5_IF = self.cal_acc1_acc5(output_fr, target_onehot)

                
                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = target.shape[0]
                metric_logger.update(loss_IF=loss_IF.item())
                metric_logger.meters["acc1_IF"].update(acc1_IF.item(), n=batch_size)
                metric_logger.meters["acc5_IF"].update(acc5_IF.item(), n=batch_size)
                num_processed_samples += batch_size
                functional.reset_net(model)
        # gather the stats from all processes

        num_processed_samples = tv_ref_classify.utils.reduce_across_processes(num_processed_samples)
        if (
            hasattr(data_loader.dataset, "__len__")
            and len(data_loader.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
        ):
            # See FIXME above
            warnings.warn(
                f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
                "samples were used for the validation, which might bias the results. "
                "Try adjusting the batch size and / or the world size. "
                "Setting the world size to 1 is always a safe bet."
            )

        metric_logger.synchronize_between_processes() 
        

        count = []
        for i in range(8):
            summary = []
            for s in spikes:
                summary.append(s[i])
        
            count.append(sum(summary)/len(summary))

        for i, c in enumerate(count):
            tb_writer.add_scalar("spikecount", c, i)
        
        print(f"spikecount:{count}")
            
        tb_writer.flush()
        tb_writer.close()



    def load_model(self, args, num_classes):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        neuron_names = cfg["spiking_neurons"]

        spiking_neurons = [neuron_dict.NEURON_DICT[name] for name in neuron_names]

        if args.model in parametric_lif_net.__all__:
            model = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True)
            functional.set_step_mode(model, step_mode='s')

            return model
        else:
            raise ValueError(f"args.model should be one of {parametric_lif_net.__all__}")
        
        
if __name__ == "__main__":
    #nohup python test_spikecount.py --neuron_type if -T 20 --data-path ~/workspace/dataset --model CIFAR10Net --device cuda:1 -j 4 --test-only --resume ./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020/checkpoint_latest.pth  > ./logs/test_spikecount_if.log 2>&1 &
    trainer = Cifar10NetSpikeCounter()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)