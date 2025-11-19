import torch 
import sys
import os
import time
import warnings
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spikingjelly')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', )))

# import spikingjelly
from spikingjelly.spikingjelly.activation_based import surrogate, neuron, functional, monitor
from spikingjelly.spikingjelly.activation_based.model import parametric_lif_net, train_classify, tv_ref_classify

# print(spikingjelly.__file__)

class Cifar10NetTester(train_classify.Trainer_step):
    def load_data(self, args):
        return super().load_CIFAR10(args)
    

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        parser.add_argument('--layer', default="conv_fc_2", type=str, help="layer to monitor")
        # parser.add_argument('--rate', default=50, type=int, help="set the weight fault inject rate")
        # parser.add_argument('--train-rate', default=0.07, type=float, help="choose which parameter trained with 'train-rate' saturate")
        parser.add_argument("--resume2", default=None, type=str, help="another path of checkpoint. If set to 'latest', it will try to load the latest checkpoint")
        
        return parser
    
    
    def get_tb_logdir_name(self, args):
        tb_dir = 'test/'+ f'{args.model}' + '_test_IFNode&LIFNode_spikerate_' + f'{args.layer}'
        return tb_dir


    def evaluate(self, args, model, criterion, data_loader, tb_writer, device, num_classes, log_suffix=""):
        model.eval()
        
        header = f"Test: {log_suffix}"
        
        # tb_dir_2 = self.get_tb_logdir_name(args) + "_2"
        # tb_dir_2 = os.path.join(args.output_dir, tb_dir_2)
        
        # os.makedirs(tb_dir_2, exist_ok=True)
        
        # tb_writer_2 = SummaryWriter(tb_dir_2, purge_step=args.start_epoch)
        

        spiking_neurons = [neuron.LIFNode] * 8
        model2 = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True)
        
        checkpoint = torch.load(args.resume2, map_location="cpu", weights_only=False)
        original_dict2 = checkpoint["model"]
        
        model2.load_state_dict(original_dict2)
        model2.to(args.device)
        model2.eval()

        spike_monitor_IF = monitor.OutputMonitor(model, neuron.IFNode)
        spike_monitor_LIF = monitor.OutputMonitor(model2, neuron.LIFNode)
        index = args.layer.rfind('_')  
        monitor_index = args.layer[:index] + '.' + args.layer[index + 1:]

        spikes_IF = []
        spikes_LIF = []
        metric_logger = tv_ref_classify.utils.MetricLogger(delimiter="  ")
        
        num_processed_samples = 0
        start_time = time.time()
        with torch.inference_mode():
            for image, target in metric_logger.log_every(data_loader, -1, header):
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                target_onehot = F.one_hot(target.long(), num_classes).float()

                output_fr_IF = 0.
                output_fr_LIF = 0.
                
                for t in range(args.time_step):
                    output_fr_IF += model(image)
                    output_fr_LIF += model2(image)
                    
                spikes_IF.append((spike_monitor_IF[monitor_index][0] == 1.).sum().item())
                spikes_LIF.append((spike_monitor_LIF[monitor_index][0] == 1.).sum().item())
                spike_monitor_IF.clear_recorded_data()
                spike_monitor_LIF.clear_recorded_data()
                    
                output_fr_IF = output_fr_IF / args.time_step
                output_fr_LIF = output_fr_LIF / args.time_step
                
                loss_IF = criterion(output_fr_IF, target_onehot)
                loss_LIF = criterion(output_fr_LIF, target_onehot)

                acc1_IF, acc5_IF = self.cal_acc1_acc5(output_fr_IF, target_onehot)
                acc1_LIF, acc5_LIF = self.cal_acc1_acc5(output_fr_LIF, target_onehot)
                
                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = target.shape[0]
                metric_logger.update(loss_IF=loss_IF.item())
                metric_logger.meters["acc1_IF"].update(acc1_IF.item(), n=batch_size)
                metric_logger.meters["acc5_IF"].update(acc5_IF.item(), n=batch_size)
                metric_logger.update(loss_LIF=loss_LIF.item())
                metric_logger.meters["acc1_LIF"].update(acc1_LIF.item(), n=batch_size)
                metric_logger.meters["acc5_LIF"].update(acc5_LIF.item(), n=batch_size)
                num_processed_samples += batch_size
                functional.reset_net(model)
                functional.reset_net(model2)
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
        
        
        avg_spike_IF = sum(spikes_IF) / len(spikes_IF)
        avg_spike_LIF = sum(spikes_LIF) / len(spikes_LIF)
        
        print(f"{args.layer} avgspike_IF: {avg_spike_IF} avgspike_LIF: {avg_spike_LIF}")

        
            
        # tb_writer.flush()
        # tb_writer.close()
        # tb_writer_2.flush()
        # tb_writer_2.close()


    def load_model(self, args, num_classes):
        spiking_neurons = [neuron.IFNode] * 8
        
        if args.model in parametric_lif_net.__all__:
            model = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True)
            functional.set_step_mode(model, step_mode='s')
            if args.cupy:
                functional.set_backend(model, 'cupy', neuron.IFNode)

            return model
        else:
            raise ValueError(f"args.model should be one of {parametric_lif_net.__all__}")
        
        
if __name__ == "__main__":
    #nohup python test_IFNode\&LIFNode_spikerate.py --layer conv_fc_2 -T 20 --data-path ~/workspace/dataset --model CIFAR10Net --device cuda:1 -j 4 --test-only --resume ./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020/checkpoint_latest.pth --resume2 ./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020_LIFNode/checkpoint_latest.pth  > ./logs/test_IFNode\&LIFNode_spikerate_conv_2_log.log 2>&1 &
    
    trainer = Cifar10NetTester()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)