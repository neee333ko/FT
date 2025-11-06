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
from spikingjelly.spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.spikingjelly.activation_based.model import parametric_lif_net, train_classify, tv_ref_classify
from FTtool import inject

# print(spikingjelly.__file__)

class Cifar10NetTester(train_classify.Trainer_step):
    def load_data(self, args):
        return super().load_CIFAR10(args)
    

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        parser.add_argument('--rate', default=50, type=int, help="set the weight fault inject rate")
        # parser.add_argument('--train-rate', default=0.07, type=float, help="choose which parameter trained with 'train-rate' saturate")
        parser.add_argument("--resume2", default=None, type=str, help="another path of checkpoint. If set to 'latest', it will try to load the latest checkpoint")
        
        return parser
    
    
    def get_tb_logdir_name(self, args):
        tb_dir = 'test/'+ f'{args.model}' + '_test_BF_' + 'conv_2_' + f'{args.rate}' + '_LIFNode'
        return tb_dir


    def evaluate(self, args, model, criterion, data_loader, tb_writer, device, num_classes, log_suffix=""):
        model.eval()
        
        header = f"Test: {log_suffix}"
        
        tb_dir_2 = self.get_tb_logdir_name(args) + "_2"
        tb_dir_2 = os.path.join(args.output_dir, tb_dir_2)
        
        os.makedirs(tb_dir_2, exist_ok=True)
        
        tb_writer_2 = SummaryWriter(tb_dir_2, purge_step=args.start_epoch)
        
        original_dict = model.state_dict()

        spiking_neurons = [neuron.LIFNode] * 8
        
        model2 = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True)
        
        checkpoint = torch.load(args.resume2, map_location="cpu", weights_only=False)
        original_dict2 = checkpoint["model"]
        
        model2.load_state_dict(original_dict2)
        model2.to(args.device)
        model2.eval()

        rate = 0
        
        while rate <= args.rate:
            round = 0
            metrics_acc1 = []
            metrics_acc5 = []
            metrics_loss = []
            metrics_velo = []
            
            metrics_acc1_LIF = []
            metrics_acc5_LIF = []
            metrics_loss_LIF = []
            metrics_velo_LIF = []
            while round < 5:
                metric_logger = tv_ref_classify.utils.MetricLogger(delimiter="  ")
                
                model = self.load_model(args,10)
                model.to(args.device)
                
                model.load_state_dict(original_dict)
                
                convs, state_dict = inject.get_convs(model)
                
                # only inject conv_2
                fixed_state_dict, maps = inject.inject_weight(convs[:1], state_dict, rate/1000, None)
                model.load_state_dict(fixed_state_dict)
                
                # raw test
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
                                    
                        output_fr = output_fr / args.time_step
                        loss = criterion(output_fr, target_onehot)

                        acc1, acc5 = self.cal_acc1_acc5(output_fr, target_onehot)
                        # FIXME need to take into account that the datasets
                        # could have been padded in distributed setup
                        batch_size = target.shape[0]
                        metric_logger.update(loss=loss.item())
                        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
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

                test_loss, test_acc1, test_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
                
                metrics_loss.append(test_loss)
                metrics_acc1.append(test_acc1)
                metrics_acc5.append(test_acc5)
                metrics_velo.append(num_processed_samples/(time.time() - start_time))
                
                # LIFNode test
                model2.load_state_dict(original_dict2)    
                convs, state_dict = inject.get_convs(model2)
                
                # only inject conv_2
                fixed_state_dict, maps = inject.inject_weight(convs[:1], state_dict, rate/1000, maps)
                model2.load_state_dict(fixed_state_dict)
            
                num_processed_samples = 0
                start_time = time.time()
                with torch.inference_mode():
                    for image, target in metric_logger.log_every(data_loader, -1, header):
                        image = image.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        target_onehot = F.one_hot(target.long(), num_classes).float()

                        output_fr = 0.
                        
                        for t in range(args.time_step):
                            output_fr += model2(image)
                                    
                        output_fr = output_fr / args.time_step
                        loss = criterion(output_fr, target_onehot)

                        acc1, acc5 = self.cal_acc1_acc5(output_fr, target_onehot)
                        # FIXME need to take into account that the datasets
                        # could have been padded in distributed setup
                        batch_size = target.shape[0]
                        metric_logger.update(loss_LIF=loss.item())
                        metric_logger.meters["acc1_LIF"].update(acc1.item(), n=batch_size)
                        metric_logger.meters["acc5_LIF"].update(acc5.item(), n=batch_size)
                        num_processed_samples += batch_size
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

                test_loss, test_acc1, test_acc5 = metric_logger.loss_LIF.global_avg, metric_logger.acc1_LIF.global_avg, metric_logger.acc5_LIF.global_avg
                
                metrics_loss_LIF.append(test_loss)
                metrics_acc1_LIF.append(test_acc1)
                metrics_acc5_LIF.append(test_acc5)
                metrics_velo_LIF.append(num_processed_samples/(time.time() - start_time))
        
                round += 1

            test_acc1 = sum(metrics_acc1) / len(metrics_acc1)
            test_acc5 = sum(metrics_acc5) / len(metrics_acc5)
            test_loss = sum(metrics_loss) / len(metrics_loss)
            test_velo = sum(metrics_velo) / len(metrics_velo)
    
            tb_writer.add_scalar("loss", test_loss, rate)
            tb_writer.add_scalar("acc1", test_acc1, rate)
            tb_writer.add_scalar("acc5", test_acc5, rate)
            tb_writer.add_scalar("velo", test_velo, rate)
            
            test_acc1_LIF = sum(metrics_acc1_LIF) / len(metrics_acc1_LIF)
            test_acc5_LIF = sum(metrics_acc5_LIF) / len(metrics_acc5_LIF)
            test_loss_LIF = sum(metrics_loss_LIF) / len(metrics_loss_LIF)
            test_velo_LIF = sum(metrics_velo_LIF) / len(metrics_velo_LIF)
            
            tb_writer_2.add_scalar("loss", test_loss_LIF, rate)
            tb_writer_2.add_scalar("acc1", test_acc1_LIF, rate)
            tb_writer_2.add_scalar("acc5", test_acc5_LIF, rate)
            tb_writer_2.add_scalar("velo", test_velo_LIF, rate)
            
            print(
                f"Test:BF_rate:{rate/1000}, "
                "IFNode: "
                f"test_acc1={test_acc1:.3f}, "
                f"test_acc5={test_acc5:.3f}, "
                f"test_loss={test_loss:.6f}, "
                f"samples/s={test_velo:.3f}, "
                "LIFNode: "
                f"test_acc1={test_acc1_LIF:.3f}, "
                f"test_acc5={test_acc5_LIF:.3f}, "
                f"test_loss={test_loss_LIF:.6f}, "
                f"samples/s={test_velo_LIF:.3f}, "
            )

              
            rate += 1  
            
        tb_writer.flush()
        tb_writer.close()
        tb_writer_2.flush()
        tb_writer_2.close()




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
    #nohup python test_BF_LIFNode_conv_2.py -T 20 --data-path ~/workspace/dataset --model CIFAR10Net --device cuda:0 -j 4 --test-only --resume ./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020/checkpoint_latest.pth --resume2 ./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020_LIFNode/checkpoint_latest.pth  > ./logs/test_BF_LIFNode_conv_2_log.log 2>&1 &
    
    trainer = Cifar10NetTester()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)