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
        parser.add_argument('--rate', default=100, type=int, help="set the weight fault inject rate")
        # parser.add_argument('--train-rate', default=0.07, type=float, help="choose which parameter trained with 'train-rate' saturate")
        parser.add_argument("--resume2", default=None, type=str, help="another path of checkpoint. If set to 'latest', it will try to load the latest checkpoint")
        parser.add_argument("--resume3", default=None, type=str, help="another path of checkpoint. If set to 'latest', it will try to load the latest checkpoint")
        parser.add_argument("--resume4", default=None, type=str, help="another path of checkpoint. If set to 'latest', it will try to load the latest checkpoint")
        
        return parser
    
    
    def get_tb_logdir_name(self, args):
        tb_dir = 'test/'+ f'{args.model}' + '_test_saturate_conv_2_rate:' + f'{args.rate}' + '_if_lif_pdlif1_pdlif2'
        return tb_dir


    def evaluate(self, args, model, criterion, data_loader, tb_writer, device, num_classes, log_suffix=""):
        model.eval()
        
        header = f"Test: {log_suffix}"
        
        original_dict = model.state_dict()
        model.to(args.device)
        model.eval()
        
        # makedir
        tb_dir_2 = self.get_tb_logdir_name(args) + "_2"
        tb_dir_2 = os.path.join(args.output_dir, tb_dir_2)
        os.makedirs(tb_dir_2, exist_ok=True)
        tb_writer_2 = SummaryWriter(tb_dir_2, purge_step=args.start_epoch)
        
        tb_dir_3 = self.get_tb_logdir_name(args) + "_3"
        tb_dir_3 = os.path.join(args.output_dir, tb_dir_3)
        os.makedirs(tb_dir_3, exist_ok=True)
        tb_writer_3 = SummaryWriter(tb_dir_3, purge_step=args.start_epoch)
        
        tb_dir_4 = self.get_tb_logdir_name(args) + "_4"
        tb_dir_4 = os.path.join(args.output_dir, tb_dir_4)
        os.makedirs(tb_dir_4, exist_ok=True)
        tb_writer_4 = SummaryWriter(tb_dir_4, purge_step=args.start_epoch)


        # checkpoint  
        original_dict2  = torch.load(args.resume2, map_location="cpu", weights_only=False)["model"]
        original_dict3  = torch.load(args.resume3, map_location="cpu", weights_only=False)["model"]
        original_dict4  = torch.load(args.resume4, map_location="cpu", weights_only=False)["model"]


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
            
            metrics_acc1_if = []
            metrics_acc5_if = []
            metrics_loss_if = []
            
            metrics_acc1_lif = []
            metrics_acc5_lif = []
            metrics_loss_lif = []
            
            metrics_acc1_pdlif1 = []
            metrics_acc5_pdlif1 = []
            metrics_loss_pdlif1 = []
            
            metrics_acc1_pdlif2 = []
            metrics_acc5_pdlif2 = []
            metrics_loss_pdlif2 = []
            
            while round < 5:
                metric_logger = tv_ref_classify.utils.MetricLogger(delimiter="  ")
                
                model = self.load_model(args,10)
                model.to(args.device)
                
                model.load_state_dict(original_dict)
                
                # 先跑一个数据
                # if
                with torch.inference_mode():
                    for image, target in metric_logger.log_every(data_loader, -1, header):
                        image = image.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        target_onehot = F.one_hot(target.long(), num_classes).float()

                        output_fr = 0.
                        
                        for t in range(args.time_step):
                            output_fr += model(image)
                            
                        break
            
                map = inject.inject_saturate_by_map(model,["conv_fc.2"],[rate/100],None)
                
                functional.reset_net(model)
            
                # lif
                spiking_neurons = [neuron.LIFNode] * 8
        
                model2 = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True)
                model2.load_state_dict(original_dict2)
                model2.to(args.device)
                model2.eval()
            
                with torch.inference_mode():
                    for image, target in metric_logger.log_every(data_loader, -1, header):
                        image = image.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        target_onehot = F.one_hot(target.long(), num_classes).float()

                        output_fr = 0.
                        
                        for t in range(args.time_step):
                            output_fr += model2(image)
                            
                        break
                    
                inject.inject_saturate_by_map(model2,["conv_fc.2"],None,map)
                functional.reset_net(model2)
                
                
                # pdlif1
                spiking_neurons = [neuron.ParametricDriveLIFNode] * 8
        
                model3 = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True)
                model3.load_state_dict(original_dict3)
                model3.to(args.device)
                model3.eval()
            
                with torch.inference_mode():
                    for image, target in metric_logger.log_every(data_loader, -1, header):
                        image = image.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        target_onehot = F.one_hot(target.long(), num_classes).float()

                        output_fr = 0.
                        
                        for t in range(args.time_step):
                            output_fr += model3(image)
                            
                        break
                    
                inject.inject_saturate_by_map(model3,["conv_fc.2"],None,map)
                functional.reset_net(model3)
                
                
                # pdlif2
                spiking_neurons = [neuron.ParametricDriveLIFNode] * 8
        
                model4 = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True)
                model4.load_state_dict(original_dict4)
                model4.to(args.device)
                model4.eval()
            
                with torch.inference_mode():
                    for image, target in metric_logger.log_every(data_loader, -1, header):
                        image = image.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        target_onehot = F.one_hot(target.long(), num_classes).float()

                        output_fr = 0.
                        
                        for t in range(args.time_step):
                            output_fr += model4(image)
                            
                        break
                    
                inject.inject_saturate_by_map(model4,["conv_fc.2"],None,map)
                functional.reset_net(model4)
            
            
                # test
                with torch.inference_mode():
                    for image, target in metric_logger.log_every(data_loader, -1, header):
                        image = image.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        target_onehot = F.one_hot(target.long(), num_classes).float()

                        output_fr_if = 0.
                        output_fr_lif = 0.
                        output_fr_pdlif1 = 0.
                        output_fr_pdlif2 = 0.
                        
                        for t in range(args.time_step):
                            output_fr_if += model(image)
                            output_fr_lif += model2(image)
                            output_fr_pdlif1 += model3(image)
                            output_fr_pdlif2 += model4(image)
                            
        
                        output_fr_if = output_fr_if / args.time_step
                        loss_if = criterion(output_fr_if, target_onehot)
                        
                        output_fr_lif = output_fr_lif / args.time_step
                        loss_lif = criterion(output_fr_lif, target_onehot)
                        
                        output_fr_pdlif1 = output_fr_pdlif1 / args.time_step
                        loss_pdlif1 = criterion(output_fr_pdlif1, target_onehot)
                        
                        output_fr_pdlif2 = output_fr_pdlif2 / args.time_step
                        loss_pdlif2 = criterion(output_fr_pdlif2, target_onehot)
                        
                        
                        acc1_if, acc5_if = self.cal_acc1_acc5(output_fr_if, target_onehot)
                        acc1_lif, acc5_lif = self.cal_acc1_acc5(output_fr_lif, target_onehot)
                        acc1_pdlif1, acc5_pdlif1 = self.cal_acc1_acc5(output_fr_pdlif1, target_onehot)
                        acc1_pdlif2, acc5_pdlif2 = self.cal_acc1_acc5(output_fr_pdlif2, target_onehot)
                        
                        # FIXME need to take into account that the datasets
                        # could have been padded in distributed setup
                        batch_size = target.shape[0]
                        metric_logger.update(loss_if=loss_if.item())
                        metric_logger.meters["acc1_if"].update(acc1_if.item(), n=batch_size)
                        metric_logger.meters["acc5_if"].update(acc5_if.item(), n=batch_size)
                        
                        metric_logger.update(loss_lif=loss_lif.item())
                        metric_logger.meters["acc1_lif"].update(acc1_lif.item(), n=batch_size)
                        metric_logger.meters["acc5_lif"].update(acc5_lif.item(), n=batch_size)
                        
                        metric_logger.update(loss_pdlif1=loss_pdlif1.item())
                        metric_logger.meters["acc1_pdlif1"].update(acc1_pdlif1.item(), n=batch_size)
                        metric_logger.meters["acc5_pdlif1"].update(acc5_pdlif1.item(), n=batch_size)
                        
                        metric_logger.update(loss_pdlif2=loss_pdlif2.item())
                        metric_logger.meters["acc1_pdlif2"].update(acc1_pdlif2.item(), n=batch_size)
                        metric_logger.meters["acc5_pdlif2"].update(acc5_pdlif2.item(), n=batch_size)
                        
                        functional.reset_net(model)
                        functional.reset_net(model2)
                        functional.reset_net(model3)
                        functional.reset_net(model4)
                # gather the stats from all processes

                # num_processed_samples = tv_ref_classify.utils.reduce_across_processes(num_processed_samples)
                # if (
                #     hasattr(data_loader.dataset, "__len__")
                #     and len(data_loader.dataset) != num_processed_samples
                #     and torch.distributed.get_rank() == 0
                # ):
                #     # See FIXME above
                #     warnings.warn(
                #         f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
                #         "samples were used for the validation, which might bias the results. "
                #         "Try adjusting the batch size and / or the world size. "
                #         "Setting the world size to 1 is always a safe bet."
                #     )

                # metric_logger.synchronize_between_processes()

                test_loss_if, test_acc1_if, test_acc5_if = metric_logger.loss_if.global_avg, metric_logger.acc1_if.global_avg, metric_logger.acc5_if.global_avg
                test_loss_lif, test_acc1_lif, test_acc5_lif = metric_logger.loss_lif.global_avg, metric_logger.acc1_lif.global_avg, metric_logger.acc5_lif.global_avg
                test_loss_pdlif1, test_acc1_pdlif1, test_acc5_pdlif1 = metric_logger.loss_pdlif1.global_avg, metric_logger.acc1_pdlif1.global_avg, metric_logger.acc5_pdlif1.global_avg
                test_loss_pdlif2, test_acc1_pdlif2, test_acc5_pdlif2 = metric_logger.loss_pdlif2.global_avg, metric_logger.acc1_pdlif2.global_avg, metric_logger.acc5_pdlif2.global_avg
                
                
                metrics_loss_if.append(test_loss_if)
                metrics_acc1_if.append(test_acc1_if)
                metrics_acc5_if.append(test_acc5_if)

                metrics_loss_lif.append(test_loss_lif)
                metrics_acc1_lif.append(test_acc1_lif)
                metrics_acc5_lif.append(test_acc5_lif)
                
                metrics_loss_pdlif1.append(test_loss_pdlif1)
                metrics_acc1_pdlif1.append(test_acc1_pdlif1)
                metrics_acc5_pdlif1.append(test_acc5_pdlif1)
                
                metrics_loss_pdlif2.append(test_loss_pdlif2)
                metrics_acc1_pdlif2.append(test_acc1_pdlif2)
                metrics_acc5_pdlif2.append(test_acc5_pdlif2)
                
                round += 1


            tb_writer.add_scalar("loss", sum(metrics_loss_if)/len(metrics_loss_if), rate)
            tb_writer.add_scalar("acc1", sum(metrics_acc1_if)/len(metrics_acc1_if), rate)
            tb_writer.add_scalar("acc5", sum(metrics_acc5_if)/len(metrics_acc5_if), rate)

            tb_writer_2.add_scalar("loss", sum(metrics_loss_lif)/len(metrics_loss_lif), rate)
            tb_writer_2.add_scalar("acc1", sum(metrics_acc1_lif)/len(metrics_acc1_lif), rate)
            tb_writer_2.add_scalar("acc5", sum(metrics_acc5_lif)/len(metrics_acc5_lif), rate)
            
            tb_writer_3.add_scalar("loss", sum(metrics_loss_pdlif1)/len(metrics_loss_pdlif1), rate)
            tb_writer_3.add_scalar("acc1", sum(metrics_acc1_pdlif1)/len(metrics_acc1_pdlif1), rate)
            tb_writer_3.add_scalar("acc5", sum(metrics_acc5_pdlif1)/len(metrics_acc5_pdlif1), rate)

            tb_writer_4.add_scalar("loss", sum(metrics_loss_pdlif2)/len(metrics_loss_pdlif2), rate)
            tb_writer_4.add_scalar("acc1", sum(metrics_acc1_pdlif2)/len(metrics_acc1_pdlif2), rate)
            tb_writer_4.add_scalar("acc5", sum(metrics_acc5_pdlif2)/len(metrics_acc5_pdlif2), rate)
            
            print(
                f"Test:saturate_rate:{rate/100}, "
                "IFNode: "
                f"test_acc1={sum(metrics_acc1_if)/len(metrics_acc1_if):.3f}, "
                f"test_acc5={sum(metrics_acc5_if)/len(metrics_acc5_if):.3f}, "
                f"test_loss={sum(metrics_loss_if)/len(metrics_loss_if):.6f}, "
                "LIFNode: "
                f"test_acc1={sum(metrics_acc1_lif)/len(metrics_acc1_lif):.3f}, "
                f"test_acc5={sum(metrics_acc5_lif)/len(metrics_acc5_lif):.3f}, "
                f"test_loss={sum(metrics_loss_lif)/len(metrics_loss_lif):.6f}, "
                "PDLIFNode1: "
                f"test_acc1={sum(metrics_acc1_pdlif1)/len(metrics_acc1_pdlif1):.3f}, "
                f"test_acc5={sum(metrics_acc5_pdlif1)/len(metrics_acc5_pdlif1):.3f}, "
                f"test_loss={sum(metrics_loss_pdlif1)/len(metrics_loss_pdlif1):.6f}, "
                "PDLIFNode2: "
                f"test_acc1={sum(metrics_acc1_pdlif2)/len(metrics_acc1_pdlif2):.3f}, "
                f"test_acc5={sum(metrics_acc5_pdlif2)/len(metrics_acc5_pdlif2):.3f}, "
                f"test_loss={sum(metrics_loss_pdlif2)/len(metrics_loss_pdlif2):.6f}, "
            )

              
            rate += 1  
            
        tb_writer.flush()
        tb_writer.close()
        tb_writer_2.flush()
        tb_writer_2.close()
        tb_writer_3.flush()
        tb_writer_3.close()
        tb_writer_4.flush()
        tb_writer_4.close()




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
    #nohup python test_saturate_conv_2_if_lif_pdlif1_pdlif2.py -T 20 --data-path ~/workspace/dataset --model CIFAR10Net --device cuda:1 -j 4 --test-only --resume ./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020/checkpoint_latest.pth --resume2 ./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020_LIFNode/checkpoint_latest.pth --resume3 ./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020_para_drive_LIFNode/checkpoint_latest.pth --resume4 "./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020_['pdlif', 'pdlif', 'pdlif', 'pdlif', 'pdlif', 'pdlif', 'pdlif', 'pdlif']/checkpoint_latest.pth"  > ./logs/test_saturate_conv_2_if_lif_pdlif1_pdlif2.log 2>&1 &
    
    trainer = Cifar10NetTester()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)