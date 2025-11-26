import torch 
import sys
import os
import time
import warnings
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import InterpolationMode

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spikingjelly')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', )))

# import spikingjelly
from spikingjelly.spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.spikingjelly.activation_based.model import parametric_lif_net, train_classify, tv_ref_classify
from FTtool import noise

# print(spikingjelly.__file__)

class Cifar10NetTester(train_classify.Trainer_step):
    def load_CIFAR10(self, args):
        # Data loading code
        print("Loading data")
        val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
       
        
        interpolation = InterpolationMode(args.interpolation)

        print("Loading training data")
        st = time.time()
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        
        if args.present_transform:
            transform = tv_ref_classify.presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
            )
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),  # 将图像转换为张量
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
            ]) # 归一化
        
        dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=True,
            transform=transform,
            download=True
        )

        print("Took", time.time() - st)

        print("Loading validation data")

        dataset_test = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=False,
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            noise.AddGaussianNoise(mean=0.0, std=0.1),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        )

        print("Creating data loaders")
        loader_g = torch.Generator()
        loader_g.manual_seed(args.seed)

        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = tv_ref_classify.sampler.RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)


        return dataset, dataset_test, train_sampler, test_sampler
    
    
    def load_data(self, args):
        return self.load_CIFAR10(args)
    

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        # parser.add_argument('--rate', default=50, type=int, help="set the saturate rate")
        # parser.add_argument('--train-rate', default=0.07, type=float, help="choose which parameter trained with 'train-rate' saturate")
        parser.add_argument("--resume2", default=None, type=str, help="another path of checkpoint. If set to 'latest', it will try to load the latest checkpoint")
        
        return parser
    
    
    def get_tb_logdir_name(self, args):
        tb_dir = 'test/'+ f'{args.model}' + '_test_Gaussian_noise_' + 'mean0_std1_ParametricDriveLIFNode' 
        return tb_dir


    def evaluate(self, args, model, criterion, data_loader, tb_writer, device, num_classes, log_suffix=""):
        model.eval()
        
        header = f"Test: {log_suffix}"
        
        
        spiking_neurons = [neuron.ParametricDriveLIFNode] * 8
        
        model2 = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True)
        
        checkpoint = torch.load(args.resume2, map_location="cpu", weights_only=False)
        original_dict2 = checkpoint["model"]
        
        model2.load_state_dict(original_dict2)
        model2.to(args.device)
        
        metric_logger = tv_ref_classify.utils.MetricLogger(delimiter="  ")
                
        num_processed_samples = 0
        processed_time = 0.
        processed_time2 = 0.
        idx = 0
        with torch.inference_mode():
            for image, target in metric_logger.log_every(data_loader, -1, header):
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                target_onehot = F.one_hot(target.long(), num_classes).float()

                output_fr = 0.
                output_fr2 = 0.
                
                start_time = time.time()
                for t in range(args.time_step):
                    output_fr += model(image)
                processed_time += time.time() - start_time
                    
                start_time = time.time()
                for t in range(args.time_step):
                    output_fr2 += model2(image)
                processed_time2 += time.time() - start_time
                            
                output_fr = output_fr / args.time_step
                output_fr2 = output_fr2 / args.time_step
                
                loss = criterion(output_fr, target_onehot)
                loss2 = criterion(output_fr2, target_onehot)

                acc1, acc5 = self.cal_acc1_acc5(output_fr, target_onehot)
                acc1_2, acc5_2 = self.cal_acc1_acc5(output_fr2, target_onehot)
                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = target.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                
                metric_logger.update(loss2=loss2.item())
                metric_logger.meters["acc1_2"].update(acc1_2.item(), n=batch_size)
                metric_logger.meters["acc5_2"].update(acc5_2.item(), n=batch_size)
                
                num_processed_samples += batch_size
                

                
                processed_time = 0.
                processed_time2 = 0.
                idx += 1
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

        test_loss, test_acc1, test_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
        test_loss2, test_acc1_2, test_acc5_2 = metric_logger.loss2.global_avg, metric_logger.acc1_2.global_avg, metric_logger.acc5_2.global_avg
    
        
        print(
            f"Test:GaussianNoise_mean0.0_std0.1, "
            "IFNode: "
            f"test_acc1={test_acc1:.3f}, "
            f"test_acc5={test_acc5:.3f}, "
            f"test_loss={test_loss:.6f}, "
            "ParametricDriveLIFNode: "
            f"test_acc1={test_acc1_2:.3f}, "
            f"test_acc5={test_acc5_2:.3f}, "
            f"test_loss={test_loss2:.6f}, "
        )

             
            




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
    #nohup python test_noise.py -T 20 --data-path ~/workspace/dataset --model CIFAR10Net --device cuda:0 -j 4 --test-only --resume ./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020/checkpoint_latest.pth --resume2 ./logs/pt/CIFAR10Net_t20_b50_e50_adamw_lr0.001_wd0.0_ls0.1_ma0.0_ca0.0_sbn0_ra0_re0.0_aaugNone_size176_232_224_seed2020_para_drive_LIFNode/checkpoint_latest.pth  > ./logs/test_noise_parametric_drive_LIFNode.log 2>&1 &
    
    trainer = Cifar10NetTester()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)