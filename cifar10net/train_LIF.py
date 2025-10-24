import torch 
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spikingjelly')))

# import spikingjelly
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import parametric_lif_net, train_classify

# print(spikingjelly.__file__)

class Cifar10NetTrainer(train_classify.Trainer_step):
    def load_data(self, args):
        return super().load_CIFAR10(args)

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        return parser

    
    def get_tb_logdir_name(self, args):
        tb_dir = super().get_tb_logdir_name(args) + '_LIFNode'
        return tb_dir


    def load_model(self, args, num_classes):
        spiking_neurons = [neuron.LIFNode] * 8
    
        if args.model in parametric_lif_net.__all__:
            model = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True)
            functional.set_step_mode(model, step_mode='s')
            if args.cupy:
                functional.set_backend(model, 'cupy', neuron.LIFNode)

            return model
        else:
            raise ValueError(f"args.model should be one of {parametric_lif_net.__all__}")
        
        
if __name__ == "__main__":
    # nohup python train_LIF.py --data-path ~/workspace/dataset --model CIFAR10Net --device cuda:4 -b 50 --epochs 50 -j 4 --lr 1e-3 --time-step 20 --opt adamw > ./logs/train_LIF_log.log 2>&1 &
    trainer = Cifar10NetTrainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)