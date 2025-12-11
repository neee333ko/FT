import torch 
import sys
import os
import yaml
import neuron_dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spikingjelly')))

# import spikingjelly
from spikingjelly.spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.spikingjelly.activation_based.model import parametric_lif_net, train_classify

# print(spikingjelly.__file__)


class Cifar10NetTrainer(train_classify.Trainer_step):
    def load_data(self, args):
        return super().load_CIFAR10(args)

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        parser.add_argument('--config', default='./config/train.yaml', type=str, help="set the neruons")
        return parser

    def get_tb_logdir_name(self, args):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        neuron_names = cfg["spiking_neurons"]
        
        dir = super().get_tb_logdir_name(args) + f'_{neuron_names}'
        
        return dir


    def load_model(self, args, num_classes):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        neuron_names = cfg["spiking_neurons"]

        spiking_neurons = [neuron_dict.NEURON_DICT[name] for name in neuron_names]
    
        if args.model in parametric_lif_net.__all__:
            model = parametric_lif_net.__dict__[args.model](spiking_neurons=spiking_neurons,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True, scale = 0.4)
            functional.set_step_mode(model, step_mode='s')

            return model
        else:
            raise ValueError(f"args.model should be one of {parametric_lif_net.__all__}")
        
        
if __name__ == "__main__":
    # nohup python train.py --data-path ~/workspace/dataset --model CIFAR10Net --device cuda:5 -b 50 --epochs 50 -j 4 --lr 1e-3 --time-step 20 --opt adamw > ./logs/train_log.log 2>&1 &
    trainer = Cifar10NetTrainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args) 