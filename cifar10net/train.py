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


    def load_model(self, args, num_classes):
        if args.model in parametric_lif_net.__all__:
            model = parametric_lif_net.__dict__[args.model](spiking_neuron=neuron.IFNode,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True)
            functional.set_step_mode(model, step_mode='s')
            if args.cupy:
                functional.set_backend(model, 'cupy', neuron.IFNode)

            return model
        else:
            raise ValueError(f"args.model should be one of {parametric_lif_net.__all__}")
        
        
if __name__ == "__main__":
    # -m torch.distributed.launch --nproc_per_node=2 spikingjelly.activation_based.model.train_imagenet_example
    # python -m spikingjelly.activation_based.model.train_imagenet_example --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_03125 --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90
    trainer = Cifar10NetTrainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)