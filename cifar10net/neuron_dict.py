import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', )))

# import spikingjelly
from spikingjelly.spikingjelly.activation_based import neuron

NEURON_DICT = {
    "if": neuron.IFNode,
    "sif": neuron.SaturateIFNode,
    "lif": neuron.LIFNode,
    "dlif": neuron.DriveLIFNode,
    "sdlif": neuron.StaticDriveLIFNode,
    "plif": neuron.ParametricLIFNode,
    "pdlif": neuron.ParametricDriveLIFNode,
}