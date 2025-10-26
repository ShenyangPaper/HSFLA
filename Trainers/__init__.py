from .poolingmil import PoolingMILTrainer
from .mil_rnn import MILRNNTrainer
from.mix_milrnn import MILRNN
from .abmil import  Origin_ABMIL, MixABMIL, MixABMIL_Inf
from .clam import CLAMTrainer
from .Supvision import Supvision
#from.transmil import TransMILTrainer, BaseTransMILTrainer, MixTransMILTrainer

trainers_fn = {
    'mean_pooling_mil': PoolingMILTrainer, 'max_pooling_mil': PoolingMILTrainer, 
    'mil_rnn': MILRNNTrainer,
    'clam_sb': CLAMTrainer, 'clam_mb': CLAMTrainer, 
    'origin_abmil': Origin_ABMIL, 
    'mix_rnn': MILRNN,
    'supvision': Supvision,
    'mix_abmil': MixABMIL
    #'stable_transmil': TransMILTrainer,
    #'transmil': BaseTransMILTrainer,
    #'mixtransmil': MixTransMILTrainer
}
