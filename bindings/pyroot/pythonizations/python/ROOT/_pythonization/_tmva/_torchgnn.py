"""
Helper functions for Python TorchGNN.

Author: Stefan van Berkum
Date: 08/2023
"""

from .. import pythonization
from cppyy.gbl.std import vector, map


class RModel_TorchGNN():
    def extractParameters(self, model):
        """Extract the parameters from a PyTorch model.

        In order for this to work, the parameterized module names in ROOT should
        be the same as those in the PyTorch state dictionary, which is named
        after the class attributes. 
        For example:
        Torch: self.linear_1 = torch.nn.Linear(5, 20)
        ROOT: model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_Linear('X',
        5, 20), 'linear_1')
        
        :param model: The PyTorch model.
        """

        # Transform Python dictionary to C++ map and load parameters.
        m = map[str, vector[float]]()
        for key, value in model.state_dict().items():
            m[key] = value.cpu().numpy().flatten().tolist()
        self.loadParameters(m)


@pythonization("RModel_TorchGNN", ns="TMVA::Experimental::SOFIE")
def pythonize_torchgnn_extractparameters(klass):
    setattr(klass, "extractParameters", RModel_TorchGNN.extractParameters)
