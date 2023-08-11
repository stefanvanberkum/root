/**
 * Tests the TorchGNN linear module (RModule_Linear).
 * 
 * To run in ROOT terminal:
 * .L path_to_root/tmva/sofie/test/TorchGNN/LinearTest.cxx
 * main()
*/

#include "TMVA/TorchGNN/RModel_TorchGNN.hxx"
#include "TMVA/TorchGNN/modules/RModule_Linear.hxx"
#include <iostream>

using namespace TMVA::Experimental::SOFIE;

int main() {
    std::vector<float> X = {1, 2, 3, 
                            3, -2, 2
                           };
    std::vector<float> A = {1, 0,
                            -1, 2,
                            2, 1
                           };
    std::vector<float> b = {0, 
                            0.5
                           };

    RModel_TorchGNN model = RModel_TorchGNN({"X"}, {{-1, 3}});
    RModule_Linear lin = RModule_Linear("X", 3, 2);
    lin.setWeights(A);
    lin.setBiases(b);
    model.addModule(lin, "out_1");
    std::vector<float> out = model.forward(X);

    for (float x: out) {
        std::cout << x << std::endl;
    }

    return 0;
}
