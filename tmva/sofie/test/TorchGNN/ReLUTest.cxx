/**
 * Tests TorchGNN ReLU module (RModule_ReLU).
 * 
 * To run in ROOT terminal:
 * .L /home/stefan/root/tmva/sofie/test/TorchGNN/ReLUTest.cxx
*/

#include "TMVA/TorchGNN/RModel_TorchGNN.hxx"
#include "TMVA/TorchGNN/modules/RModule_ReLU.hxx"
#include <iostream>

using namespace TMVA::Experimental::SOFIE;

int main() {
    std::vector<float> a = {-2, -1, 0, 1, 2};

    RModel_TorchGNN model = RModel_TorchGNN({"a"}, {{-1}});
    model.addModule(std::make_shared<RModule_ReLU>("a"), "out_1");
    std::vector<float> out = model.forward(a);

    for (float x: out) {
        std::cout << x << std::endl;
    }

    return 0;
}
