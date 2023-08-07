/**
 * Tests the TorchGNN softmax module (RModule_Softmax).
 * 
 * To run in ROOT terminal:
 * .L /home/stefan/root/tmva/sofie/test/TorchGNN/SoftmaxTest.cxx
*/

#include "TMVA/TorchGNN/RModel_TorchGNN.hxx"
#include "TMVA/TorchGNN/modules/RModule_Softmax.hxx"
#include <iostream>

using namespace TMVA::Experimental::SOFIE;

int main() {
    std::vector<float> a = {0, 0, 0, 4, 1, 2, -10, -10, 10};

    RModel_TorchGNN model = RModel_TorchGNN({"a"}, {{-1, 3}});
    model.addModule(RModule_Softmax("a"), "out_1");
    std::vector<float> out = model.forward(a);

    for (float x: out) {
        std::cout << x << std::endl;
    }

    return 0;
}
