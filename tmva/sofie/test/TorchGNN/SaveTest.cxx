/**
 * Tests saving functionality of TorchGNN.
 * 
 * To run in ROOT terminal:
 * .L /home/stefan/root/tmva/sofie/test/TorchGNN/SaveTest.cxx
*/

#include "TMVA/TorchGNN/RModel_TorchGNN.hxx"
#include "TMVA/TorchGNN/modules/RModule_Add.hxx"

using namespace TMVA::Experimental::SOFIE;

int main() {
    std::vector<float> a = {1, 1.5, 2, 2.5};
    std::vector<float> b = {1, 2, 4, 8};
    std::vector<float> c = {0, 0.5, 0, 0.5};

    RModel_TorchGNN model = RModel_TorchGNN({"a", "b"}, {{-1}, {-1}});
    model.addModule(std::make_shared<RModule_Add>("a", "b"), "out_1");
    model.addModule(std::make_shared<RModule_Add>("out_1", "c"), "out_2");
    model.save("/home/stefan/root-model", "model", true);

    return 0;
}
