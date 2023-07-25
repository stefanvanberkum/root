/**
 * Tests basic machinery of the TorchGNN functionality.
*/

#include "TMVA/TorchGNN/RModel_TorchGNN.hxx"
#include "TMVA/TorchGNN/layers/RLayer_Add.hxx"

using namespace TMVA::Experimental::SOFIE;

int main() {
    std::vector<float> a = {1, 1.5, 2, 2.5};
    std::vector<float> b = {1, 2, 4, 8};
    std::vector<float> c = {0, 0.5, 0, 0.5};

    RModel_TorchGNN model = RModel_TorchGNN({"a", "b", "c"});
    model.addModule(std::make_shared<RLayer_Add>("a", "b"), "out_1");
    model.addModule(std::make_shared<RLayer_Add>("out_1", "c"), "out_2");
    std::vector<float> out = model.forward(a, b, c);

    for (float x: out) {
        std::cout << x;
    }

    return 0;
}
