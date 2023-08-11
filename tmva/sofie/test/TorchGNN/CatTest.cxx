/**
 * Tests the TorchGNN concatenation module (RModule_Cat).
 * 
 * To run in ROOT terminal:
 * .L path_to_root/tmva/sofie/test/TorchGNN/CatTest.cxx
 * main()
*/

#include "TMVA/TorchGNN/RModel_TorchGNN.hxx"
#include "TMVA/TorchGNN/modules/RModule_Cat.hxx"
#include <iostream>

using namespace TMVA::Experimental::SOFIE;

int main() {
    // 2x3x2.
    std::vector<float> a = {    1, 2,
                                3, 4,
                                5, 6,
                                    7, 8,
                                    9, 10,
                                    11, 12
    };
    // 1x3x2.
    std::vector<float> b = {    -1, -2,
                                -3, -4,
                                -5, -6
    };

    RModel_TorchGNN model = RModel_TorchGNN({"a", "b"}, {{-1, 3, 2}, {-1, 3, 2}});
    model.addModule(RModule_Cat("a", "b", 0), "out_1");
    std::vector<float> out = model.forward(a, b);

    for (float x: out) {
        std::cout << x << std::endl;
    }

    std::cout << std::endl;

    // 2x1x2.
    b = {   -1, -2,
                -3, -4
    };

    model = RModel_TorchGNN({"a", "b"}, {{-1, 3, 2}, {-1, 1, 2}});
    model.addModule(RModule_Cat("a", "b", 1), "out_1");
    out = model.forward(a, b);

    for (float x: out) {
        std::cout << x << std::endl;
    }

    std::cout << std::endl;

    // 2x3x1.
    b = {   -1, 
            -2,
            -3, 
                -4,
                -5, 
                -6
    };

    model = RModel_TorchGNN({"a", "b"}, {{-1, 3, 2}, {-1, 3, 1}});
    model.addModule(RModule_Cat("a", "b", 2), "out_1");
    out = model.forward(a, b);

    for (float x: out) {
        std::cout << x << std::endl;
    }

    std::cout << std::endl;

    return 0;
}
