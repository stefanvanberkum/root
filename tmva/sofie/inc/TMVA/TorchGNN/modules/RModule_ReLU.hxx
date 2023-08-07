/**
 * ReLU module.
*/

#ifndef TMVA_SOFIE_RMODULE_RELU_H_
#define TMVA_SOFIE_RMODULE_RELU_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule_ReLU: public RModule {
    public:
        /**
         * Construct the ReLU module.
         * 
         * The ReLU operation will be applied element-wise.
         * 
         * @param x The input.
        */
        RModule_ReLU(std::string x) {
            inputs = {x};
            args = {};
        }

        /** Destruct the module. */
        ~RModule_ReLU() {};

        /**
         * Apply the ReLU operation min(0, x).
         * 
         * @returns Result min(0, x).
        */
        std::vector<float> forward() {
            std::vector<float> x = input_modules[0] -> getOutput();

            for (std::size_t i = 0; i < x.size(); i++) {
                if (x[i] < 0) {
                    x[i] = 0;
                }
            }

            return x;
        }

        /**
         * Infer the output shape.
         * 
         * For this module, the output shape is the same as the input shape.
         * 
         * @returns The output shape.
        */
        std::vector<int> inferShape() {
            std::vector<int> shape = input_modules[0] -> getShape();
            return shape;
        }

        /**
         * Get the operation.
         * 
         * @returns The name of the operation.
        */
        std::string_view getOperation() {
            return "ReLU";
        }

        /** 
         * Save parameters.
         * 
         * Does nothing for this module.
         * 
         * @param dir Save directory.
         */
        void saveParameters([[maybe_unused]] std::string dir) {}

        /**
         * Load saved parameters.
         * 
         * Does nothing for this module.
        */
        void loadParameters() {}

        /**
         * Load parameters from PyTorch state dictionary.
         * 
         * Does nothing for this module.
         * 
         * @param state_dict The state dictionary.
        */
        void loadParameters([[maybe_unused]] std::map<std::string, std::vector<float>> state_dict) {}
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RMODULE_RELU_H_
