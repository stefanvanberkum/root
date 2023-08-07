/**
 * Softmax module.
*/

#ifndef TMVA_SOFIE_RMODULE_SOFTMAX_H_
#define TMVA_SOFIE_RMODULE_SOFTMAX_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"
#include <cmath>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule_Softmax: public RModule {
    public:
        /**
         * Construct the softmax module.
         * 
         * @param x The input.
        */
        RModule_Softmax(std::string x) {
            inputs = {x};
            args = {};
        }

        /** Destruct the module. */
        ~RModule_Softmax() {};

        /**
         * Apply the softmax operation exp(x_i) / sum(exp(x_j)).
         * 
         * The sum is taken over the last dimension.
         * 
         * @returns Result exp(x_i) / sum(exp(x_j)).
        */
        std::vector<float> forward() {
            std::vector<float> x = input_modules[0] -> getOutput();
            int last_dim = input_modules[0] -> getShape().back();

            for (std::size_t i = 0; i < x.size(); i += last_dim) {
                float exps[last_dim];
                float sum = 0;
                for (std::size_t j = i; j < i + last_dim; j++) {
                    exps[j - i] = exp(x[j]);
                    sum += exps[j - i];
                }
                for (std::size_t j = i; j < i + last_dim; j++) {
                    x[j] = exps[j - i] / sum;
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
            return "Softmax";
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

#endif  // TMVA_SOFIE_RMODULE_SOFTMAX_H_
