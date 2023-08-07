/**
 * Input module.
 * 
 * Used internally.
*/

#ifndef TMVA_SOFIE_RMODULE_INPUT_H_
#define TMVA_SOFIE_RMODULE_INPUT_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"
#include <algorithm>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule_Input: public RModule {
    public:
        /**
         * Construct the input module.
         * 
         * The module stores the input and its shape so that child modules can access it.
         * 
         * @param input_shape The shape of the input. 
        */
        RModule_Input(std::vector<int> input_shape) {
            in_shape = input_shape;
            wildcard = std::find(input_shape.begin(), input_shape.end(), -1) - input_shape.begin();
            
            inputs = {};  // No previous inputs to this module.
            args = {};
        }

        /** Destruct the module. */
        ~RModule_Input() {};

        /**
         * Assign input.
         * 
         * @param input The input.
        */
        void setParams(std::vector<float> input) {
            in = input;
        }

        /**
         * Simply forward the input.
         * 
         * @returns The input.
        */
        std::vector<float> forward() {
            return in;
        }

        /**
         * Infer the output shape.
         * 
         * For this module, the output shape is the same as the input shape
         * with an inferred value for the wildcard dimension.
         * 
         * @returns The output shape.
        */
        std::vector<int> inferShape() {
            int cprod = 1;
            for (std::size_t i = 0; i < in_shape.size(); i++) {
                if (i != wildcard) {
                    cprod *= in_shape[i];
                }
            }
            std::vector<int> shape = in_shape;
            shape[wildcard] = in.size() / cprod;
            return shape;
        }

        /**
         * Get the operation.
         * 
         * @returns The name of the operation.
        */
        std::string_view getOperation() {
            return "Input";
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
    private:
        std::size_t wildcard;  // Index of the wildcard dimension.
        std::vector<float> in;  // Input.
        std::vector<int> in_shape;  // Input shape.
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RMODULE_INPUT_H_
