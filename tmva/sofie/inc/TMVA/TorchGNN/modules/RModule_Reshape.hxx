/**
 * Reshape module.
*/

#ifndef TMVA_SOFIE_RMODULE_RESHAPE_H_
#define TMVA_SOFIE_RMODULE_RESHAPE_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"
#include <algorithm>
#include <stdexcept>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule_Reshape: public RModule {
    public:
        /**
         * Construct the reshape module.
         * 
         * Accepts one wildcard dimension (-1).
         * 
         * @param x The input.
         * @param shape The desired shape.
        */
        RModule_Reshape(std::string x, std::vector<int> shape) {
            s = shape;
            wildcard = std::find(shape.begin(), shape.end(), -1) - shape.begin();

            // Check shape.
            if (std::any_of(shape.begin(), shape.end(), [](int i){return i == 0;})) {
                throw std::invalid_argument("Dimension cannot be zero.");
            }
            if (std::any_of(shape.begin(), shape.end(), [](int i){return i < -1;})) {
                throw std::invalid_argument("Shape cannot have negative entries (except for the wildcard dimension).");
            }
            if (std::count(shape.begin(), shape.end(), -1) > 1) {
                throw std::invalid_argument("Shape may have at most one wildcard.");
            }

            // Translate shape argument to string.
            std::string shape_arg = "{";
            bool first = true;
            for (int i: shape) {
                if (!first) {
                    shape_arg += ", ";
                } else {
                    first = false;
                }
                shape_arg += i;
            }
            shape_arg += "}";

            inputs = {x};
            args = {shape_arg};
        }

        /** Destruct the module. */
        ~RModule_Reshape() {};

        /**
         * Simply forward the input.
         * 
         * Reshaping is done through the inferShape() method.
         * 
         * @returns The input.
        */
        std::vector<float> forward() {
            std::vector<float> x = input_modules[0] -> getOutput();
            return x;
        }

        /**
         * Infer the output shape.
         * 
         * For this module, the output shape is given by the user.
         * 
         * @returns The output shape.
        */
        std::vector<int> inferShape() {
            int cprod = 1;
            for (std::size_t i = 0; i < s.size(); i++) {
                if (i != wildcard) {
                    cprod *= s[i];
                }
            }
            std::vector<int> shape = s;
            std::vector<float> x = input_modules[0] -> getOutput();
            shape[wildcard] = x.size() / cprod;
            return shape;
        }

        /**
         * Get the operation.
         * 
         * @returns The name of the operation.
        */
        std::string_view getOperation() {
            return "Reshape";
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
        std::vector<int> s;  // Desired output shape.
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RMODULE_RESHAPE_H_
