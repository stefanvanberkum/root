/**
 * Input module.
 * 
 * Used internally.
*/

#ifndef TMVA_SOFIE_RMODULE_INPUT_H_
#define TMVA_SOFIE_RMODULE_INPUT_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule_Input: public RModule {
    public:
        /**
         * Construct the input module.
         * 
         * Simply stores the input so that child modules can access it.
        */
        RModule_Input() {
            inputs = {};  // No previous inputs to this module.
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
         * Does nothing for this operation.
         */
        void saveParameters([[maybe_unused]] std::string dir) {}

        /**
         * Load parameters.
         * 
         * Does nothing for this operation.
        */
        void loadParameters() {}
    private:
        std::vector<float> in;
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RMODULE_INPUT_H_
