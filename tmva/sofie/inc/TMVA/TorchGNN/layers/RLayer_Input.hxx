/**
 * Input layer.
 * 
 * Used internally.
*/

#ifndef TMVA_SOFIE_RLAYER_INPUT_H_
#define TMVA_SOFIE_RLAYER_INPUT_H_

#include "TMVA/TorchGNN/RModule.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RLayer_Input: public RModule {
    public:
        /**
         * Construct the input layer.
         * 
         * Simply stores the input so that child modules can access it.
        */
        RLayer_Input() {
            inputs = {};  // No previous inputs to this layer.
        }

        /** Destruct the module. */
        ~RLayer_Input() {};

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

    private:
        std::vector<float> in;
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RLAYER_INPUT_H_
