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
        RLayer_Input() {
        /**
         * Construct the input layer.
         * 
         * Simply stores the input so that child modules can access it.
        */
            inputs = {};  // No previous inputs to this layer.
        }

        void setParams(std::vector<float> input) {
            /**
             * Assign input.
             * 
             * @param input The input.
            */
            in = input;
        }

        std::vector<float> forward() {
            /**
             * Simply forward the input.
             * 
             * @returns The input.
            */
            return in;
        }

    private:
        std::vector<float> in;
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RLAYER_INPUT_H_
