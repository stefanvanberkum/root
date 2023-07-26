/**
 * Addition layer.
*/

#ifndef TMVA_SOFIE_RLAYER_ADD_H_
#define TMVA_SOFIE_RLAYER_ADD_H_

#include "TMVA/TorchGNN/RModule.hxx"
#include <gsl/gsl_cblas.h>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RLayer_Add: public RModule {
    public:
        /**
         * Construct the addition layer.
         * 
         * @param a The first argument.
         * @param b The second argument.
        */
        RLayer_Add(std::string a, std::string b) {
            inputs = {a, b};
        }

        /** Destruct the module. */
        ~RLayer_Add() {};

        /**
         * Add the arguments a and b.
         * 
         * @returns Result (a + b).
        */
        std::vector<float> forward() {
            std::vector<float> a = input_modules[0] -> getOutput();
            std::vector<float> b = input_modules[1] -> getOutput();

            int n = a.size();

            cblas_saxpy(n, 1, a.data(), 1, b.data(), 1);

            return b;
        }
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RLAYER_ADD_H_
