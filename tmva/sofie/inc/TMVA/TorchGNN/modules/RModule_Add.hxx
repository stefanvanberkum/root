/**
 * Addition module.
*/

#ifndef TMVA_SOFIE_RMODULE_ADD_H_
#define TMVA_SOFIE_RMODULE_ADD_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"
#include <gsl/gsl_cblas.h>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule_Add: public RModule {
    public:
        /**
         * Construct the addition module.
         * 
         * @param a The first argument.
         * @param b The second argument.
        */
        RModule_Add(std::string a, std::string b) {
            inputs = {a, b};
        }

        /** Destruct the module. */
        ~RModule_Add() {};

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

        /**
         * Get the operation.
         * 
         * @returns The name of the operation.
        */
        std::string_view getOperation() {
            return "Add";
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
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RMODULE_ADD_H_
