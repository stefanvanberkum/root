/**
 * Base class for RModule objects used in GNNs.
 * 
 * Modules define the operations that can be performed in a forward pass. They
 * can be layers, activations, or generic operations.
*/

#ifndef TMVA_SOFIE_RMODULE_H_
#define TMVA_SOFIE_RMODULE_H_

#include "TMVA/SOFIE_common.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule {
    public:
        virtual ~RModule() = default;

        void initialize(std::vector<std::shared_ptr<RModule>> module_list, std::map<std::string, int> module_map) {
            /**
             * Initialize the RModule by binding it to its input modules.
             * 
             * @param module_list Vector containing the modules.
             * @param module_map Map from module name to module index (in modules).
            */

            for (std::string input: inputs) {
                input_modules.push_back(module_list[module_map[input]]);
            }
        }

        void execute() {
            /**
             * Execute the module.
             * 
             * This triggers the module's forward method and stores the output.
            */

            std::vector<float> out = forward();
            output = out;
        }

        virtual std::vector<float> forward() = 0;  // Forward method to be implemented by each module.

        void setName(std::string new_name) {name = new_name;}  // Change this module's name.
        std::string getName() {return name;}  // Get this module's name.

        std::vector<float> getOutput() {return output;}  // Get output of last call.
    protected:
        std::vector<std::shared_ptr<RModule>> input_modules;  // Vector of input modules.
        std::vector<std::string> inputs;  // Input names.
    private:
        std::string name;  // Module name.  
        std::vector<float> output;  // Output of last call.
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RMODULE_H_
