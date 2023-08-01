/**
 * Base class for RModule objects used in GNNs.
 * 
 * Modules define the operations that can be performed in a forward pass. They
 * can be layers, activations, or generic operations.
*/

#ifndef TMVA_SOFIE_RMODULE_H_
#define TMVA_SOFIE_RMODULE_H_

#include <vector>
#include <memory>
#include <map>
#include <string>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule {
    public:
        virtual ~RModule() {};

        /**
         * Initialize the RModule by binding it to its input modules.
         * 
         * @param module_list Vector containing the modules.
         * @param module_map Map from module name to module index (in modules).
        */
        void initialize(std::vector<std::shared_ptr<RModule>> module_list, std::map<std::string, int> module_map) {
            for (std::string input: inputs) {
                input_modules.push_back(module_list[module_map[input]]);
            }
        }

        /**
         * Execute the module.
         * 
         * This triggers the module's forward method and stores the output and
         * its shape.
        */
        void execute() {
            out_shape = inferShape();  // Infer shape on the fly. TODO: Test impact on performance and possibly execute once for static models.
            output = forward();
        }

        /**
         * Change this module's name.
         * 
         * @param new_name New module name.
        */
        void setName(std::string new_name) {name = new_name;}

        /**
         * Get this module's name.
         * 
         * @returns The module name.
        */
        std::string_view getName() {return name;}

        /**
         * Get the module inputs.
         * 
         * @returns The inputs for this module.
        */
        std::vector<std::string> getInputs() {return inputs;}

        /**
         * Get the output of the last call to this module.
         * 
         * @returns The output of the last call.
        */
        std::vector<float> getOutput() {return output;}

        /**
         * Get the output shape of the last call to this module.
         * 
         * @returns The output shape of the last call.
        */
        std::vector<int> getShape() {return out_shape;}

        virtual std::vector<float> forward() = 0;  // Forward method to be implemented by each module.

        virtual std::vector<int> inferShape() = 0;  // Output shape inference to be implemented by each module.

        virtual std::string_view getOperation() = 0;  // Operation name getter to be implemented by each module.
        
        virtual void saveParameters(std::string dir) = 0;  // Parameter saver to be implemented by each module.

        virtual void loadParameters() = 0;  // Parameter loader to be implemented by each module.
    protected:
        std::vector<std::shared_ptr<RModule>> input_modules;  // Vector of input modules.
        std::vector<std::string> inputs;  // Input names. 
    private:
        std::string name;  // Module name.  
        std::vector<float> output;  // Output of last call.
        std::vector<int> out_shape;  // Output shape of last call.
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RMODULE_H_
