/**
 * Header file for PyTorch Geometric models.
 * 
 * Models are created by the user and parameters can then be loaded into each layer.
*/

#ifndef TMVA_SOFIE_RMODEL_TORCHGNN_H_
#define TMVA_SOFIE_RMODEL_TORCHGNN_H_

#include "TMVA/TorchGNN/RModule.hxx"
#include "TMVA/TorchGNN/layers/RLayer_Input.hxx"

// Forward method loops through forward list.
                // While evaluating, it stores only those intermediate outputs
                // that are required in a map: if (index != last) {add to those
                // that should be stored}. Always store last output.
                
                // When a module is called, the corresponding input is fetched
                // and fed through the module. Private RModule attribute?

                // Final output is returned.

            // Input index for each module is stored somewhere, by default it's
            // module_count (the output of the previous module). But index or
            // module name can be set manually. if (input=="") {use count} else
            // {find corresponding index}.

            // Save RModel_TorchGNN using WriteTObject?

            // User-convenience script in Python to convert tensors to BLAS?

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModel_TorchGNN {
    public:
        RModel_TorchGNN() {}

        RModel_TorchGNN(std::vector<std::string> input_names) {
            /**
             * RModel constructor with manual input names.
             * 
             * @param input_names Vector of input names.
            */

            inputs = input_names;

            // Generate input layers.
            for (std::string name: input_names) {
                addModule(std::make_shared<RLayer_Input>(), name);
            }
        }
        //RModel_TorchGNN(std::vector<std::string> input_names, std::vector<std::string> module_list) {
            // TODO: Initialize modules directly.
        //}
        //RModel_TorchGNN(std::vector<std::string> input_names, std::vector<std::string> module_list, std::vector<std::string> param_files) {
            // TODO: Initialize modules directly and add weights.
        //}

        void addModule(std::shared_ptr<RModule> module, std::string name="") {
            /**
             * Add a module to the module list.
             * 
             * @param module Module to add.
             * @param name Module name. Defaults to the module type with a count
             * value (e.g., GCNConv_1).
            */
           
            std::string new_name = (name == "") ? typeid(module).name() : name;
            if (auto search = module_counts.find(new_name); search != module_counts.end()) {
                // Module exists, so increment count.
                module_counts[new_name]++;
            } else {
                // First module of its kind.
                module_counts[new_name] = 1;
            }
            new_name += "_" + std::to_string(module_counts[new_name]);
            module -> setName(new_name);

            // Initialize the module.
            module -> initialize(modules, module_map);

            // Add module to the module list.
            modules.push_back(module);
            module_map[module -> getName()] = module_count;
            module_count++;
        }
        
        template<class... Types>
        std::vector<float> forward(Types... args) {
            /**
             * Run the forward function.
             * 
             * @param args Any number of input arguments.
             * @returns The output of the last layer.
            */
            auto input = make_tuple(args...);
            std::size_t n_inputs = std::tuple_size<decltype(input)>{};

            // Instantiate input layers.
            for (int i = 0; i < n_inputs; i++) {
                std::dynamic_pointer_cast<RLayer_Input>(modules[i]) -> setParams(std::get<i>(input));
            }

            // Loop through and execute modules.
            for (std::shared_ptr<RModule> module: modules) {
                module -> execute();
            }

            // Return output of the last layer.
            return modules.back() -> getOutput();
        }

    private:
        std::vector<std::string> inputs;  // Input names.
        std::map<std::string, int> module_counts;  // Map from module name to number of occurrences.
        std::vector<std::shared_ptr<RModule>> modules;  // Vector containing the modules.
        std::map<std::string, int> module_map; // Map from module name to module index (in modules).
        int module_count = 0;  // Number of modules.
};

}  // SOFIE.
}  // Experimental.
}  // TMVA.

#endif  // TMVA_SOFIE_RMODEL_TORCHGNN_H_
