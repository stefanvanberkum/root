/**
 * Header file for PyTorch Geometric models.
 * 
 * Models are created by the user and parameters can then be loaded into each layer.
 * 
 * IMPORTANT: Changes to the format (e.g., namespaces) may affect the emit
 * defined in RModel_TorchGNN.cxx (save).
*/

#ifndef TMVA_SOFIE_RMODEL_TORCHGNN_H_
#define TMVA_SOFIE_RMODEL_TORCHGNN_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"
#include "TMVA/TorchGNN/modules/RModule_Input.hxx"
#include <stdexcept>
#include <iostream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModel_TorchGNN {
    public:
        /** Model constructor without inputs. */
        RModel_TorchGNN() {}

        /**
         * Model constructor with manual input names.
         * 
         * @param input_names Vector of input names.
         * @param input_shapes Vector of input shapes. Each element may contain
         * at most one wildcard (-1).
        */
        RModel_TorchGNN(std::vector<std::string> input_names, std::vector<std::vector<int>> input_shapes) {
            inputs = input_names;
            shapes = input_shapes;

            // Generate input layers.
            for (std::size_t i = 0; i < input_names.size(); i++) {
                // Check shape.
                if (std::any_of(input_shapes[i].begin(), input_shapes[i].end(), [](int j){return j == 0;})) {
                    throw std::invalid_argument("Invalid input shape for input " + input_names[i] + ". Dimension cannot be zero.");
                }
                if (std::any_of(input_shapes[i].begin(), input_shapes[i].end(), [](int j){return j < -1;})) {
                    throw std::invalid_argument("Invalid input shape for input " + input_names[i] + ". Shape cannot have negative entries (except for the wildcard dimension).");
                }
                if (std::count(input_shapes[i].begin(), input_shapes[i].end(), -1) > 1) {
                    throw std::invalid_argument("Invalid input shape for input " + input_names[i] + ". Shape may have at most one wildcard.");
                }
                addModule(RModule_Input(input_shapes[i]), input_names[i]);
            }
        }

        /**
         * Add a module to the module list.
         * 
         * @param module Module to add.
         * @param name Module name. Defaults to the module type with a count
         * value (e.g., GCNConv_1).
        */
        template<typename T>
        void addModule(T module, std::string name="") {
            std::string new_name = (name == "") ? std::string(module.getOperation()) : name;
            if (module_counts[new_name] > 0) {
                // Module exists, so add discriminator and increment count.
                new_name += "_" + std::to_string(module_counts[new_name]);
                module_counts[new_name]++;

                if (name != "") {
                    // Issue warning.
                    std::cout << "WARNING: Module with duplicate name \"" << name << "\" renamed to \"" << new_name << "\"." << std::endl;
                }
            } else {
                // First module of its kind.
                module_counts[new_name] = 1;
            }
            module.setName(new_name);

            // Initialize the module.
            module.initialize(modules, module_map);

            // Add module to the module list.
            modules.push_back(std::make_shared<T>(module));
            module_map[std::string(module.getName())] = module_count;
            module_count++;
        }
        
        /**
         * Run the forward function.
         * 
         * @param args Any number of input arguments.
         * @returns The output of the last layer.
        */
        template<class... Types>
        std::vector<float> forward(Types... args) {
            auto input = std::make_tuple(args...);

            // Instantiate input layers.
            int k = 0;
            std::apply(
                [&](auto&... in) {
                    ((std::dynamic_pointer_cast<RModule_Input>(modules[k++]) -> setParams(in)), ...);
                }, input);

            // Loop through and execute modules.
            for (std::shared_ptr<RModule> module: modules) {
                module -> execute();
            }

            // Return output of the last layer.
            return modules.back() -> getOutput();
        }

        /**
         * Load parameters from PyTorch state dictionary for all modules.
         * 
         * @param state_dict The state dictionary.
        */
        void loadParameters(std::map<std::string, std::vector<float>> state_dict) {
            for (std::shared_ptr<RModule> module: modules) {
                module -> loadParameters(state_dict);
            }
        }

        /**
         * Load saved parameters for all modules.
        */
        void loadParameters() {
            for (std::shared_ptr<RModule> module: modules) {
                module -> loadParameters();
            }
        }

        /**
         * Save the model as standalone inference code.
         * 
         * @param path Path to save location.
         * @param name Model name.
         * @param overwrite True if any existing directory should be
         * overwritten. Defaults to false.
        */
        void save(std::string path, std::string name, bool overwrite=false);
    private:
        /**
         * Get a timestamp.
         * 
         * @returns The timestamp in string format.
        */
        static std::string getTimestamp() {
            time_t rawtime;
            struct tm * timeinfo;
            char timestamp [80];
            time(&rawtime);
            timeinfo = localtime(&rawtime);
            strftime(timestamp, 80, "Timestamp: %d-%m-%Y %T.", timeinfo);
            return timestamp;
        }

        /**
         * Write the methods to create a self-contained package.
         * 
         * @param dir Directory to save to.
         * @param name Model name.
         * @param timestamp Timestamp.
        */
        void writeMethods(std::string dir, std::string name, std::string timestamp);

        /**
         * Write the model to a file.
         * 
         * @param dir Directory to save to.
         * @param name Model name.
         * @param timestamp Timestamp.
        */
        void writeModel(std::string dir, std::string name, std::string timestamp);

        /**
         * Write the CMakeLists file.
         * 
         * @param dir Directory to save to.
         * @param name Model name.
         * @param timestamp Timestamp.
        */
        void writeCMakeLists(std::string dir, std::string name, std::string timestamp);

        std::vector<std::string> inputs;  // Vector of input names.
        std::vector<std::vector<int>> shapes;  // Vector of input shapes.
        std::map<std::string, int> module_counts;  // Map from module name to number of occurrences.
        std::vector<std::shared_ptr<RModule>> modules;  // Vector containing the modules.
        std::map<std::string, int> module_map;  // Map from module name to module index (in modules).
        int module_count = 0;  // Number of modules.
};

}  // SOFIE.
}  // Experimental.
}  // TMVA.

#endif  // TMVA_SOFIE_RMODEL_TORCHGNN_H_
