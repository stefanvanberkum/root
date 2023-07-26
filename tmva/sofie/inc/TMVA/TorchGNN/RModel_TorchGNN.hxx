/**
 * Header file for PyTorch Geometric models.
 * 
 * Models are created by the user and parameters can then be loaded into each layer.
*/

#ifndef TMVA_SOFIE_RMODEL_TORCHGNN_H_
#define TMVA_SOFIE_RMODEL_TORCHGNN_H_

#include "TMVA/TorchGNN/RModule.hxx"
#include "TMVA/TorchGNN/layers/RLayer_Input.hxx"
#include <tuple>
#include <iostream>
#include <filesystem>
#include <fstream>

// TODO: User-convenience script in Python to convert tensors to float vector?

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
        */
        RModel_TorchGNN(std::vector<std::string> input_names) {
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

        /**
         * Add a module to the module list.
         * 
         * @param module Module to add.
         * @param name Module name. Defaults to the module type with a count
         * value (e.g., GCNConv_1).
        */
        void addModule(std::shared_ptr<RModule> module, std::string name="") {
            std::string new_name = (name == "") ? typeid(module).name() : name;
            if (auto search = module_counts.find(new_name); search != module_counts.end()) {
                // Module exists, so add discriminator and increment count.
                new_name += "_" + std::to_string(module_counts[new_name]);
                module_counts[new_name]++;

                if (name != "") {
                    // Issue warning.
                    std::cout << "WARNING: Module with name \"" << name << "\" renamed to \"" << new_name << "\"." << std::endl;
                }
            } else {
                // First module of its kind.
                module_counts[new_name] = 1;
            }
            module -> setName(new_name);

            // Initialize the module.
            module -> initialize(modules, module_map);

            // Add module to the module list.
            modules.push_back(module);
            module_map[module -> getName()] = module_count;
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
            auto input = make_tuple(args...);

            // Instantiate input layers.
            int k = 0;
            std::apply(
                [&](auto&... in) {
                    ((std::dynamic_pointer_cast<RLayer_Input>(modules[k++]) -> setParams(in)), ...);
                }, input);

            // Loop through and execute modules.
            for (std::shared_ptr<RModule> module: modules) {
                module -> execute();
            }

            // Return output of the last layer.
            return modules.back() -> getOutput();
        }

        /**
         * Save the model as standalone inference code.
         * 
         * @param path Path to save location.
         * @param name Model name.
         * @param overwrite True if any existing directory should be
         * overwritten. Defaults to false.
        */
        void save(std::string path, std::string name, bool overwrite=false) {
            std::filesystem::copy_options copyOptions;
            if (overwrite) {
                copyOptions = std::filesystem::copy_options::overwrite_existing | std::filesystem::copy_options::recursive;
            } else {
                copyOptions = std::filesystem::copy_options::recursive;
            }

            // Copy methods.
            std::string dir = path + "/" + name;
            std::filesystem::copy("TMVA/TorchGNN", dir, copyOptions);

            // Iterate over the files to fix the namespaces.
            std::filesystem::recursive_directory_iterator file_iter = std::filesystem::recursive_directory_iterator(dir);
            for (const std::filesystem::directory_entry& entry : file_iter) {
                // Load file.
                std::ifstream fin;
                fin.open(entry.path());

                // Create a temporary file.
                std::ofstream temp;
                std::filesystem::path temp_path = entry.path();
                temp_path.replace_filename("temp");
                temp.open(temp_path);
            } 
        }
    private:
        std::vector<std::string> inputs;  // Vector of input names.
        std::map<std::string, int> module_counts;  // Map from module name to number of occurrences.
        std::vector<std::shared_ptr<RModule>> modules;  // Vector containing the modules.
        std::map<std::string, int> module_map;  // Map from module name to module index (in modules).
        int module_count = 0;  // Number of modules.
};

}  // SOFIE.
}  // Experimental.
}  // TMVA.

#endif  // TMVA_SOFIE_RMODEL_TORCHGNN_H_
