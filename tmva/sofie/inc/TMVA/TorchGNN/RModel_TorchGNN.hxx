/**
 * Header file for PyTorch Geometric models.
 * 
 * Models are created by the user and parameters can then be loaded into each layer.
*/

#ifndef TMVA_SOFIE_RMODEL_TORCHGNN_H_
#define TMVA_SOFIE_RMODEL_TORCHGNN_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"
#include "TMVA/TorchGNN/modules/RModule_Input.hxx"
#include <stdexcept>

// TODO: User-convenience script in Python to load parameters from state_dict?.
// - Load parameters for all modules.
// - Use getInputs() and get corresponding variable from state_dict.

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
                if (std::count(input_shapes[i].begin(), input_shapes[i].end(), -1) > 1) {
                    throw std::invalid_argument("Invalid input shape for input " + input_names[i] + ". Shape may have at most one wildcard.");
                }
                addModule(std::make_shared<RModule_Input>(input_shapes[i]), input_names[i]);
            }
        }

        /**
         * Add a module to the module list.
         * 
         * @param module Module to add.
         * @param name Module name. Defaults to the module operation with a count
         * value (e.g., ReLU_1).
        */
        void addModule(std::shared_ptr<RModule> module, std::string name="");
        
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
