/**
 * Header file for PyTorch Geometric models.
 * 
 * Models are created by the user and parameters can then be loaded into each layer.
*/

#ifndef TMVA_SOFIE_RMODEL_TORCHGNN
#define TMVA_SOFIE_RMODEL_TORCHGNN

#include "TMVA/RModel.hxx"
#include "TMVA/TorchGNN/RModule.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModel_TorchGNN: public RModel_GNNBase {
    public:
        // TODO: What do all these constructors do? Are they all required?
        // Explicit move ctor/assn.
        RModel_TorchGNN(RModel_GNN&& other);

        RModel_TorchGNN& operator=(RModel_TorchGNN&& other);

        // Disallow copy. Why?
        RModel_TorchGNN(const RModel_TorchGNN& other) = delete;
        RModel_TorchGNN& operator=(const RModel_TorchGNN& other) = delete;

        RModel_TorchGNN(){}
    
        void Generate();

        ~RModel_TorchGNN(){}

        void addModule(std::string module, std::string input="", std::string name="") {
            /**
             * Add a module to the module list.
             * 
             * @param module Module to add.
             * @param name Module name. Defaults to the module type with a count
             * value (e.g., GCNConv_1).
            */
           
            std::string new_name = (name == "") ? module : name
            if (auto search = module_counts.find(new_name); search != module_counts.end()) {
                // Module exists, so increment count.
                module_counts[new_name]++;
            } else {
                // First module of its kind.
                module_counts[new_name] = 1;
            }
            new_name += "_" + std::to_string(module_counts[new_name]);

            modules

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
        }
    private:
        std::map<std::string, int> module_counts;  // Map from module name to number of occurrences.
        std::forward_list<RModule> modules;  // List of modules.
        int module_count;  // Number of modules.

        std::size_t num_node_features;
        std::size_t num_edge_features;
        std::size_t num_global_features;
};

}  // SOFIE.
}  // Experimental.
}  // TMVA.

#endif // TMVA_SOFIE_RMODEL_TORCHGNN.
