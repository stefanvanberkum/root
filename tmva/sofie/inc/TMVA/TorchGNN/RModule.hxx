/**
 * Base class for RModule objects used in GNNs.
 * 
 * Modules define the operations that can be performed in a forward pass. They
 * can be layers, activations, or generic operations.
*/

#ifndef TMVA_SOFIE_RMODULE
#define TMVA_SOFIE_RMODULE

#include <string>
#include <SOFIE_common.hxx>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule {
    public:
        // TODO: What do these do? Can they be removed?
        RModule(){}
        virtual ~RModule(){}

        // Module name is set to operation name.
        RModule(std::string new_op):
            op_type(UTILITY::Clean_name(new_op)), name(UTILITY::Clean_name(new_op)){}

        // Manual name.
        RModule(std::string new_op, std::string new_name):
            op_type(UTILITY::Clean_name(new_op)), name(UTILITY::Clean_name(new_name)){}

        std::string getOperation() {return op_type;}
        std::string setName(std::string new_name) {name = new_name;}
        std::string getName() {return name;}
    protected:
        ModuleType op_type;  // Operation type.
        std::string name;  // Module name.
};

enum class ModuleType {
    Linear,
    GCNConv,
    GATConv,
    relu,
    global_mean_pool,
    reshape,
    cat
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif // TMVA_SOFIE_RMODULE.
