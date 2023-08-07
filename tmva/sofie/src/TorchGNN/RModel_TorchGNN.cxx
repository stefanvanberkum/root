/**
 * Source file for PyTorch Geometric models.
 * 
 * Models are created by the user and parameters can then be loaded into each layer.
 * 
 * IMPORTANT: Changes to the format (e.g., namespaces) may affect the emit
 * defined in RModel_TorchGNN.cxx (save).
*/

#include "TMVA/TorchGNN/RModel_TorchGNN.hxx"
#include <memory>
#include <string>
#include <map>
#include <tuple>
#include <filesystem>
#include <fstream>
#include <ctime>
#include <set>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

/**
 * Save the model as standalone inference code.
 * 
 * @param path Path to save location.
 * @param name Model name.
 * @param overwrite True if any existing directory should be
 * overwritten. Defaults to false.
*/
void RModel_TorchGNN::save(std::string path, std::string name, bool overwrite /*=false*/) {
    std::string dir = path + "/" + name;

    // Get timestamp.
    std::string timestamp = getTimestamp();
    
    if (std::filesystem::exists(dir)) {
        if (overwrite) {
            // Clean directory.
            std::filesystem::remove_all(dir);
        } else {
            // Display warning.
            std::cout << "WARNING: Could not save model. Directory " << dir << " exists and overwrite is set to false.";
        }
    }
    std::filesystem::create_directories(dir);

    // Write methods.
    writeMethods(dir, name, timestamp);
    
    // Write model.
    writeModel(dir, name, timestamp);

    // Write CMakeLists.
    writeCMakeLists(dir, name, timestamp);

    // Create parameter directory.
    std::filesystem::path param_dir = std::filesystem::path(dir);
    param_dir /= "params";
    std::filesystem::create_directory(param_dir);
    
    // Save parameters.
    for (std::shared_ptr<RModule> m: modules) {
        m -> saveParameters(param_dir);
    }
}

/**
 * Write the methods to create a self-contained package.
 * 
 * @param dir Directory to save to.
 * @param name Model name.
 * @param timestamp Timestamp.
*/
void RModel_TorchGNN::writeMethods(std::string dir, std::string name, std::string timestamp) {
    // Retrieve directories.
    std::filesystem::path src_dir = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path inc_dir = src_dir.parent_path().parent_path();
    inc_dir /= "inc";
    inc_dir /= "TMVA";
    inc_dir /= "TorchGNN";

    // Copy header files.
    std::filesystem::create_directory(dir + "/inc");
    std::filesystem::copy(inc_dir, dir + "/inc", std::filesystem::copy_options::recursive);

    // Copy source files.
    std::filesystem::create_directory(dir + "/src");
    std::filesystem::copy(src_dir, dir + "/src", std::filesystem::copy_options::recursive);

    // Iterate over the files to fix the namespaces and other issues.
    std::filesystem::recursive_directory_iterator file_iter = std::filesystem::recursive_directory_iterator(dir);
    std::string line;
    for (const std::filesystem::directory_entry& entry: file_iter) {
        if (entry.is_regular_file()) {
            // Load file.
            std::ifstream fin;
            fin.open(entry.path());

            // Create a temporary file.
            std::ofstream temp;
            std::filesystem::path temp_path = entry.path();
            temp_path.replace_filename("temp" + std::string(temp_path.extension()));
            temp.open(temp_path);
            
            // Write header.
            temp << "// Automatically generated for " << name << "." << std::endl;
            temp << "// " << timestamp << std::endl << std::endl;

            while (std::getline(fin, line)) {
                if (
                    (
                        (line.find("namespace TMVA {") == std::string::npos) && 
                        (line.find("namespace Experimental {") == std::string::npos) &&
                        (line.find("namespace SOFIE {") == std::string::npos) &&
                        (line.find("}  // SOFIE.") == std::string::npos) &&
                        (line.find("}  // Experimental.") == std::string::npos) &&
                        (line.find("}  // TMVA.") == std::string::npos)
                    )
                        || (line.find("line.find") != std::string::npos)
                   ) {
                    // Not a namespace line, so fix other issues and write
                    // to file.
                    std::string del_string = " TMVA_SOFIE_";
                    if ((line.find(del_string) != std::string::npos) && (line.find("del_string") == std::string::npos)) {
                        line.replace(line.find(del_string), del_string.size(), " ");
                    }
                    del_string = "\"TMVA/TorchGNN/";
                    if ((line.find(del_string) != std::string::npos)  && (line.find("del_string") == std::string::npos)) {
                        line.replace(line.find(del_string), del_string.size(), "\"");
                    }
                    del_string = "\"modules/";
                    if ((entry.path().filename().string().find("RModule_") != std::string::npos) && 
                        (line.find(del_string) != std::string::npos)  && 
                        (line.find("del_string") == std::string::npos)) {
                        line.replace(line.find(del_string), del_string.size(), "\"");
                    }
                    temp << line << std::endl;
                }
            }
            temp << std::endl;

            fin.close();
            temp.close();

            std::filesystem::path new_path = temp_path;
            new_path.replace_filename(entry.path().filename());
            std::filesystem::rename(temp_path, new_path);
        }
    }
}

/**
 * Write the model to a file.
 * 
 * @param dir Directory to save to.
 * @param name Model name.
 * @param timestamp Timestamp.
*/
void RModel_TorchGNN::writeModel(std::string dir, std::string name, std::string timestamp) {
    std::ofstream model;
    model.open(dir + "/inc/" + name + ".hxx");
    
    // Write header.
    model << "// Automatically generated for " << name << "." << std::endl;
    model << "// " << timestamp << std::endl << std::endl;
    model << "/** Model definition. */" << std::endl << std::endl;
    
    // Write includes and save parameters.
    model << "#include \"RModel_TorchGNN.hxx\"" << std::endl;
    std::set<std::string_view> used_modules;
    for (std::shared_ptr<RModule> m: modules) {
        // Record module operation.
        used_modules.insert(m -> getOperation());

        // Save parameters.
        std::string module_dir = dir + "/" + std::string(m -> getName());
        std::filesystem::create_directory(dir + "/");
        m -> saveParameters(module_dir);
    }
    for (std::string_view m: used_modules) {
        model << "#include \"modules/RModule_" << m << ".hxx\"" << std::endl;
    }

    model << std::endl;

    // Construct model.
    model << "class " << name << ": public RModel_TorchGNN {" << std::endl;
    model << "\tpublic:" << std::endl;

    // Write model construction.
    model << "\t\t" << name << "(): RModel_TorchGNN({";
    bool first = true;
    for (std::string in: inputs) {  // Input names.
        if (!first) {
            model << ", ";
        } else {
            first = false;
        }
        model << "\"" << in << "\"";
    }
    model << "}, {";
    first = true;
    for (std::vector<int> in_shape: shapes) {  // Input shapes.
        if (!first) {
            model << ", ";
        } else {
            first = false;
        }
        model << "{";
        bool first_dim = true;
        for (int dim: in_shape) {
            if (!first_dim) {
                model << ", ";
            } else {
                first_dim = false;
            }
            model << dim;
        }
        model << "}";
    }
    model << "}) {" << std::endl;
    
    // Write module additions.
    for (std::shared_ptr<RModule> m: modules) {
        if ((m -> getOperation()) == "Input") {
            // Skip input modules.
            continue;
        }

        std::string_view module_name = m -> getName();
        std::string_view op = m -> getOperation();
        std::vector<std::string> module_inputs = m -> getInputs();
        model << "\t\t\taddModule(RModule_" << op << "(";
        first = true;
        for (std::string in: module_inputs) {  // Input names.
            if (!first) {
                model << ", ";
            } else {
                first = false;
            }
            model << "\"" << in << "\"";
        }
        std::vector<std::string> module_args = m -> getArgs();
        for (std::string arg: module_args) {  // Other arguments.
            model << ", " << arg;
        }
        model << "), \"" << module_name << "\");" << std::endl;
    }
    // Write parameter loading.
    model << "\t\t\tloadParameters();" << std::endl;

    model << "\t}" << std::endl;
    model << "};" << std::endl;
    model.close();
}

/**
 * Write the CMakeLists file.
 * 
 * @param dir Directory to save to.
 * @param name Model name.
 * @param timestamp Timestamp.
*/
void RModel_TorchGNN::writeCMakeLists(std::string dir, std::string name, std::string timestamp) {
    std::ofstream f;
    f.open(dir + "/CMakeLists.txt");
    
    // Write header.
    f << "# Automatically generated for " << name << "." << std::endl;
    f << "# " << timestamp << std::endl << std::endl;

    f << "add_library(" << std::endl;
    f << "\t" << name << std::endl;
    f << "\tinc/" << name << ".hxx" << std::endl;
    f << "\tinc/RModel_TorchGNN.hxx" << std::endl;
    f << "\tsrc/RModel_TorchGNN.cxx" << std::endl;
    f << ")" << std::endl << std::endl;

    f << "target_include_directories(" << name << " PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)" << std::endl;
    f.close();
}

}  // SOFIE.
}  // Experimental.
}  // TMVA.
