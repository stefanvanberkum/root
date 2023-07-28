/**
 * Source file for PyTorch Geometric models.
 * 
 * Models are created by the user and parameters can then be loaded into each layer.
*/

#include "TMVA/TorchGNN/RModel_TorchGNN.hxx"
#include <memory>
#include <string>
#include <map>
#include <tuple>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <ctime>
#include <set>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

/**
 * Add a module to the module list.
 * 
 * @param module Module to add.
 * @param name Module name. Defaults to the module type with a count
 * value (e.g., GCNConv_1).
*/
void RModel_TorchGNN::addModule(std::shared_ptr<RModule> module, std::string name /*=""*/) {
    std::string new_name = (name == "") ? std::string(module -> getOperation()) : name;
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
    module -> setName(new_name);

    // Initialize the module.
    module -> initialize(modules, module_map);

    // Add module to the module list.
    modules.push_back(module);
    module_map[std::string(module -> getName())] = module_count;
    module_count++;
}

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
    
    if (std::filesystem::exists(path)) {
        if (overwrite) {
            // Clean directory.
            std::filesystem::remove_all(path);
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

    // Iterate over the files to fix the namespaces.
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
                if ((line.find("namespace ") == std::string::npos) && (line.find("}  //") == std::string::npos)) {
                    // Not a namespace line, so fix other issues and write
                    // to file.
                    std::string del_string = " TMVA_SOFIE_";
                    if (line.find(del_string) != std::string::npos) {
                        line.replace(line.find(del_string), del_string.size(), " ");
                    }
                    del_string = "\"TMVA/TorchGNN/";
                    if (line.find(del_string) != std::string::npos) {
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
    model.open(dir + "/" + name + ".hxx");
    
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
        model << "#include \"modules/RModule_" << m << ".hxx" << std::endl;
    }

    model << std::endl;

    // Construct model.
    model << "RModel_TorchGNN build() {" << std::endl;

    // Write model construction.
    model << "\tRModel_TorchGNN model = RModel_TorchGNN({";
    bool first = true;
    for (std::string in: inputs) {
        if (!first) {
            model << ", ";
        } else {
            first = false;
        }
        model << in;
    }
    model << "});" << std::endl;
    
    // Write module additions.
    for (std::shared_ptr<RModule> m: modules) {
        if ((m -> getOperation()) == "Input") {
            // Skip input modules.
            continue;
        }

        std::string_view module_name = m -> getName();
        std::string_view op = m -> getOperation();
        std::vector<std::string> module_inputs = m -> getInputs();
        model << "\tmodel.addModule(std::make_shared<" << op << ">(";
        first = true;
        for (std::string in: module_inputs) {
            if (!first) {
                model << ", ";
            } else {
                first = false;
            }
            model << "\"" << in << "\"";
        }
        model << "), \"" << module_name << "\");" << std::endl;
    }
    // Write parameter loading.
    model << "\tmodel.loadParameters();" << std::endl;

    // Return model.
    model << "\treturn model;" << std::endl;
    model << "}" << std::endl << std::endl;
}

/**
 * Write the CMakeLists file.
 * 
 * @param dir Directory to save to.
 * @param name Model name.
 * @param timestamp Timestamp.
*/
void RModel_TorchGNN::writeCMakeLists(std::string dir, std::string name, std::string timestamp) {
    // TODO.
}

}  // SOFIE.
}  // Experimental.
}  // TMVA.
