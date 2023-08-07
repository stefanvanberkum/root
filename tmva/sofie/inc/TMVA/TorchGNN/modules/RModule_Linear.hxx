/**
 * Linear module.
 * 
 * This module applies a linear transformation (Ax + b) to the data. For
 * matrix inputs of shape (n_obs, n_features), it applies the transformation 
 * (XA^T + 1b^T).
*/

#ifndef TMVA_SOFIE_RMODULE_LINEAR_H_
#define TMVA_SOFIE_RMODULE_LINEAR_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"
#include <gsl/gsl_cblas.h>
#include <fstream>
#include <iostream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule_Linear: public RModule {
    public:
        /**
         * Construct the linear module.
         * 
         * @param x The input.
         * @param in_features The size of each input sample.
         * @param out_features The size of each output sample.
         * @param bias True if a bias is included. Defaults to true.
        */
        RModule_Linear(std::string x, int in_features, int out_features, bool bias=true) {
            input_features = in_features;
            output_features = out_features;
            include_bias = bias;

            if (!include_bias) {
                b = std::vector<float>(output_features);
            }

            inputs = {x};
            args = {std::to_string(in_features), std::to_string(out_features), std::to_string(bias)};
        }

        /** Destruct the module. */
        ~RModule_Linear() {};

        /**
         * Applies the linear transformation (y = Ax + b) to each element in the
         * input.
         * 
         * @returns Result (Ax + b) for each element.
        */
        std::vector<float> forward() {
            std::vector<float> in = input_modules[0] -> getOutput();
            std::vector<float> out;

            if (row_dim > 1) {
                // Perform matrix multiplications (Y = XA^T + 1b^T := XA^T + B).
                // X, shape (m, k) -> (row_dim, input_features).
                // A, shape (n, k) -> (output_features, input_features).
                // B, shape (m, n) -> (row_dim, output_features).
                int m = row_dim;
                int k = input_features;
                int n = output_features;
                for (std::size_t i = 0; i < in.size(); i += row_dim * input_features) {
                    // Get input matrix.
                    std::vector<float> X(in.begin() + i, in.begin() + i + row_dim * input_features);

                    std::vector<float> B(row_dim * output_features, 0); 
                    if (include_bias) {
                        // Construct bias matrix (B = 1b^T).
                        std::vector<float> one(row_dim, 1);
                        cblas_sger(CblasRowMajor, row_dim, output_features, 1, one.data(), 1, b.data(), 1, B.data(), output_features);
                    }

                    // Perform matrix multiplication (Y = XA^T + B).
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, X.data(), k, A.data(), k, 1, B.data(), n);
                    for (float elem: B) {  // cblas sets (B = XA^T + B).
                        out.push_back(elem);
                    }
                }
            } else {
                // Perform matrix-vector multiplications (y = Ax + b).
                // A, shape (n, m) -> (output_features, input_features).
                // x, shape (m, 1) -> (input_features, 1).
                // b, shape (n, 1) -> (output_features, 1).
                int m = input_features;
                int n = output_features;
                for (std::size_t i = 0; i < in.size(); i += input_features) {
                    // Get input vector.
                    std::vector<float> x(in.begin() + i, in.begin() + i + input_features);

                    // Copy bias.
                    std::vector<float> y(b);

                    // Perform matrix-vector multiplication (y = Ax + b).
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, A.data(), m, x.data(), 1, 1, y.data(), 1);
                    for (float elem: y) {  // cblas sets (b = Ax + b).
                        out.push_back(elem);
                    }
                }
            }
            return out;
        }

        /**
         * Infer the output shape.
         * 
         * For this module, the output shape is the same as the input shape but
         * with out_features on the last dimension instead of in_features.
         * 
         * @returns The output shape.
        */
        std::vector<int> inferShape() {
            std::vector<int> shape = input_modules[0] -> getShape();
            shape.back() = output_features;

            if (shape.size() > 1) {
                row_dim = shape[shape.size() - 2];
            }
            return shape;
        }

        /**
         * Get the operation.
         * 
         * @returns The name of the operation.
        */
        std::string_view getOperation() {
            return "Linear";
        }

        /**
         * Set the weights.
         * 
         * @param weights The weight matrix.
        */
        void setWeights(std::vector<float> weights) {A = weights;}

        /**
         * Set the biases.
         * 
         * @param biases The bias vector.
        */
        void setBiases(std::vector<float> biases) {b = biases;}

        /** 
         * Save parameters.
         * 
         * @param dir Save directory.
         */
        void saveParameters(std::string dir) {
            // Save weights.
            std::string fdir = dir + "/" + name + "_weight.dat";
            std::ofstream outfile = std::ofstream(fdir, std::ios::out | std::ios::binary);
            outfile.write(reinterpret_cast<char*>(&A[0]), A.size() * sizeof(float));
            
            if (include_bias) {
                // Save biases.
                fdir = dir + "/" + name + "_bias.dat";
                outfile = std::ofstream(fdir, std::ios::out | std::ios::binary);
                outfile.write(reinterpret_cast<char*>(&b[0]), b.size() * sizeof(float));
            }
            outfile.close();
        }

        /**
         * Load saved parameters.
        */
        void loadParameters() {
            std::string dir = __FILE__;
            std::string del_string = "inc/modules/RModule_Linear.hxx";
            dir.replace(dir.find(del_string), del_string.size(), "params/");

            // Load weights.
            std::string param_dir = dir + name + "_weight.dat";
            std::ifstream infile = std::ifstream(param_dir, std::ios::in | std::ios::binary);
            A = std::vector<float>(input_features * output_features);
            infile.read(reinterpret_cast<char*>(&A[0]), A.size() * sizeof(float));
            infile.close();

            if (include_bias) {
                // Load biases.
                param_dir = dir + name + "_bias.dat";
                infile = std::ifstream(param_dir, std::ios::in | std::ios::binary);
                b = std::vector<float>(output_features);
                infile.read(reinterpret_cast<char*>(&b[0]), b.size() * sizeof(float));
                infile.close();
            }
        }

        /**
         * Load parameters from PyTorch state dictionary.
         * 
         * @param state_dict The state dictionary.
        */
        void loadParameters(std::map<std::string, std::vector<float>> state_dict) {
            if (auto search = state_dict.find(name + ".weight"); search != state_dict.end()) {
                A = state_dict[name + ".weight"];
            } else {
                std::cout << "WARNING: Weights for module " << name << " not found." << std::endl;
            }
            
            if (include_bias) {
                if (auto search = state_dict.find(name + ".weight"); search != state_dict.end()) {
                    b = state_dict[name + ".bias"];
                } else {
                    std::cout << "WARNING: Biases for module " << name << " not found." << std::endl;
                }
            }
        }
    private:
        int input_features;  // The size of each input sample.
        int output_features;  // The size of each output sample.
        bool include_bias;  // True if a bias is included.
        int row_dim = 1;  // Size of the second to last input dimension.
        std::vector<float> A;  // Weight matrix A.
        std::vector<float> b;  // Bias vector b.
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RMODULE_LINEAR_H_
