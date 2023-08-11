/**
 * Graph convolution module.
 * 
 * This module applies the graph convolution operation X = D^(-1/2) * A *
 * D^(-1/2) * X * Theta.
*/

#ifndef TMVA_SOFIE_RMODULE_GCNCONV_H_
#define TMVA_SOFIE_RMODULE_GCNCONV_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"
#include <gsl/gsl_cblas.h>
#include <cmath>
#include <fstream>
#include <iostream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule_GCNConv: public RModule {
    public:
        /**
         * Construct the graph convolution module without edge weights.
         * 
         * @param x The input feature matrix of shape (n_nodes, in_features).
         * @param edge_index The edge indices matrix of shape (2, n_edges).
         * @param in_features The size of each input sample.
         * @param out_features The size of each output sample.
         * @param improved True if self-loops should have a weight of two (A = A +
         * 2I). Defaults to false.
         * @param add_self_loops True if self-loops should be added (A = A + I).
         * Defaults to true.
         * @param normalize True if self-loops should be added and symmetric
         * normalization should be computed on the fly.
         * @param bias True if a bias is included. Defaults to true.
        */
        RModule_GCNConv(std::string x, std::string edge_index, int in_features, int out_features, bool improved=false, bool add_self_loops=true, bool normalize=true, bool bias=true) {
            input_features = in_features;
            output_features = out_features;
            improve = improved;
            self_loops = add_self_loops;
            normalization = normalize;
            include_bias = bias;
            edge_weights = false;

            if (!include_bias) {
                b = std::vector<float>(output_features);
            }

            inputs = {x, edge_index};
            args = {std::to_string(in_features), std::to_string(out_features), std::to_string(improved), std::to_string(add_self_loops), std::to_string(normalize), std::to_string(bias)};
        }

        /**
         * Construct the graph convolution module with edge weights.
         * 
         * @param x The input feature matrix of shape (n_nodes, in_features).
         * @param edge_index The edge indices matrix of shape (2, n_edges).
         * @param edge_weight The edge weights vector of shape (n_edges).
         * @param in_features The size of each input sample.
         * @param out_features The size of each output sample.
         * @param improved True if self-loops should have a weight of two (A = A +
         * 2I). Defaults to false.
         * @param add_self_loops True if self-loops should be added (A = A + I).
         * Defaults to true.
         * @param normalize True if self-loops should be added and symmetric
         * normalization should be computed on the fly.
         * @param bias True if a bias is included. Defaults to true.
        */
        RModule_GCNConv(std::string x, std::string edge_index, std::string edge_weight, int in_features, int out_features, bool improved=false, bool add_self_loops=true, bool normalize=true, bool bias=true) {
            input_features = in_features;
            output_features = out_features;
            improve = improved;
            self_loops = add_self_loops;
            normalization = normalize;
            include_bias = bias;
            edge_weights = true;

            if (!include_bias) {
                b = std::vector<float>(output_features);
            }

            inputs = {x, edge_index, edge_weight};
            args = {std::to_string(in_features), std::to_string(out_features), std::to_string(improved), std::to_string(add_self_loops), std::to_string(normalize), std::to_string(bias)};
        }

        /** Destruct the module. */
        ~RModule_GCNConv() {};

        /**
         * Applies the graph convolution operation to each node.
         * 
         * @returns The updated feature matrix.
        */
        std::vector<float> forward() {
            std::vector<float> X = input_modules[0] -> getOutput();
            std::vector<float> edge_index_f = input_modules[1] -> getOutput();
            std::vector<int> edge_index(edge_index_f.begin(), edge_index_f.end());

            std::size_t n_nodes = X.size() / input_features;
            std::size_t n_edges = edge_index.size() / 2;
            
            std::vector<float> edge_weight;
            if (edge_weights) {
                std::vector<float> edge_weight_f = input_modules[2] -> getOutput();
            } else {
                edge_weight = std::vector<float>(n_edges, 1);
            }
            std::vector<float> out;
            out.reserve(n_nodes * output_features);
            std::vector<float> X_agg;
            std::vector<float> degree;
            
            if (normalization) {
                if (self_loops) {
                    if (improve) {
                        degree = std::vector<float>(n_nodes, 2);
                    } else {
                        degree = std::vector<float>(n_nodes, 1);
                    }
                } else {
                    degree = std::vector<float>(n_nodes, 0);
                }

                // Loop through edges to get node degrees.
                for (std::size_t i = 0; i < n_edges; i++) {
                    int target = edge_index[i + n_edges];
                    degree[target] += edge_weight[i];
                }
            }

            if (normalization && self_loops) {
                // Add self loops.
                X_agg = X;
                int self_weight;
                if (improve) {
                    self_weight = 2;
                } else {
                    self_weight = 1;
                }
                for (std::size_t i = 0; i < n_nodes; i++) {
                    for (int j = 0; j < input_features; j++) {
                        X_agg[i * input_features + j] *= self_weight / degree[i];
                    }
                }
            } else {
                // Set X_agg to zero.
                X_agg = std::vector<float>(X.size());
            }

            // Loop through the edges to aggregate information from neighboring
            // nodes.
            for (std::size_t i = 0; i < n_edges; i++) {
                int source = edge_index[i];
                int target = edge_index[i + n_edges];
                
                int x_start = source * input_features;
                int x_agg_start = target * input_features;
                for (int j = 0; j < input_features; j++) {
                    if (normalization) {
                        X_agg[x_agg_start + j] += edge_weight[i] / sqrt(degree[source] * degree[target]) * X[x_start + j];
                    } else {
                        X_agg[x_agg_start + j] += edge_weight[i] * X[x_start + j];
                    }
                }
            }

            // Update node features.
                // Perform matrix multiplications (Y = X_agg * W^T + 1b^T :=
                // X_agg W^T + B).
                    // X_agg, shape (m, k) -> (n_nodes, input_features).
                    // W, shape (n, k) -> (output_features, input_features).
                    // B, shape (m, n) -> (n_nodes, output_features).
            int m = n_nodes;
            int k = input_features;
            int n = output_features;

            std::vector<float> B(n_nodes * output_features, 0); 
            if (include_bias) {
                // Construct bias matrix (B = 1b^T).
                std::vector<float> one(n_nodes, 1);
                cblas_sger(CblasRowMajor, n_nodes, output_features, 1, one.data(), 1, b.data(), 1, B.data(), output_features);
            }

            // Perform matrix multiplication (Y = X_agg * W^T + B).
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, X_agg.data(), k, W.data(), k, 1, B.data(), n);
            for (float elem: B) {  // cblas sets (B = X_agg * W^T + B).
                out.push_back(elem);
            }
            return out;
        }

        /**
         * Infer the output shape.
         * 
         * For this module, the output shape is the same as the shape of the
         * input feature matrix but with out_features on the last dimension 
         * instead of in_features.
         * 
         * @returns The output shape.
        */
        std::vector<int> inferShape() {
            std::vector<int> shape = input_modules[0] -> getShape();
            shape.back() = output_features;
            return shape;
        }

        /**
         * Get the operation.
         * 
         * @returns The name of the operation.
        */
        std::string_view getOperation() {
            return "GCNConv";
        }

        /**
         * Set the weights.
         * 
         * @param weights The weight matrix.
        */
        void setWeights(std::vector<float> weights) {W = weights;}

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
            std::string fdir = dir + "/" + name + "_lin_weight.dat";
            std::ofstream outfile = std::ofstream(fdir, std::ios::out | std::ios::binary);
            outfile.write(reinterpret_cast<char*>(&W[0]), W.size() * sizeof(float));
            
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
            std::string del_string = "inc/modules/RModule_GCNConv.hxx";
            dir.replace(dir.find(del_string), del_string.size(), "params/");

            // Load weights.
            std::string param_dir = dir + name + "_lin_weight.dat";
            std::ifstream infile = std::ifstream(param_dir, std::ios::in | std::ios::binary);
            W = std::vector<float>(input_features * output_features);
            infile.read(reinterpret_cast<char*>(&W[0]), W.size() * sizeof(float));
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
            if (auto search = state_dict.find(name + ".lin.weight"); search != state_dict.end()) {
                W = state_dict[name + ".lin.weight"];
            } else {
                std::cout << "WARNING: Weights for module " << name << " not found." << std::endl;
            }
            
            if (include_bias) {
                if (auto search = state_dict.find(name + ".bias"); search != state_dict.end()) {
                    b = state_dict[name + ".bias"];
                } else {
                    std::cout << "WARNING: Biases for module " << name << " not found." << std::endl;
                }
            }
        }
    private:
        int input_features;  // The size of each input sample.
        int output_features;  // The size of each output sample.
        bool edge_weights;  // True if edge weights are provided.
        bool improve;  // True if self-loops should get a weight of two.
        bool self_loops;  // True if self-loops should be added.
        bool normalization;  // True if edge weights should be normalized.
        bool include_bias;  // True if a bias is included.
        std::vector<float> W;  // Weight matrix W.
        std::vector<float> b;  // Bias vector b.
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RMODULE_GCNCONV_H_
