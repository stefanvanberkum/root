/**
 * Graph convolution module using Eigen for sparse operations.
 * 
 * This module applies the graph convolution operation X = D^(-1/2) * A *
 * D^(-1/2) * X * Theta.
 * 
 * Its performance is currently worse than the regular GCNConv module, so it is
 * not made available to users.
*/

#ifndef RMODULE_SPARSEGCNCONV_H_
#define RMODULE_SPARSEGCNCONV_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"
#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseCore>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule_SparseGCNConv: public RModule {
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
        RModule_SparseGCNConv(std::string x, std::string edge_index, int in_features, int out_features, bool improved=false, bool add_self_loops=true, bool normalize=true, bool bias=true) {
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
        RModule_SparseGCNConv(std::string x, std::string edge_index, std::string edge_weight, int in_features, int out_features, bool improved=false, bool add_self_loops=true, bool normalize=true, bool bias=true) {
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
        ~RModule_SparseGCNConv() {};

        /**
         * Applies the graph convolution operation to each node.
         * 
         * @returns The updated feature matrix.
        */
        std::vector<float> forward() {
            std::vector<float> X = input_modules[0] -> getOutput();
            std::vector<float> edge_index_f = input_modules[1] -> getOutput();

            std::size_t n_nodes = X.size() / input_features;
            std::size_t n_edges = edge_index_f.size() / 2;
            
            std::vector<float> edge_weight;
            if (edge_weights) {
                std::vector<float> edge_weight_f = input_modules[2] -> getOutput();
            } else {
                edge_weight = std::vector<float>(n_edges, 1);
            }
            std::vector<float> degree;

            typedef Eigen::Triplet<float> T;
            std::vector<T> edge_list;
            edge_list.reserve(n_edges);
            for(std::size_t i = 0; i < n_edges; i++) {
                int source = edge_index_f[i];
                int target = edge_index_f[i + n_edges];
                edge_list.push_back(T(source, target, edge_weight[i]));
            }
            Eigen::SparseMatrix<float, Eigen::RowMajor> A(n_nodes, n_nodes);
            A.setFromTriplets(edge_list.begin(), edge_list.end());
            
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rMatrix;
            if (normalization) {
                if (self_loops) {
                    if (improve) {
                        A = A + 2 * rMatrix::Identity(n_nodes, n_nodes);
                        degree = std::vector<float>(n_nodes, 2);
                    } else {
                        A = A + rMatrix::Identity(n_nodes, n_nodes);
                        degree = std::vector<float>(n_nodes, 1);
                    }
                } else {
                    degree = std::vector<float>(n_nodes, 0);
                }

                // Loop through edges to get node degrees.
                for (std::size_t i = 0; i < n_edges; i++) {
                    int target = edge_index_f[i + n_edges];
                    degree[target] += edge_weight[i];
                }
            }

            Eigen::Map<Eigen::VectorXf> d(degree.data(), n_nodes);
            d = d.array().sqrt().inverse();
            auto D = d.asDiagonal();
            A = D * A * D;
            
            Eigen::Map<rMatrix> X_m(X.data(), n_nodes, input_features);
            Eigen::Map<rMatrix> theta(W.data(), output_features, input_features);
            rMatrix out = A * X_m * theta.transpose();

            if (include_bias) {
                Eigen::Map<Eigen::RowVectorXf> bias(b.data(), output_features);
                out = out.rowwise() + bias;
            }

            return std::vector<float>(out.data(), out.data() + n_nodes * output_features);
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
            return "SparseGCNConv";
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
            std::string del_string = "inc/modules/RModule_SparseGCNConv.hxx";
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

#endif  // RMODULE_SPARSEGCNCONV_H_
