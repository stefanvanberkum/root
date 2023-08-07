/**
 * Concatenation module.
*/

#ifndef TMVA_SOFIE_RMODULE_CAT_H_
#define TMVA_SOFIE_RMODULE_CAT_H_

#include "TMVA/TorchGNN/modules/RModule.hxx"
#include <gsl/gsl_cblas.h>
#include <stdexcept>

// TODO: Remove.
#include <iostream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModule_Cat: public RModule {
    public:
        /**
         * Construct the concatenation module.
         * 
         * This will concatenate a and b along the specified dimension. The
         * inputs need to have the same shape along every axis except the
         * concatenation dimension.
         * 
         * @param a The first argument.
         * @param b The second argument.
         * @param dim Dimension along which to concatenate.
        */
        RModule_Cat(std::string a, std::string b, int dim) {
            cdim = dim;
            
            inputs = {a, b};
            args = {std::to_string(dim)};
        }

        /** Destruct the module. */
        ~RModule_Cat() {};

        /**
         * Concatenate the inputs.
         * 
         * @returns The concatenated array.
        */
        std::vector<float> forward() {
            std::vector<float> a = input_modules[0] -> getOutput();
            std::vector<float> b = input_modules[1] -> getOutput();
            std::shared_ptr<std::vector<float>> out = std::make_shared<std::vector<float>>();
            recursive_cat(a, b, {}, out);
            return *out;
        }

        /**
         * Infer the output shape.
         * 
         * For this module, the output shape is the shape of input a, extended
         * along the concatenation dimension with input b.
         * 
         * @returns The output shape.
        */
        std::vector<int> inferShape() {
            shape_a = input_modules[0] -> getShape();
            shape_b = input_modules[1] -> getShape();

            // Check shapes.
            for (std::size_t i = 0; i < shape_a.size(); i++) {
                if (i != cdim && shape_a[i] != shape_b[i]) {
                    std::vector<int> expected_shape = shape_a;
                    expected_shape[cdim] = -1;
                    std::string ex_s_str = "[" + std::to_string(expected_shape[0]);
                    std::string b_s_str = "[" + std::to_string(shape_b[0]);
                    for (std::size_t j = 1; j < expected_shape.size(); j++) {
                        ex_s_str += ", " + std::to_string(expected_shape[j]);
                        b_s_str += ", " + std::to_string(shape_b[j]);
                    }
                    ex_s_str += "]";
                    b_s_str += "]";
                    throw std::invalid_argument("Incompatible shapes in concatenation layer " + std::string(getName()) + ". Expected shape " + ex_s_str + ", got shape " + b_s_str + ".");
                }
            }

            std::vector<int> shape = shape_a;
            shape[cdim] += shape_b[cdim];
            dims = shape;
            return shape;
        }

        /**
         * Get the operation.
         * 
         * @returns The name of the operation.
        */
        std::string_view getOperation() {
            return "Cat";
        }

        /** 
         * Save parameters.
         * 
         * Does nothing for this module.
         * 
         * @param dir Save directory.
         */
        void saveParameters([[maybe_unused]] std::string dir) {}

        /**
         * Load saved parameters.
         * 
         * Does nothing for this module.
        */
        void loadParameters() {}

        /**
         * Load parameters from PyTorch state dictionary.
         * 
         * Does nothing for this module.
         * 
         * @param state_dict The state dictionary.
        */
        void loadParameters([[maybe_unused]] std::map<std::string, std::vector<float>> state_dict) {}
    private:
        /**
         * Recursively concatenate the inputs along a prespecified dimension.
         * 
         * The alorithm recursively loops through the dimensions, up until it
         * hits the concatenation dimension (cdim). At this point, it appends all 
         * elements of a and b at the current position to out, by consecutively 
         * looping through the remaining dimensions in a and b. The current 
         * position is stored in inds. Whenever the last index in inds exceeds the 
         * desired output dimension, it is popped from the array and the preceding 
         * index is incremented. When this is not the case and inds has not 
         * reached the concatenation dimension yet, we append a new dimension and 
         * set its index to zero. The algorithm terminates when all elements in 
         * inds are popped (i.e., when the list is empty again).
         * 
         * @param a The first input.
         * @param b The second input.
         * @param inds The current position of the algorithm. To start the
         * algorithm, this should be an empty vector.
         * @param out The output.
        */
        void recursive_cat(std::vector<float> a, std::vector<float> b, std::vector<int> inds, std::shared_ptr<std::vector<float>> out) {
            if (inds.size() == 0) {
                // We are at the start of the algorithm.
                if (cdim == 0) {
                    // The concatenation dimension is zero, so just append
                    // everything.
                    for (float elem: a) {
                        out -> push_back(elem);
                    }
                    for (float elem: b) {
                        out -> push_back(elem);
                    }
                    return;
                } else {
                    // Add first dimension to inds.
                    inds.push_back(0);
                }
            } else if (inds.back() >= dims[inds.size() - 1]) {
                // The last index is at the desired output dimension.
                if (inds.size() == 1) {
                    // Popping would empty the list, so we are done.
                    return;
                } else {
                    // Pop the last element from inds and increment the
                    // preceding index.
                    inds.pop_back();
                    inds.back()++;
                }
            } else if (inds.size() == cdim) {
                // Inds has reached the concatenation dimension, so add the
                // elements of a and b at this position.
                for (int i = 0; i < shape_a[cdim]; i++) {
                    append(a, shape_a[cdim], i, inds, out);
                }
                for (int i = 0; i < shape_b[cdim]; i++) {
                    append(b, shape_b[cdim], i, inds, out);
                }
                inds.back()++;
            } else if (inds.size() < cdim) {
                // Concatenation dimension is not reached yet, so add new
                // dimension to inds.
                inds.push_back(0);
            } else {
                throw std::runtime_error("Error in concatenation layer.");
            }
            recursive_cat(a, b, inds, out);
        }

        /**
         * Append all elements of x at the current position to out.
         * 
         * @param x The input vector.
         * @param c_shape The shape of x at the concatenation dimension.
         * @param c_count The current position in the concatenation dimension
         * for this input.
         * @param inds The current position of the algorithm.
         * @param out The output.
        */
        void append(std::vector<float> x, int c_shape, int c_count, std::vector<int> inds, std::shared_ptr<std::vector<float>> out) {
            // Find the remaining dimensions after the concatenation dimension.
            std::vector<int> remaining_dims = std::vector<int>(dims.begin() + inds.size() + 1, dims.end());

            // Determine the number of elements to be added.
            int n_add = 1;
            for (int i: remaining_dims) {
                n_add *= i;
            }

            // Find starting index, noting that at each addition operation, the
            // same number of elements is added (n_add).
            int start = 0;
            for (std::size_t i = 0; i < inds.size(); i++) {

                // Compute number of append loops that are performed for one
                // full "round" over this index.
                int rest = 1;
                for (std::size_t j = i + 1; j < inds.size(); j++) {
                    rest *= dims[j];
                }
                
                // Compute the number of append loops for the given number of
                // "rounds".
                start += inds[i] * rest;
            }
            start *= c_shape * n_add;  // Number of appended elements before this append loop.
            start += c_count * n_add;  // Number of appended elements in the current append loop.

            // Append the elements to the output.
            for (int i = start; i < start + n_add; i++) {
                out -> push_back(x[i]);
            }
        }
    
        std::size_t cdim;  // Concatenation dimension.
        std::vector<int> shape_a;  // Shape of input a.
        std::vector<int> shape_b;  // Shape of input b.
        std::vector<int> dims;  // Output dimensions.     
};

}  // TMVA.
}  // Experimental.
}  // SOFIE.

#endif  // TMVA_SOFIE_RMODULE_CAT_H_
