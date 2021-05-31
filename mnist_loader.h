#ifndef __MNIST_H
#define __MNIST_H

#include <vector>
#include <iostream> 
#include <string>
#include "utils.h"
#include "Eigen/Dense"

using namespace Eigen;
    
class Dataset {
    private: 
        ArrayXXc data; 
        ArrayXXb binary_data; 
        int num_rows;
        int num_cols;
        int img_size;
        int num_images;
        unsigned char thresh;
        static int stopping_size;
    public:
        Dataset();
        static int get_stop_condition();
        void init_data();
        void set_thresh(unsigned char);
        void set_num_rows(int);
        void set_num_cols(int);
        void set_num_images(int);
        void set_size(int);
        void set_data(ArrayXXc);
        void set_binary_data(ArrayXXb);
        ArrayXXc get_data();
        unsigned char get_thresh();
        int get_num_rows();
        int get_num_cols();
        int get_num_images();
        int get_size();
        ArrayXXb get_binary_data();
        void binarize_data();
        Dataset* hash_dataset(HashFunc);
        void read_mnist(std::string);
        float compute_nice_rho(int);
};

int reverseInt(char*);



#endif