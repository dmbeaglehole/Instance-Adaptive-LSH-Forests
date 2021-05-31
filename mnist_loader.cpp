#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "mnist_loader.h"
#include "utils.h"
#include "Eigen/Dense"
#include "Eigen/Core"
#include <numeric>
#include <algorithm>

using namespace Eigen;
#define BUCKET_SIZE 5
#define FRAC 80 // 60000/FRAC images

void Dataset::init_data() {
    binary_data = ArrayXXb::Zero(num_images, img_size);
    data = ArrayXXc::Zero(num_images, img_size);
}

void Dataset::set_size(int size){
    img_size = size;
}
int Dataset::get_size() {
    return img_size;
}
void Dataset::set_thresh(unsigned char t) {
    thresh = t;
}
unsigned char Dataset::get_thresh() {
    return thresh;
}

ArrayXXb Dataset::get_binary_data()
{
    return binary_data;
}

short int apply_threshold(unsigned char pixel, unsigned char threshold) {
    return (short int) ((pixel < threshold) ? 0 : 1);
}

void Dataset::binarize_data() {
    for (int i = 0; i<num_images; i++) {
        for (int j = 0; j<img_size; j++) {
            binary_data(i,j) = apply_threshold(data(i,j), thresh);
        }
    }
}

void Dataset::set_num_rows(int n_rows) {
    num_rows = n_rows;
}

void Dataset::set_num_cols(int n_cols) {
    num_cols = n_cols;
}

void Dataset::set_num_images(int n_images) {
    num_images = n_images;
}

void Dataset::set_data(ArrayXXc dataset) {
    data = dataset;
}

ArrayXXc Dataset::get_data() {
    return data;
}

int Dataset::get_num_rows() {
    return num_rows;
}

int Dataset::get_num_cols() {
    return num_cols;
}

int Dataset::get_num_images() {
    return num_images;
}

void Dataset::set_binary_data(ArrayXXb dataset) {
    binary_data = dataset;
}

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
void Dataset::read_mnist(std::string file_path)
{
    std::ifstream file (file_path);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        // load magic number
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        
        // load number of imgs 
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        /*
        // load number of rows
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        // load number of cols
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        */

        n_cols = 1;
        n_rows = 1;


        num_cols = n_cols;
        num_rows = n_rows;
        num_images = number_of_images/FRAC; // reduced size
        img_size = n_cols * n_rows;

        init_data();

        for(int i=0;i<num_images;++i)
        {
            for(int j=0;j<img_size;++j)
            {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                data(i,j) = temp;

            }
        }
    }
    else {
        std::cout << "Throwing user exception" << std::endl;
        throw file_path + " not open!";
        exit(EXIT_FAILURE);
    }

}


Dataset* Dataset::hash_dataset(HashFunc hashf) {
/* Return dataset object with dataset bucketed on hash function
*/
    Dataset* dataset_obj_ptr = new Dataset;
    std::vector<int> ri;
    for (int i=0;i<num_images;i++) {
        int tmp = binary_data(i,hashf.idx);
        if (tmp == hashf.bit) {
            ri.push_back(i);
        }
    }
    
    dataset_obj_ptr->set_size(img_size);
    dataset_obj_ptr->set_num_images(ri.size());
    dataset_obj_ptr->init_data();
    ArrayXXb sliced_data = ArrayXXb::Zero(dataset_obj_ptr->get_num_images(), img_size);
    slice_on_rows(sliced_data, binary_data, ri, img_size);
    dataset_obj_ptr->set_binary_data(sliced_data);
    return dataset_obj_ptr;
}


Dataset::Dataset(){
    // empty constructor
}


int Dataset::stopping_size = BUCKET_SIZE;

int Dataset::get_stop_condition() {
    return stopping_size;
}
