#include "utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "Eigen/Dense"
#include <string>

using namespace Eigen; 

float comp_succ(float rho, int ri, int dim, const Ref<const ArrayXf>& balances) { 
    float r = float(ri);
    float d = float(dim);
    float obj_sum = Eigen::pow(balances, -rho).sum();
    return (1 - (r/d))*(1/d)*obj_sum;
}

void compute_balances(Ref<ArrayXf> balances, const Ref<const ArrayXXb>& binary_data, int dim) {
    float n = float(binary_data.rows());
    for (int i=0; i<dim; i++) {
        float ones = float(binary_data.col(i).sum());
        balances(i,0) = std::max(ones,n - ones)/n;
    }
}

float compute_nice_rho(int r, int img_size, const Ref<const ArrayXXb>& binary_data) {
    // compute rho for uniform dist assuming dataset is independent coordinates
    float upper = 1;
    float lower = 0;
    float thresh = 1e-4;
    
    float mid = (upper + lower)/2;
    int dim = img_size;

    ArrayXf balances = ArrayXf::Zero(dim);
    compute_balances(balances, binary_data, dim);

    while (std::abs(comp_succ(mid, r, dim, balances) - 1) > thresh) {
        if (std::abs(mid - 1) < thresh) {
            return mid;
        }
        
        if (comp_succ(mid, r, dim, balances) > 1) {
            upper = mid;
        } else {
            lower = mid;
        }
        mid = (upper + lower)/2;
    }

    return mid;
}

int highestPowerof2(int n) 
{ 
    int res = 0; 
    for (int i=n; i>=1; i--) 
    { 
        // If i is a power of 2 
        if ((i & (i-1)) == 0) 
        { 
            res = i; 
            return res;
        } 
    } 
    return res; 
} 

int hamming(const Ref<const ArrayXb>& d1 , const Ref<const ArrayXXb>& d2) {
    return (d1 - d2).abs().sum();
}

void flip_dataset(const Ref<const ArrayXXb>& dataset, Ref<ArrayXXb> flipped_dataset) {
    int r = dataset.rows();
    int c = dataset.cols();
    for (int i=0; i<r; i++) {
        for (int j=0; j<c; j++) {
            flipped_dataset(i,j) = !dataset(i,j);
        }
    }
}

void write_dist_to_file(const Ref<const ArrayXf>& pi, std::string node_file) {
    std::ofstream file;
    file.open(node_file, std::ios_base::app);
    if (file.is_open()) {
        file << pi.transpose() << std::endl;
    }
    file.close();
}

bool fexists(const char *filename)
{
  std::ifstream ifile;
  ifile.open(filename);
  if(ifile) {
      ifile.close();
      return 1;
  } else {
      return 0;
  }
}

void slice_on_rows(Ref<ArrayXXb> new_dataset, const Ref<const ArrayXXb>& binary_data, std::vector<int>& ri, int num_cols) {
    int num_rows = ri.size();
    for (int i=0; i<num_rows;i++) {
        for (int j=0; j<num_cols;j++) {
            new_dataset(i,j) = binary_data(ri.at(i),j);
        }
    }
}


LSHtree::LSHtree() {
    //cout << "Constructor called" << endl;
    ltree = NULL;
    rtree = NULL;
}

std::string LSHtree::get_label() {
    return label;
}

void LSHtree::set_label(std::string new_label) {
    label = new_label;
}

void LSHtree::append_hash(HashFunc new_hash) {
    used_hashes.push_back(new_hash);
}

LSHtree* LSHtree::add_hash(HashFunc new_hash) {
    LSHtree* new_tree_ptr = new LSHtree;
    new_tree_ptr->used_hashes = used_hashes;
    new_tree_ptr->append_hash(new_hash);
    if (new_hash.bit==1) {
        rtree = new_tree_ptr;
    } 
    else {
        ltree = new_tree_ptr;
    }
    return new_tree_ptr; 
}

std::vector<HashFunc> LSHtree::get_used_hashes() {
    return used_hashes;
}
    
void LSHtree::set_left(LSHtree* left){
    ltree = left;
}

void LSHtree::set_right(LSHtree* right){
    rtree = right;
}

LSHtree* LSHtree::get_right(){
    return rtree;
}

LSHtree* LSHtree::get_left(){
    return ltree;
}

void LSHtree::DestroyRecursive(LSHtree* tree){
    if (tree) {
        LSHtree::DestroyRecursive(tree->ltree);
        LSHtree::DestroyRecursive(tree->rtree);
        delete tree;
    }
}

LSHtree::~LSHtree(){
    DestroyRecursive(ltree);
    DestroyRecursive(rtree);
}