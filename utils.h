#ifndef __UTILS_H
#define __UTILS_H

#include <vector> 
#include "Eigen/Dense"
#include <iostream> 

using namespace Eigen;

typedef Array<unsigned char,Dynamic,Dynamic> ArrayXXc;
typedef Array<unsigned char,1,Dynamic> ArrayXc;

typedef Array<short int,Dynamic,Dynamic> ArrayXXb;
typedef Array<short int,1,Dynamic> ArrayXb;

void slice_on_rows(Ref<ArrayXXb>, const Ref<const ArrayXXb>&, std::vector<int>&, int);
float compute_nice_rho(int , int, const Ref<const ArrayXXb>&);
bool fexists(const char *);
void write_dist_to_file(const Ref<const ArrayXf>&, std::string);
void flip_dataset(const Ref<const ArrayXXb>&, Ref<ArrayXXb>);
int hamming(const Ref<const ArrayXb>&, const Ref<const ArrayXXb>&);
int highestPowerof2(int);

struct HashFunc {
    // container for a hash function
    bool bit; // bucket chosen on division
    int idx; // index divided on
};

struct QueryPair {
    ArrayXb query;
    ArrayXb data_point;
    int datapoint_idx;
};
    
    
class LSHtree {
    
    private:
    // variables 
        LSHtree* ltree; // 0 bit
        LSHtree* rtree; // 1 bit
        std::vector<HashFunc> used_hashes;
        std::string label;
    
    public:
    // methods
        LSHtree();
        LSHtree* add_hash(HashFunc);
        void append_hash(HashFunc);
        std::vector<HashFunc> get_used_hashes();
        void set_left(LSHtree*);
        void set_right(LSHtree*);
        LSHtree* get_right();
        LSHtree* get_left();
        std::string get_label();
        void set_label(std::string);
        void DestroyRecursive(LSHtree*);
        ~LSHtree();
    
};

#endif