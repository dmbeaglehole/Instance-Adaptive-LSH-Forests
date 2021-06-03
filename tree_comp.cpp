#include <iostream>
#include "Eigen/Dense"
#include <vector>
#include "utils.h"
#include "mnist_loader.h"
#include "tree_comp.h"
#include <random>
#include <cmath> 
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <thread>
#include <omp.h>
#include <stdio.h>
#include "math.h"
#include <queue>
#include <chrono> 
#include <ctime> 

#define BUCKET_SIZE 5
#define TIME_HORIZON 20000
using namespace Eigen;

std::string dir_path = "";

/* write and read matrices to binary file */
template<class Matrix>
void write_binary(std::string filename, const Matrix& matrix){
    std::cout << "inside write binary " << filename << std::endl;
    std::ofstream out;
    out.open(filename.c_str(), std::ios::binary);
    std::cout << out.is_open() << std::endl;

    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}
template<class Matrix>
void read_binary(std::string filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
}



/* Bit MWU helper functions */
void compute_losses(const Ref<const ArrayXXf>& objective_vals, Ref<ArrayXf> losses, 
        QueryPair* query_strat_ptr, int dim) {
        

    //#pragma omp parallel for default(shared)
    for (int i=0; i<dim; i++) {
        //std::cout << "Computing losses for bit player on dim " << i << std::endl;
        if (query_strat_ptr->data_point(0,i) == query_strat_ptr->query(0,i)) {
            // note: if they agree bucket size is at least 1
            losses(i,0) = 1-objective_vals(i, query_strat_ptr->data_point(0,i));
        } else {
            losses(i,0) = 1; 
        }
        
    }

    /*
    printf("Losses\n");
    #pragma omp critical 
    std::cout << losses.transpose() << std::endl;
    */
}

void bit_MWU(const Ref<const ArrayXXf>& objective_vals, Ref<ArrayXf> bit_weights, 
        QueryPair* query_strat_ptr, int dim) {

    // agressive beta setting
    //double t1 = double(2*highestPowerof2(t+1));
    //double eps = std::min(0.5, std::sqrt((3/(t+1))));
    double T1 = 30;
    double eps = std::sqrt(3/T1);
    double beta = 1 - eps;

    ArrayXf losses = ArrayXf::Zero(dim);
    compute_losses(objective_vals, losses, query_strat_ptr, dim);
    bit_weights = bit_weights * pow(beta, losses);
    bit_weights /= bit_weights.sum(); // normalize
    
    
}


/* Query Blackbox helper functions */

float compute_min_query(const Ref<const ArrayXXf>& objective_vals, const Ref<const ArrayXf>& pi,
        const Ref<const ArrayXb>& datapoint, Ref<ArrayXb> query, int r, int dim) {
    
    ArrayXf weighted_obj_vals = ArrayXf::Zero(dim);

    for (int i=0; i<dim; i++) {
        // bucket size is at least 1
        weighted_obj_vals(i,0) = objective_vals(i, datapoint(0,i)); 
    }

    weighted_obj_vals *= pi;

    // sort objective values in descending order
    std::vector<float> vec(weighted_obj_vals.data(), weighted_obj_vals.data() + dim); // vectorize obj vals
    std::vector<int> sorted_indices(dim); // vector of indices
    int x=0;
    std::iota(sorted_indices.begin(), sorted_indices.end(),x++); //Initializing indices
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return vec.at(i)>vec.at(j);} );

    // used sorted obj values to construct worst query for worst datapoint
    int idx = -1;
    float total_obj = 0;
    for (int t=0; t<dim; t++) {
        idx = sorted_indices.at(t);
        if (t < r) {
            query(0,idx) = !datapoint(0,idx);
        } else {
            total_obj += weighted_obj_vals(idx,0); 
        }
    }

    return total_obj;
}

void query_argmin(const Ref<const ArrayXXb>& dataset, const Ref<const ArrayXXf>& objective_vals, 
        const Ref<const ArrayXf>& pi, QueryPair* query_strat_ptr, int r, int dim, int num_images) {

    // compute worst datapoint (requiring most hashes)
    float min_obj = INFINITY;
    int min_datapoint_idx = -1;

    #pragma omp parallel for default(shared) num_threads(38)
    for (int i=0; i<num_images ; i++) {

        /*
        if (i % 100 == 0) {
            #pragma omp critical 
            std::cout << "thread id for image " << i << " " << std::this_thread::get_id() << std::endl;
        }
        */

        ArrayXb current_datapoint = dataset.row(i);
        ArrayXb current_query = current_datapoint; 
        float current_obj;

        current_obj = compute_min_query(objective_vals, pi, current_datapoint, current_query, r, dim);

        if (current_obj < min_obj) {
            min_obj = current_obj;
            min_datapoint_idx = i;
        }
    }


    // used sorted obj values to construct worst query for worst datapoint
    query_strat_ptr->data_point = dataset.row(min_datapoint_idx);

    // create min query
    ArrayXb min_query = dataset.row(min_datapoint_idx); 
    compute_min_query(objective_vals, pi,
        query_strat_ptr->data_point, min_query, r, dim);


    query_strat_ptr->query = min_query; 
    query_strat_ptr->datapoint_idx = min_datapoint_idx;

    //print_image(query_strat_ptr->data_point, 28, 28);
    //print_image(query_strat_ptr->query, 28, 28);
}


// compute objective values (i.e. n_i^{rho_i} for i=1,...,d) 
void compute_objective_vals(const Ref<const ArrayXXb>& dataset, int c, int dim, 
        int num_images, Ref<ArrayXXf> objective_vals) {

    // compute bucket sizes
    for (int i=0; i<dim; i++) {
        objective_vals(i,1) = float(dataset.col(i).sum());
        objective_vals(i,0) = float(num_images) - objective_vals(i,1);
    }

    objective_vals = Eigen::pow(objective_vals, -1/float(c));
}

void write_query_to_file(const Ref<const ArrayXb>& datapoint, const Ref<const ArrayXb>& query, std::string query_file){
    std::ofstream file;
    file.open(query_file, std::ios_base::app);
    if (file.is_open()) {
        file << datapoint << std::endl;
        file << query << std::endl;
    }
    file.close();
}

/* MinMax Optimization main function */
void min_max_opt(Ref<ArrayXf> pi, int r, int c, Dataset* dataset_obj_ptr, 
    std::string node_file, std::string query_file) {

    omp_set_nested(1); // enable nested parallelism

    int dim = dataset_obj_ptr->get_size();
    int num_images = dataset_obj_ptr->get_num_images();
    int T = TIME_HORIZON; // T >= 16*ln(d)/eps^2
    ArrayXXb dataset = dataset_obj_ptr->get_binary_data();

    ArrayXf bit_weights = ArrayXf::Zero(dim) + (1/float(dim)); // initialize bit weights to uniform
    QueryPair query_strat = {dataset.row(0), dataset.row(0),0}; // initialize query arbitrarily
    QueryPair new_query_strat = query_strat;

    ArrayXXf objective_vals = ArrayXXf::Zero(dim,2);
    compute_objective_vals(dataset, c, dim, num_images, objective_vals);

    std::ofstream file;
    file.open(dir_path + "/regret/obj_vals.txt", std::ios_base::app);
    if (file.is_open()) {
        file << objective_vals << std::endl;
    }
    file.close();

    for (int t=0; t<T; t++) {

        if (t%200 == 0) {
            //#pragma omp critical 
            std::cout << "Game in round " << t << std::endl;
        }

        // time
        /*
        auto timenow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); 
        #pragma omp critical
        std::cout << std::ctime(&timenow) << std::endl;
        */

        // update distribution history
        pi *= float(t);
        pi += bit_weights;
        pi /= float(t+1);


        // blackboxes
        #pragma omp parallel num_threads(2) 
        {
            #pragma omp single 
            bit_MWU(objective_vals, bit_weights, &query_strat, dim);

            #pragma omp single 
            query_argmin(dataset, objective_vals, pi, &new_query_strat, r, dim, num_images);
        }

        query_strat = new_query_strat;

        write_query_to_file(query_strat.data_point, query_strat.query, query_file);
        write_dist_to_file(bit_weights, node_file);
        
        //#pragma omp critical
        //std::cout << "Query/pair distance is " << hamming(query_strat.data_point, query_strat.query) << std::endl;

    }

    

    std::string pifile = dir_path + "/pi_init.txt";

    if (!fexists(pifile.c_str())) {
        write_dist_to_file(pi, pifile);
    }
}

/* draw from a discrete distribution pi */
int draw_from_pi(const Ref<const ArrayXf>& pi) {
    //generate uniform(0,1) RV
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0, 1);
    float rand = dis(gen);

    float tot = 0;
    int dim = pi.size();


    for (int i=0; i<dim; i++) {
        tot += pi(i,0);
        if (tot > rand) {
            return i;
        }
    }
    return dim-1;
}

void compute_node(int r, int c, Dataset* dataset_ptr, LSHtree* tree_ptr, bool root, 
        std::string tree_label, const Ref<const ArrayXf>& pi_prev) {

    int dim = dataset_ptr->get_size();
    if ((dataset_ptr->get_num_images() <= dataset_ptr->get_stop_condition()) || 
        ((int)tree_ptr->get_used_hashes().size() >= dim)) {
        delete dataset_ptr;
        return;
    }

    std::string node_file = dir_path + "/regret/dists" + tree_ptr->get_label();
    std::string query_file = dir_path + "/regret/queries" + tree_ptr->get_label();
    ArrayXf pi = ArrayXf::Zero(dim);

    if (pi_prev.rows() > 1) {
        pi = pi_prev;
    } else {
        min_max_opt(pi, r, c, dataset_ptr, node_file, query_file);
    }


    Dataset* left_dataset_ptr = NULL;
    Dataset* right_dataset_ptr = NULL;
    HashFunc hash0;
    HashFunc hash1;

    int idx = draw_from_pi(pi);
    hash0 = {0, idx};
    hash1 = {1, idx};


    //recurse on two buckets
    left_dataset_ptr = dataset_ptr->hash_dataset(hash0);
    right_dataset_ptr = dataset_ptr->hash_dataset(hash1);

    if (!root) {
        delete dataset_ptr; // delete parent tree if not the root
    }

    
    LSHtree* left_tree = tree_ptr->add_hash(hash0);
    LSHtree* right_tree = tree_ptr->add_hash(hash1); 
    left_tree->set_label(tree_ptr->get_label() + "l");
    right_tree->set_label(tree_ptr->get_label() + "r");

    int smaller_bucket = std::min(left_dataset_ptr->get_num_images(), 
                        right_dataset_ptr->get_num_images());

    ArrayXf pi_copy;
    if (smaller_bucket == 0) {
        pi_copy = pi;
    } else {
        pi_copy = ArrayXf::Zero(1);
    }

    compute_node(r, c, left_dataset_ptr, left_tree, 0, tree_label, pi_copy);
    compute_node(r, c, right_dataset_ptr, right_tree, 0, tree_label, pi_copy);
}

void read_init_file(std::string filename, Ref<ArrayXf> pi_init) {
    std::ifstream file(filename, std::ios::in);
    std::string file_content;
    char c;
    while (file.get(c)) {
        file_content += c;
    }

    std::istringstream ss(file_content);
    std::vector<std::string> text;
  
    std::string word; // for storing each word 
  
    // Traverse through all words 
    // while loop till we get  
    // strings to store in string word 
    while (ss >> word)  
    { 
        text.push_back(word);
    } 

    for (int i=0; i<784; i++) {
        pi_init(i,0) = std::stof(text.at(i),NULL);
    }
} 


LSHtree* compute_tree(int r, int c, Dataset* dataset_ptr, std::string tree_label) {
    LSHtree* tree_ptr = new LSHtree;
    tree_ptr->set_label("o");
    ArrayXf pi_init;
    std::string pifile = dir_path + "/pi_init.txt";
    if (fexists(pifile.c_str())) {
        pi_init = ArrayXf::Zero(dataset_ptr->get_size());
        read_init_file(pifile, pi_init); 
    } else {
        printf("pi_init doesnt exist\n");
        pi_init = ArrayXf::Zero(1);
    }
    compute_node(r, c, dataset_ptr, tree_ptr, 1, tree_label, pi_init);
    return tree_ptr;
}


void write_tree_file(std::string filename, LSHtree* tree_ptr, std::ofstream& file) {
    if (tree_ptr) {
        if (!file) {
            throw "File not open! When writing tree file";
        }

        file << " { "; 

        std::vector<HashFunc> used_hashes = tree_ptr->get_used_hashes();

        if (used_hashes.size() > 0) {
            HashFunc recent_hash = used_hashes.back();
            std::string bit = (recent_hash.bit ? "1" : "0");
            std::string index = std::to_string(recent_hash.idx);
            file << "(" + bit + "," + index + ")";
        }

        write_tree_file(filename, tree_ptr->get_right(), file); 
        write_tree_file(filename, tree_ptr->get_left(), file); 

        file << " } "; 
        
    }
    return;
}


