#ifndef __TREE_H
#define __TREE_H

#include <iostream>
#include "Eigen/Dense"
#include "mnist_loader.h"
#include "utils.h"
#include <random>
#include <vector>
#include <string>

using namespace Eigen;

LSHtree* compute_tree(int, int, Dataset*, std::string);
void write_tree_file(std::string, LSHtree*, std::ofstream&);

#endif
