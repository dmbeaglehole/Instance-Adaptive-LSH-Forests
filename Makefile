CXX      = g++-5
CXXFLAGS = -std=c++17 -g -Wall -Wextra -O3 -fopenmp -I/home/dmb2266/include 
OPTFLAGS = -march=native 

TGT   = tree_main 
SRCS  = mnist_loader.cpp, utils.cpp, tree_comp.cpp
BIN   = $(TGT)
OBJS  = $(SRCS:.cc=.o)

tree_main: mnist_loader.o utils.o tree_main.o tree_comp.o
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) mnist_loader.o utils.o tree_comp.o tree_main.o -o tree_main 

tree_comp.o: tree_comp.cpp
	$(CXX) $(CXXFLAGS) -c tree_comp.cpp -o tree_comp.o

mnist_loader.o: mnist_loader.cpp
	$(CXX) $(CXXFLAGS) -c mnist_loader.cpp -o mnist_loader.o

utils.o: utils.cpp
	$(CXX) $(CXXFLAGS) -c utils.cpp -o utils.o

treemain.o: tree_main.cpp 
	$(CXX) $(CXXFLAGS) -c tree_main.cpp -o tree_main.o

clean:
	rm *.o tree_main 
