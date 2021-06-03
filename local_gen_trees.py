import os

num_trees = 1
for t in range(num_trees):
    tree_label = "tree" + str(t)
    dir_path = "../mnist_v1/trees/" + tree_label + "/"
    os.system("mkdir " + dir_path)
    os.system("mkdir " + dir_path + "dists/")
    os.system("mkdir " + dir_path + "hard_queries/")


    os.system("./tree_main " + str(t))
