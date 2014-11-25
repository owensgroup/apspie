// Reads in mtx files
//
// #_of_rows #_of_columns #_of_nnz
// start end edge_weight (optional)
// start end edge_weight (optional)
// etc.

#pragma once

#include <stdio.h>

class Graph {
private:
    int edges;
    int *coo;
public:
    int vertices;
    Graph();
    void read();    // Reads stdin and stores variables
    //void write(); // Writes stdout
};


