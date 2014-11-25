// Reads in mtx files
//
// #_of_rows #_of_columns #_of_nnz
// start end edge_weight (optional)
// start end edge_weight (optional)
// etc.

#include <stdio.h>
#include <graph.h>

Graph::Graph( void ) {
    printf("A graph is being created.\n");
}

// Reads stdin into graph.
// Assumes stdin is processed to erase comments
void Graph::read() {
    int N;

    scanf("%d", &N);
    vertices = N;
    scanf("%d", &N);
    scanf("%d", &N);
    edges = N;

    while( scanf("%d", &N) != EOF ) {
        printf("%d\n", N);
    }
}
