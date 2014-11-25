// Tests Graph Class

#include <stdio.h>
#include <graph.h>

// Test read() of graph.h
void testGraphRead( void ) 
{
    Graph A;
    fgets("5 5 4\n1 2\n3 4\n3 5\n3 3", 100, stdin);
    A.read();
    assert A.nnz = 4;
}
