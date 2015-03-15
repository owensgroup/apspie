// Test all the components
//

#define NDEBUG

#include <stdio.h>
#include <graph.h>
#include <assert.h>

// Test read() of graph.h
void testGraphRead( void ) {
    Graph A;
    stdin = "5 5 4\n1 2\n3 4\n3 5\n3 3";
    A.read();
    assert( A.vertices == 4 );
};

int main() {
//    freopen("sample.in","r",stdin);
//    freopen("sample.out","w",stdout);
    
    testGraphRead();

    return 0;
}


