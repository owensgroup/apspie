#include <coo.cuh>
#include <queue>
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

#define MARK_PREDECESSORS 0

using namespace boost;

class CompareDist {
public:
   bool operator() ( const std::pair<int, float>& lhs, const std::pair<int, float>& rhs ) const {
       return lhs.second > rhs.second;
   }
};

template<typename T> void print_queue(T& q, int m) {
   int count = 0;
   std::pair<int, float> Edge;
   while(!q.empty() && count<m ) {
       //Edge = q.top();
       printf("[%d]: %f ", q.top().first, q.top().second);
       q.pop();
       count++;
   }
   printf("\n");
}

// A simple CPU-based reference SSSP ranking implementation
template<typename VertexId, typename Value>
int SimpleReferenceSssp(
   const int m, const VertexId *h_rowPtrA, const VertexId *h_colIndA, const Value *h_csrValA,
   Value                                   *source_path,
   VertexId                                *predecessor,
   VertexId                                src,
   VertexId                                stop)
{
   typedef std::pair<VertexId, Value> Edge;

   // Initialize queue for managing previously-discovered nodes
   std::priority_queue<std::pair<VertexId, Value>, std::vector<std::pair<VertexId, Value> >, CompareDist> frontier;

   //initialize distances
   //  use -1 to represent infinity for source_path
   //                      undefined for predecessor
   for (VertexId i = 0; i < m; ++i) {
       source_path[i] = -1;
       //Edge = std::make_pair(i, h_csrValA[i]);
       if( i!=src )
           frontier.push(std::pair<VertexId, Value>(i, h_csrValA[i]));
       if (MARK_PREDECESSORS)
           predecessor[i] = -1;
   }
   source_path[src] = 0;
   frontier.push(std::pair<VertexId, Value>(src, 0));
   VertexId search_depth = 0;

   //print_queue(frontier, 10);

   //
   //Perform SSSP
   //

   CpuTimer cpu_timer;
   cpu_timer.Start();
   while (!frontier.empty()) {
       
       // Dequeue node from frontier
       Edge dequeued_node = frontier.top();
       frontier.pop();

       // Set v as vertex index, d as distance
       VertexId v = dequeued_node.first;
       Value d = dequeued_node.second;
       //printf("Popped node: %d %f\n", v, d);

       // Locate adjacency list
       int edges_begin = h_rowPtrA[v];
       int edges_end = h_rowPtrA[v+1];

       // Checks that we only iterate through once
       //   -necessary because we will be having redundant vertices in
       //   queue so we will only do work when we have the best one
       //   -source_path[v] == -1 means we haven't explored it before
       if( source_path[v] != -1 || d <= source_path[v] ) {
           for( int edge = edges_begin; edge < edges_end; ++edge ) {
               //Lookup neighbor and enqueue if undiscovered
               VertexId neighbor = h_colIndA[edge];
               Value alt_dist = source_path[v] + h_csrValA[edge];
               //printf("source: %d, target: %d, old_d: %f, new_d: %f\n", v, neighbor, source_path[neighbor], alt_dist);
               //printf("edge: %d, weight: %f, path_weight: %f\n", edge, h_csrValA[edge], source_path[v]);
               if( source_path[neighbor] == -1 || alt_dist < source_path[neighbor] ) {
                   source_path[neighbor] = alt_dist;
                   frontier.push(std::pair<VertexId,Value>(neighbor,alt_dist));
                   if(MARK_PREDECESSORS) 
                       predecessor[neighbor] = dequeued_node.first;
               }
           }
       }
   }
   
   if (MARK_PREDECESSORS)
       predecessor[src] = -1;

   cpu_timer.Stop();
   float elapsed = cpu_timer.ElapsedMillis();
   search_depth++;

   printf("CPU SSSP finished in %lf msec. Search depth is: %d\n", elapsed, search_depth);

   return search_depth;
}

int ssspCPU( const int src, const int m, const int *h_rowPtrA, const int *h_colIndA, const float* h_csrValA, float *h_ssspResultCPU, const int stop ) {

   typedef int VertexId; // Use as the node identifier type
   typedef float Value;

   VertexId *reference_check_preds = NULL;

   int depth = SimpleReferenceSssp<VertexId, Value>(
       m, h_rowPtrA, h_colIndA, h_csrValA,
       h_ssspResultCPU,
       reference_check_preds,
       src,
       stop);

   print_array(h_ssspResultCPU, m);
   return depth;
}

// A simple CPU-based reference SSSP ranking implementation
template<typename VertexId, typename Value>
void BoostReferenceSssp(
   const int m, const int edge, const VertexId *h_rowPtrA, const VertexId *h_colIndA, const Value *h_csrValA,
   Value                                   *source_path,
   VertexId                                *predecessor,
   VertexId                                src,
   VertexId                                stop)
{
   // Prepare Boost Datatype and Data structure
   typedef adjacency_list<vecS, vecS, directedS, no_property,
                          property <edge_weight_t, unsigned int> > Graph;

   typedef graph_traits<Graph>::vertex_descriptor vertex_descriptor;
   typedef graph_traits<Graph>::edge_descriptor edge_descriptor;

   typedef std::pair<VertexId, VertexId> Edge;

   Edge   *edges = ( Edge*)malloc(sizeof( Edge)*edge);
   Value *weight = (Value*)malloc(sizeof(Value)*edge);

   for (int i = 0; i < m; ++i)
   {
       for (int j = h_rowPtrA[i]; j < h_rowPtrA[i+1]; ++j)
       {
           edges[j] = Edge(i, h_colIndA[j]);
           weight[j] = h_csrValA[j];
       }
   }

   Graph g(edges, edges + edge, weight, m);

   std::vector<Value> d(m);
   std::vector<vertex_descriptor> p(m);
   vertex_descriptor s = vertex(src, g);

   property_map<Graph, vertex_index_t>::type indexmap = get(vertex_index, g);

   //
   // Perform SSSP
   //

   CpuTimer cpu_timer;
   cpu_timer.Start();

   if (MARK_PREDECESSORS) {
       dijkstra_shortest_paths(g, s,
           predecessor_map(make_iterator_property_map(
                   p.begin(), get(boost::vertex_index, g))).distance_map(
                       boost::make_iterator_property_map(
                           d.begin(), get(boost::vertex_index, g))));
   } else {
       dijkstra_shortest_paths(g, s,
           distance_map(boost::make_iterator_property_map(
                   d.begin(), get(boost::vertex_index, g))));
   }
   cpu_timer.Stop();
   float elapsed = cpu_timer.ElapsedMillis();

   printf("Boost SSSP finished in %lf msec.\n", elapsed);

   Coo<Value, Value>* sort_dist = NULL;
   Coo<VertexId, VertexId>* sort_pred = NULL;
   sort_dist = (Coo<Value, Value>*)malloc(
       sizeof(Coo<Value, Value>) * m);
   if (MARK_PREDECESSORS) {
       sort_pred = (Coo<VertexId, VertexId>*)malloc(
           sizeof(Coo<VertexId, VertexId>) * m);
   }
   graph_traits < Graph >::vertex_iterator vi, vend;
   for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
   {
       sort_dist[(*vi)].row = (*vi);
       sort_dist[(*vi)].col = d[(*vi)];
   }
   std::stable_sort(
       sort_dist, sort_dist + m,
       RowFirstTupleCompare<Coo<Value, Value> >);

   if (MARK_PREDECESSORS)
   {
       for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
       {
           sort_pred[(*vi)].row = (*vi);
           sort_pred[(*vi)].col = p[(*vi)];
       }
       std::stable_sort(
           sort_pred, sort_pred + m,
           RowFirstTupleCompare< Coo<VertexId, VertexId> >);
   }

   for (int i = 0; i < m; ++i)
   {
       source_path[i] = sort_dist[i].col;
   }
   if (MARK_PREDECESSORS) {
       for (int i = 0; i < m; ++i)
       {
           predecessor[i] = sort_pred[i].col;
       }
   }
   if (sort_dist) free(sort_dist);
   if (sort_pred) free(sort_pred);

}

void ssspBoost( const int src, const int m, const int edge, const int *h_rowPtrA, const int *h_colIndA, const float* h_csrValA, float *h_ssspResultCPU, const int stop ) {

   typedef int VertexId; // Use as the node identifier type
   typedef float value;

   VertexId *reference_check_preds = NULL;

   BoostReferenceSssp<VertexId, value>(
       m, edge, h_rowPtrA, h_colIndA, h_csrValA,
       h_ssspResultCPU,
       reference_check_preds,
       src,
       stop);

   print_array(h_ssspResultCPU, m);
}