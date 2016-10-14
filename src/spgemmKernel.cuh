#include "triple.hpp"

#define __GR_CUDA_ARCH__ 300

/**
 * Arch dispatch
 */

/**
 * Not valid for this arch (default)
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam VALID
 */
template<
    typename    KernelPolicy,
    typename    ProblemData,
    typename    Functor,
    bool        VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename KernelPolicy::Value    Value;
    typedef typename ProblemData::DataSlice DataSlice;

    // Get neighbor list sizes, scan to get both
    // fine_counts (for two per-thread methods) and
    // coarse_counts (for balanced-path per-block method)
    static __device__ void Inspect(
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            VertexId    *&d_edge_list,
                            SizeT       *&d_degrees,
                            SizeT       *&d_flags,
                            SizeT       &input_length,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
    }

    static __device__ SizeT IntersectTwoSmallNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            SizeT       *&d_degrees,
                            DataSlice   *&problem,
                            SizeT       *&d_output_counts,
                            SizeT       *&d_output_total,
                            SizeT       &input_length,
                            SizeT       &stride,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
    }

    static __device__ SizeT IntersectTwoLargeNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            VertexId    *&d_edge_list,
                            SizeT       *&d_degrees,
                            DataSlice   *&problem,
                            SizeT       &input_length,
                            SizeT       &nv_per_block,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
    }

};

/*
 * @brief Dispatch data structure.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
struct Dispatch<KernelPolicy, ProblemData, Functor, true>
{
    typedef typename KernelPolicy::VertexId         VertexId;
    typedef typename KernelPolicy::SizeT            SizeT;
    typedef typename KernelPolicy::Value            Value;
    typedef typename ProblemData::DataSlice         DataSlice;

    static __device__ void IntersectTwoSmallNL(
                            SizeT       *&d_row_offsets,
                            VertexId    *&d_column_indices,
                            VertexId    *&d_src_node_ids,
                            VertexId    *&d_dst_node_ids,
                            SizeT       *&d_degrees,
                            DataSlice   *&problem,
                            SizeT       *&d_output_counts,
                            SizeT       *&d_output_total,
                            SizeT       &input_length,
                            SizeT       &stride,
                            SizeT       &num_vertex,
                            SizeT       &num_edge)
    {
        // each thread process NV edge pairs
        // Each block get a block-wise intersect count
        VertexId start = threadIdx.x + blockIdx.x * blockDim.x;
        //VertexId end = (start + stride * KernelPolicy::THREADS > input_length)? input_length :
        //                (start + stride * KernelPolicy::THREADS);
        //typedef cub::BlockReduce<SizeT, KernelPolicy::THREADS> BlockReduceT;
        //__shared__ typename BlockReduceT::TempStorage temp_storage;

        for (VertexId idx = start; idx < input_length; idx += KernelPolicy::BLOCKS*KernelPolicy::THREADS) {
            SizeT count = 0;
            // get nls start and end index for two ids
            VertexId sid = __ldg(d_src_node_ids+idx);
            VertexId did = __ldg(d_dst_node_ids+idx);
            SizeT src_it = __ldg(d_row_offsets+sid);
            SizeT src_end = __ldg(d_row_offsets+sid+1);
            SizeT dst_it = __ldg(d_row_offsets+did);
            SizeT dst_end = __ldg(d_row_offsets+did+1);
            if (src_it == src_end || dst_it == dst_end) continue;
            SizeT src_nl_size = src_end - src_it;
            SizeT dst_nl_size = dst_end - dst_it;
            SizeT min_nl = (src_nl_size > dst_nl_size) ? dst_nl_size : src_nl_size;
            SizeT max_nl = (src_nl_size < dst_nl_size) ? dst_nl_size : src_nl_size;
            SizeT total = min_nl + max_nl;
            if ( min_nl * ilog2((unsigned int)(max_nl)) * 10 < min_nl + max_nl ) {
                // search
                SizeT min_it = (src_nl_size < dst_nl_size) ? src_it : dst_it;
                SizeT min_end = min_it + min_nl;
                SizeT max_it = (src_nl_size < dst_nl_size) ? dst_it : src_it;
                VertexId *keys = &d_column_indices[max_it];
                //printf("src:%d,dst:%d, src_it:%d, dst_it:%d, min_it:%d max_it:%d, min max nl size: %d, %d\n",sid, did, src_it, dst_it, min_it, max_it, min_nl, max_nl);
                while ( min_it < min_end) {
                    VertexId small_edge = d_column_indices[min_it++];
                    count += BinarySearch(keys, max_nl, small_edge);
                }
            } else {
                VertexId src_edge = __ldg(d_column_indices+src_it);
                VertexId dst_edge = __ldg(d_column_indices+dst_it);
                while (src_it < src_end && dst_it < dst_end) {
                    VertexId diff = src_edge - dst_edge;
                    src_edge = (diff <= 0) ? __ldg(d_column_indices+(++src_it)) : src_edge;
                    dst_edge = (diff >= 0) ? __ldg(d_column_indices+(++dst_it)) : dst_edge;
                    count += (diff == 0);
                }
            }
            d_output_total[idx] += total;
            d_output_counts[idx] += count;
        }
    }
};

/**
 * @brief Kernel entry for IntersectTwoSmallNL function
 *
 * @tparam KernelPolicy Kernel policy type for intersection.
 * @tparam ProblemData Problem data type for intersection.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_row_offset      Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices queue
 * @param[in] d_src_node_ids    Device pointer of VertexId to the incoming frontier queue (source node ids)
 * @param[in] d_dst_node_ids    Device pointer of VertexId to the incoming frontier queue (destination node ids)
 * @param[in] d_edge_list       Device pointer of VertexId to the edge list IDs
 * @param[in] d_degrees         Device pointer of SizeT to degree array
 * @param[in] problem           Device pointer to the problem object
 * @param[out] d_output_counts  Device pointer to the output counts array
 * @param[in] input_length      Length of the incoming frontier queues (d_src_node_ids and d_dst_node_ids should have the same length)
 * @param[in] num_vertex        Maximum number of elements we can place into the incoming frontier
 * @param[in] num_edge          Maximum number of elements we can place into the outgoing frontier
 *
 */

  template<typename KernelPolicy, typename ProblemData, typename Functor>
  __launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
  __global__
  void IntersectTwoSmallNL(
            typename KernelPolicy::SizeT        *d_row_offsets,
            typename KernelPolicy::VertexId     *d_column_indices,
            typename KernelPolicy::VertexId     *d_src_node_ids,
            typename KernelPolicy::VertexId     *d_dst_node_ids,
            typename KernelPolicy::SizeT        *d_degrees,
            typename ProblemData::DataSlice     *problem,
            typename KernelPolicy::SizeT        *d_output_counts,
            typename KernelPolicy::SizeT        *d_output_total,
            typename KernelPolicy::SizeT        input_length,
            typename KernelPolicy::SizeT        stride,
            typename KernelPolicy::SizeT        num_vertex,
            typename KernelPolicy::SizeT        num_edge)
{
    Dispatch<KernelPolicy, ProblemData, Functor>::IntersectTwoSmallNL(
            d_row_offsets,
            d_column_indices,
            d_src_node_ids,
            d_dst_node_ids,
            d_degrees,
            problem,
            d_output_counts,
            d_output_total,
            input_length,
            stride,
            num_vertex,
            num_edge);
}

// Kernel Entry point for performing batch intersection computation
template <typename typeVal>//, typename ProblemData, typename Functor>
    float LaunchKernel( d_matrix *C, d_matrix *A, d_matrix *B, 
        //gunrock::app::EnactorStats              &enactor_stats,
        //gunrock::app::FrontierAttribute<typename KernelPolicy::SizeT>
        //                                        &frontier_attribute,
        //typename ProblemData::DataSlice         *data_slice,
  // d_matrix A
        //typename KernelPolicy::SizeT            *d_row_offsets,
        //typename KernelPolicy::VertexId         *d_column_indices,
  // d_matrix B
        //typename KernelPolicy::VertexId         *d_src_node_ids,
        //typename KernelPolicy::VertexId         *d_dst_node_ids,
        //typename KernelPolicy::VertexId         *d_degrees,
        d_triple *d_output_counts,
        int *d_output_total,
  // d_matrix->nnz
        //typename KernelPolicy::SizeT            input_length,
        //typename KernelPolicy::SizeT            max_vertex,
        //typename KernelPolicy::SizeT            max_edge,
        //util::CtaWorkProgress                   work_progress,
        mgpu::CudaContext                             &context)
{
    int stride = (A->nnz + KernelPolicy::BLOCKS * KernelPolicy::THREADS - 1)
                        >> (KernelPolicy::LOG_THREADS + KernelPolicy::LOG_BLOCKS);
    
    IntersectTwoSmallNL<typeVal>
    <<<KernelPolicy::BLOCKS, KernelPolicy::THREADS>>>(
			C, A, B,
            d_output_counts,
            d_output_total,
            stride);

    long total = mgpu::Reduce(d_output_total, A->nnz, context);
    long tc_count = mgpu::Reduce(d_output_counts, C->nnz, context);
    printf("tc_total:%ld\n, tc_count:%ld\n", total, tc_count);
    return (float)tc_count / (float)total;
    //return total_counts[0];
}

