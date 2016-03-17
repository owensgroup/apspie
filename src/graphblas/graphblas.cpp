
namespace graphblas{
    
    // @brief  Builds matrix from three arrays that represent
    //         (coordinate1, coordinate2, val)
    //
    // @tparam[in] IndexT
    // @tparam[in] T
    //
    // @param[in] I  Row of entry
    // @param[in] J  Col of entry
    // @param[in] K  Val of entry
    // @param[in] m  Number of entries
    template< typename IndexT,
    	      typename T>
    void buildMatrix( IndexT*   I,
    		      IndexT*   J,
    		      T*        K,
    		      IndexT    m,
    		      Matrix<T> A );
    
}