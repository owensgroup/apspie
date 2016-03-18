

namespace graphblas{
	namespace io{

		// @brief 
		//
		// @tparam[in]
		// @tparam[out]
		//
		// @param[in]
		// @param[in]
		template< typename IndexT, 
			      typename T >
        void readMTXfile( FILE*   input_file,
				          int     argc,
						  char**  argv, 
						  IndexT* I,
						  IndexT* J,
						  T*	  K ) {

			int nrow, ncol, nedge;
            parseArgs;
            readEdge( nrow, ncol, nedge, input_file );


		}

	}
}
