

namespace graphblas{
	namespace io{

		template< typename IndexT, 
			      typename T >
        void readMTXfile( FILE*   input_file,
				          
						  IndexT* I,
						  IndexT* J,
						  T*	  K ) {
			int nrow, ncol, nedge;

            readEdge( nrow, ncol, nedge, input_file );


		}

	}
}
