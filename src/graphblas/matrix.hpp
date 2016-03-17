#ifndef APSPIE_SRC_GRAPHBLAS_MATRIX_HPP
#define APSPIE_SRC_GRAPHBLAS_MATRIX_HPP

namespace graphblas {

    // CSC format by default
    // Temporarily keeping member variables public to avoid having to use get
    template<typename T>
    class Matrix {
        public:
            IndexT* rowind;
            IndexT* colptr;
            T*         val;
            IndexT   nrows; // m in math spec   
  			IndexT   ncols; // n in math spec

			// @brief  Constructor
			//
			// @param[in] nrows  Number of rows.
			// @param[in] ncols  Number of columns.
            Matrix( IndexT m, 
					IndexT n ) 
				: nrows(m), ncols(n)
			{};

            // @brief  Copy constructor (disabled)
			//
			// @param[in] obj  Matrix we are trying to copy.
			Matrix( Matrix<T> const& obj ) = delete; 

			// @brief  Assignment operator (disabled)
			//
			// @param[in] obj  Matrix we are trying to copy.
			Matrix<T> operator=( Matrix<T> const& obj ) = delete;

			// @brief  Destructor
            ~Matrix();

	};

}

#endif  // APSPIE_SRC_GRAPHBLAS_MATRIX_HPP
