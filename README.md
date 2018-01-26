apspie
======

<h3>Introduction</h3>

This is the source code of the following paper:

Yang, Carl, Yangzihao Wang, and John D. Owens. "Fast sparse matrix and sparse vector multiplication algorithm on the gpu." Parallel and Distributed Processing Symposium Workshop (IPDPSW), 2015 IEEE International. IEEE, 2015. [<a href="https://cloudfront.escholarship.org/dist/prd/content/qt1rq9t3j3/qt1rq9t3j3.pdf">pdf</a>][<a href="http://www.ece.ucdavis.edu/~ctcyang/pub/ipdpsw-slides2015.pdf">slides</a>]

Contact: <a href="http://www.ece.ucdavis.edu/~ctcyang/">Carl Yang</a>, Yangzihao Wang and John D. Owens.


<h3>Execution</h3>

1. `git clone --recursive https://github.com/owensgroup/apspie.git`
2. In base folder:  
   ```
   git checkout d247428
   git submodule update --remote --merge
   ```
3. Build individual algorithms:  
   bfs - Breadth First Search  
   mis - Maximal Independent Set  
   mm - Matrix Multiplication  
   sssp - Single Source Shortest Path  
   tc - Triangle Counting  

   From base folder:  
   ```
   cd test/[insert algorithm here]  
   vi CMakeLists.txt  
   cmake .  
   make -j16
   sh run.sh
   ```
