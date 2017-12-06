apspie
======

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
