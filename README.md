apspie
======

1. git clone https://github.com/owensgroup/apspie.git
2. In base folder:  
     mkdir ext  
     cd ext  
     git submodule add https://github.com/NVlabs/cub.git  
     git submodule add https://github.com/NVlabs/moderngpu.git
3. Build individual algorithms:  
   bfs - Breadth First Search  
   mis - Maximal Independent Set  
   mm - Matrix Multiplication  
   sssp - Single Source Shortest Path  
   tc - Triangle Counting  

   From base folder:
   cd tests/[insert algorithm here]  
   vi CMakeLists.txt  
   cmake .  
   make  
   sh run.sh  
