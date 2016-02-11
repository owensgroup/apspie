apspie
======

1. git clone
2. in base folder:
     mkdir ext
     cd ext
     git submodule add https://github.com/NVlabs/cub.git
     git submodule add https://github.com/NVlabs/moderngpu.git
3. cd tests/bfs
   cmake .
   make
   sh run.sh
