

ARCH="GEN_SM35"
DELTA="-delta 0.01"

if [ "$ARCH" = "GEN_SM15" ] ; then
    ./test ../../dataset/small/test_mis.mtx
    ./test ../../dataset/small/test_mesh.mtx
fi

if [ "$ARCH" = "GEN_SM20" ] ; then
    ./test ../../dataset/small/test_cc.mtx
    ./test ../../dataset/small/test_bc.mtx
    ./test ../../dataset/small/test_pr.mtx
    ./test ../../dataset/small/chesapeake.mtx
fi

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 kron_g500-logn18 kron_g500-logn19 kron_g500-logn21
do
    if [ "$ARCH" = "GEN_SM35" ] ; then
        ./test /data/gunrock_dataset/large/$i/$i.mtx $DELTA
    else
        if [ "$ARCH" = "GEN_SM40" ] ; then
            ./test /data/gunrock_dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx $DELTA
            ./test /data/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx $DELTA
            break
        fi
    fi
done

for i in 2-bitcoin 6-roadnet 4-pld 
do
    if [ "$ARCH" = "GEN_SM45" ] ; then
        ./test /data/PPOPP15/$i.mtx
    fi
done
