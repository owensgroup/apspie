

ARCH="GEN_SM35"

if [ "$ARCH" = "GEN_SM20" ] ; then
    ./test ../dataset/small/test_cc.mtx
    ./test ../dataset/small/test_bc.mtx
    ./test ../dataset/small/test_pr.mtx
    ./test ../dataset/small/chesapeake.mtx
fi

for i in kron_g500-logn16 kron_g500-logn17 kron_g500-logn18 kron_g500-logn19 kron_g500-logn20 kron_g500-logn21
do
    if [ "$ARCH" = "GEN_SM25" ] ; then
        ./test /data/gunrock_dataset/large/$i/$i.mtx
    fi
done

for i in rgg_n_2_15_s0 rgg_n_2_16_s0 rgg_n_2_17_s0 rgg_n_2_18_s0 rgg_n_2_19_s0 rgg_n_2_20_s0 rgg_n_2_21_s0 rgg_n_2_22_s0 rgg_n_2_23_s0 rgg_n_2_24_s0
do
    if [ "$ARCH" = "GEN_SM30" ] ; then
        ./test /data/vertexAPI2graphs/$i/$i.mtx
    fi
done 

for i in delaunay_n10 delaunay_n11 delaunay_n12 delaunay_n13 delaunay_n14 delaunay_n15 delaunay_n16 delaunay_n17 delaunay_n18 delaunay_n19 delaunay_n20 delaunay_n21 delaunay_n22 delaunay_n23 delaunay_n24
do
    if [ "$ARCH" = "GEN_SM31" ] ; then
        ./test /data/vertexAPI2graphs/$i/$i.mtx
    fi
done 

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
    if [ "$ARCH" = "GEN_SM35" ] ; then
        ./test /data/gunrock_dataset/large/$i/$i.mtx
    else
        if [ "$ARCH" = "GEN_SM40" ] ; then
            ./test /data/gunrock_dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx
            ./test /data/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx
        fi
    fi
done

for i in 2-bitcoin 6-roadnet 4-pld 
do
    if [ "$ARCH" = "GEN_SM45" ] ; then
        ./test /data/PPOPP15/$i.mtx
    fi
done
