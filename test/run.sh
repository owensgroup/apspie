

ARCH="GEN_SM35"

if [ "$ARCH" = "GEN_SM20" ] ; then
    ./test ../dataset/small/test_cc.mtx
    ./test ../dataset/small/test_bc.mtx
    ./test ../dataset/small/test_pr.mtx
    ./test ../dataset/small/chesapeake.mtx
fi

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
    if [ "$ARCH" = "GEN_SM35" ] ; then
        ./test /data/gunrock_dataset/large/$i/$i.mtx
    fi
done
