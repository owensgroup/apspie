

ARCH="GEN_SM20"

if [ "$ARCH" = "GEN_SM20" ] ; then
    ./test01 ../dataset/small/test_cc.mtx
    ./test01 ../dataset/small/test_bc.mtx
    ./test01 ../dataset/small/test_pr.mtx
    ./test01 ../dataset/small/chesapeake.mtx
else
    ./test01 /data/gunrock_dataset/small/test_cc.mtx
    ./test01 /data/gunrock_dataset/small/test_bc.mtx
    ./test01 /data/gunrock_dataset/small/test_pr.mtx
    ./test01 /data/gunrock_dataset/small/chesapeake.mtx
fi

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
    if [ "$ARCH" = "GEN_SM20" ] ; then
        ./test01 ../dataset/large/$i/$i.mtx
    else
        ./test01 /data/gunrock_dataset/large/$i/$i.mtx
    fi
done
