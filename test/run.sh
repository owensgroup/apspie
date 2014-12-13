./test ../dataset/small/test_cc.mtx
./test ../dataset/small/test_bc.mtx
./test ../dataset/small/test_pr.mtx
./test ../dataset/small/chesapeake.mtx

ARCH="GEN_SM20"

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M #soc-LiveJournal1 kron_g500-logn21
do
    if [ "$ARCH" = "GEN_SM20" ] ; then
        ./test ../dataset/large/$i/$i.mtx
        #./test01 ../dataset/large/$i/$i.mtx
    else
        ./test /data/gunrock_dataset/large/$i/$i.mtx
    fi
done
