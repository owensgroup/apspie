

ARCH="GEN_SM0"

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

for i in uk-2005 webbase-2001
do
    if [ "$ARCH" = "GEN_SM20" ] ; then
        ./test ../dataset/large/$i/$i.mtx
    else
        ./test /data/gunrock_dataset/large/$i/$i.mtx
    fi
done
