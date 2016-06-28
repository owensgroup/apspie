cd $MEMBERWORK/csc103
date

ARCH="GEN_SM15"

if [ "$ARCH" = "GEN_SM10" ] ; then
    aprun -n4 -N1 ./test dataset/small/test_ac.mtx -multi 4
    aprun -n4 -N1 ./test dataset/small/test_dc.mtx -multi 4
fi

if [ "$ARCH" = "GEN_SM15" ] ; then
    #aprun -n4 -N1 ./test dataset/large/belgium_osm/belgium_osm.mtx -multi 4
    #aprun -n4 -N1 ./test dataset/large/delaunay_n13/delaunay_n13.mtx -multi 4
    #aprun -n4 -N1 ./test dataset/large/coAuthorsDBLP/coAuthorsDBLP.mtx -multi 4
	#aprun -n4 -N1 ./test dataset/large/ak2010/ak2010.mtx -multi 4
	#aprun -n4 -N1 ./test dataset/large/delaunay_n21/delaunay_n21.mtx -multi 4
	#aprun -n4 -N1 ./test dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx -multi 4
	aprun -n4 -N1 ./test dataset/large/kron_g500-logn21/kron_g500-logn21.mtx -multi 4
fi

if [ "$ARCH" = "GEN_SM45" ] ; then
	aprun -b -n4 -N1 nvprof -o output.%h.%p --profile-from-start off ./test dataset/large/kron_g500-logn21/kron_g500-logn21.mtx -multi 4
fi

if [ "$ARCH" = "GEN_SM50" ] ; then
    aprun -n4 -N1 ./test+pat dataset/large/kron_g500-logn21/kron_g500-logn21.mtx -multi 4
	#pat_report -O samp_profile+src test+pat*.xf
    #pat_report -T -o patrpt.txt test+pat*.xf
fi

if [ "$ARCH" = "GEN_SM20" ] ; then
    aprun -n4 -N1 ./test dataset/small/test_cc.mtx -multi 4
    aprun -n4 -N1 ./test dataset/small/test_bc.mtx -multi 4
    aprun -n4 -N1 ./test dataset/small/test_pr.mtx -multi 4
    aprun -n4 -N1 ./test dataset/small/chesapeake.mtx -multi 4
fi

for i in 1 2 4 6 8 16 32 64
do
	if [ "$ARCH" = "GEN_SM5" ] ; then
		#aprun -n$i -N1 ./test dataset/large/ak2010/ak2010.mtx -multi $i
		aprun -n$i -N1 ./test dataset/large/ak2010/ak2010.mtx -multi $i
	fi
done

for i in kron_g500-logn16 kron_g500-logn17 kron_g500-logn18 kron_g500-logn19 kron_g500-logn20 kron_g500-logn21
do
        if [ "$ARCH" = "GEN_SM25" ] ; then
        aprun -n 1 ./test dataset/large/$i/$i.mtx -undirected
    fi
done

for i in 579593 897318 666033 194754 796384 924094 932129 912391 344516
do
    if [ "$ARCH" = "GEN_SM30" ] ; then
            aprun -n 1 ./test dataset/large/kron_g500-logn20/kron_g500-logn20.mtx -source $i -undirected
        fi
done

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
    if [ "$ARCH" = "GEN_SM35" ] ; then
        aprun -n4 -N1 ./test dataset/large/$i/$i.mtx -multi 4
    else
        if [ "$ARCH" = "GEN_SM40" ] ; then
            aprun -n4 -N1 ./test dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx -multi 4
            aprun -n4 -N1 ./test dataset/large/kron_g500-logn21/kron_g500-logn21.mtx -multi 4
            break
        fi
    fi
done
