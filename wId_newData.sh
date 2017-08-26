angle=8;
for view in {X,Y,Z};do \
if [ $view == "X" ];then param2xfm -rotation $angle 0 0 rot${view}pos${angle}.xfm;param2xfm -rotation -$angle 0 0 rot${view}neg${angle}.xfm;
elif [ $view == "Y" ];then param2xfm -rotation 0 $angle 0 rot${view}pos${angle}.xfm;param2xfm -rotation 0 -$angle 0 rot${view}neg${angle}.xfm;
else param2xfm -rotation 0 0 $angle rot${view}pos${angle}.xfm;param2xfm -rotation 0 0 -$angle rot${view}neg${angle}.xfm;
fi;done
 


for ii in data/sub-*/*_acq-*;do \
id=`dirname $ii|rev|cut -d/ -f1|rev|cut -c5-`;
echo $id;
mincaverage $ii `echo $ii|cut -d. -f1`_sum.mnc -avgdim time -width_weighted;
for vv in rot*;do \
mincresample `echo $ii|cut -d. -f1`_sum.mnc data_new/sub_${id}_task-`echo ${vv}|cut -d. -f1`_`basename $ii|cut -d_ -f2`.mnc -transformation $vv -like `echo $ii|cut -d. -f1`_sum.mnc -nearest_neighbour -clobber;
done;
done;


for ii in data/sub-*/*labels.mnc;do \
id=`dirname $ii|rev|cut -d/ -f1|rev|cut -c5-`;
echo $id;
for vv in rot*;do \
mincresample $ii data_new/sub_${id}_labels-`echo ${vv}|cut -d. -f1`.mnc -transformation $vv -like $ii -nearest_neighbour -clobber;
done;
done;


for ii in data/sub-*/*labels_brainmask.mnc;do \
id=`dirname $ii|rev|cut -d/ -f1|rev|cut -c5-`;
echo $id;
for vv in rot*;do \
mincresample $ii data_new/sub_${id}_labels-`echo ${vv}|cut -d. -f1`_`basename $ii|cut -d_ -f3` -transformation $vv -like $ii -nearest_neighbour -clobber;
done;
done;


for ii in data/sub-*/*sum.mnc;do id=`dirname $ii|cut -d/ -f2|cut -c5-`;cp $ii data_new/sub_${id}_task-rot0_`basename $ii|cut -d_ -f2`.mnc;done
for ii in data/sub-*/*labels.mnc;do id=`dirname $ii|cut -d/ -f2|cut -c5-`;cp $ii data_new/sub_${id}_labels-rot0.mnc;done
for ii in data/sub-*/*labels_brainmask.mnc;do id=`dirname $ii|cut -d/ -f2|cut -c5-`;cp $ii data_new/sub_${id}_labels-rot0_`basename $ii|cut -d_ -f3`;done
