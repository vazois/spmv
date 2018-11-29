make

echo "1. benchmarking as-skitter.txt ..."
./gmain -f=../data/as-skitter.txt > as-skitter.time

echo "2. benchmarking bn-human-Jung2015_M87113878.edges ..."
./gmain -f=../data/bn-human-Jung2015_M87113878.edges > bn-human-Jung2015_M87113878.time

echo "3. benchmarking cit-Patents.txt ..."
./gmain -f=../data/cit-Patents.txt > cit-Patents.time

echo "4. com-orkut.ungraph.txt ..."
./gmain -f=../data/com-orkut.ungraph.txt > com-orkut.ungraph.time

echo "5. roadNet-CA.txt ..."
./gmain -f=../data/roadNet-CA.txt > roadNet-CA.time
 
echo "6. twitter_rv.next ..."
./gmain -f=../data/twitter_rv.net > twitter_rv.time

echo "7. wiki-Talk.txt ..."
./gmain -f=../data/wiki-Talk.txt > wiki-Talk.time

echo "8. sd-arc ..."
./gmain -f=../data/sd-arc > sd-arc.time

echo "9. soc-friendster.mtx ..."
./gmain -f=../data/soc-friendster.mtx > soc-friendster.time

