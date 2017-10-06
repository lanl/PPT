set size ratio 4
set terminal fig size 10,10
set xrange [-50:250]
set yrange [-50:250]
set zrange [-50:250]


do for [n=0:600] {
    set output sprintf('png/img%03.0f.fig',n)
    splot 'data.dat' u 2:3:4:5 every ::(n*10000)::(n+1)*10000 with points pt 7 ps variable
}
