#!/bin/bash

gnuplot main.gp
cd png
for i in {0..600}
do
   infile="img"
   outfile="img"
   if [ $i -lt 100 ]
     then
       infile=$infile"0"
       outfile=$outfile"0"
   fi	
   if [ $i -lt 10 ] 
     then
       infile=$infile"0"
       outfile=$outfile"0"
   fi
   infile=$infile$i".fig"
   outfile=$outfile$i".png"
   
   convert $infile -flatten -rotate 90 $outfile
   #convert $outfile -flatten $outfile
done

convert -delay 10 -loop 0 img*.png animation.gif

animate animation.gif
