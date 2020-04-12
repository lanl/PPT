#!/bin/bash

    	for func in Ovhd  Add  Abs  Bfind  Clz  Cnot  Copysign  DFAdd  DFDiv  Div  DivU  Ex2  FastSqrt  FDiv  HFAdd  Lg2  MAdd_cc  MMad_cc  MSubc  Mul  Mul24  Mul64Hi  Popc  Rcp  Rem  RemU  Rsqrt  Sad  Sin  Sqrt
    	do 
        	echo "**************************************************************" >> output/$func
			echo "Function --> " >> output/$func
			echo $func >> output/$func
			echo -e "\n" >> output/$func
			./a.out $func >> output/$func
			echo $func done >> output/$func
			echo "**************************************************************" >> output/$func
			echo $func done
    	done
    echo All done
