/*
Copyright (c) 2015, Los Alamos National Security, LLC
All rights reserved.

Copyright 2015. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
	Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/*
Author: Nandakishore Santhi
Date: 23 May, 2016
Copyright: Open source, must acknowledge original author
Purpose: JITed PDES Engine in MasalaChai JavaScript, a custom dialect of Mozilla Spidermonkey
   Priority Queue or heap, with head being least time
*/
var eventQ = class {
    constructor(list) {
        this.list = list;
    }

    toString() {
        return "eventQ Time Priority Queue";
    }

    push(item) {
        var temp, list=this.list, floor=Math.floor;
        var curPos, parentPos = list.length;
        list.push(item);
        while (parentPos > 0) {
            curPos = parentPos;
            parentPos = floor(parentPos/2);
            if (list[curPos].time < list[parentPos].time) {
                temp = list[curPos];
                list[curPos] = list[parentPos];
                list[parentPos] = temp;
            }
            else break;
        }
    }

    peek() {
        return this.list[0].time;
    }

    length() {
        return this.list.length;
    }

    pop() {
        var temp, list=this.list;
        var length=list.length-1;
        temp = list[0];
        list[0] = list[length];
        list[length] = temp;

        var topItem=list.pop(), curPos=0, left=1, right=2, state=0, min=0;
        while (true) {
            if ((left < length) && (list[min].time > list[left].time)) {
                state = 1;
                min = left;
            }
            if ((right < length) && (list[min].time > list[right].time)) {
                state = 2;
                min = right;
            }
            if (state == 0) break;
            else if (state == 1) {
                temp = list[curPos];
                list[curPos] = list[left];
                list[left] = temp;

                curPos = left;
                left *= 2;
                right = left+1;
                state = 0;
            } else { //if (state == 2)
                temp = list[curPos];
                list[curPos] = list[right];
                list[right] = temp;

                curPos = right;
                left = right*2;
                right = left+1;
                state = 0;
            }
        }
        return topItem;
    }
}
