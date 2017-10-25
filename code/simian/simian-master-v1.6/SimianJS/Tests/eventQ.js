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

        var topItem=list.pop(), curPos=0, left=1, right=2, min=0, state=0;
        while (true) {
            if (left < length) {
                if (list[min].time > list[left].time) {
                    state = 1;
                    min = left;
                }
            }
            if (right < length) {
                if (list[min].time > list[right].time) {
                    state = 2;
                    min = right;
                }
            }
            if (state == 0) break;

            temp = list[curPos];
            list[curPos] = list[min];
            list[min] = temp;

            curPos = min;
            left = min*2;
            right = left+1;
            state = 0;
        }
        return topItem;
    }
}
