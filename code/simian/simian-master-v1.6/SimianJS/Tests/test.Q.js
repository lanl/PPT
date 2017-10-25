//#!/usr/bin/env js --enable-avx
const load = masala.io.load, print = masala.io.print, random = Math.random, now = masala.time.now;
//const load = masala.io.load, print = masala.io.print, random = masala.random.uniform, now = masala.time.now;

load("./eventQ.js");
const COUNT = 10000000;

var myList = [];
var Q = new eventQ(myList);

print("Pushing:");
for (var i=0; i<COUNT; i++) Q.push({"time" : random()});

print("Checking Queue Pop Order:");
var prvMin = -1;
var topItem;
while (myList.length > 0) {
    topItem = Q.pop();
    if (prvMin > topItem.time) print("Out of order");
    prvMin = topItem.time;
}

print("Pushing Time Check:");
var start = now();
for (var i=0; i<COUNT; i++) Q.push({"time" : random()});
var finish = now() - start;
print("One Push took: " + finish/COUNT + " ms")

print("Poping Time Check:");
var topItem;
var start = now();
while (myList.length > 0) topItem = Q.pop();
var finish = now() - start;
print("One Pop took: " + finish/COUNT + " ms")
