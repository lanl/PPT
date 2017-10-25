//#!/usr/bin/env js --enable-avx
const load = masala.io.load, print = masala.io.print, now = masala.time.now;

load("./hash.js");

const COUNT = 10000000;

var start = now();
var h = 0;
for (var i=0; i<COUNT; i++) h = hash(i.toString());
var finish = now() - start;
print(h);
print("Took: " + finish + " ms");
