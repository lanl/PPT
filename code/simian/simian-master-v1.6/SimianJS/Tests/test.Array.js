"use strict";
"use wasm";

var print = print;
if (!print) print = masala.io.print;

const count = 10000000, active = count/2, clique = 20, random = Math.random;

var start = new Date();
var A = new Array(count);
for (var i = 0; i < active; i++) A[i] = random();

var value = 0.0;
var opCount = 0;
for (var i = 0; i < active; i++)
    for (var j = 0; j < clique; j++) {
        value += A[i]*random();
        opCount++;
    }

var finish = new Date() - start;

print(value, opCount);
print("Took: " + (finish/1000) + "(s)");
