//#!/usr/bin/env js --enable-avx --baseline-eager --baseline-warmup-threshold=5 --unboxed-arrays --ion-regalloc=testbed --code-coverage --ion-shared-stubs=on
const print = masala.io.print, sin = Math.sin, random = Math.random;
const COUNT = 10000000;

function valFun() {
    var val = 0.0;
    for (var i = 0; i < COUNT; i++) {
        val += sin(i*random());
    }
    return val;
}

if (masala.jit.status("baseline") && masala.jit.status("ion")) print("JIT is enabled (baseline + ion)");

var val = valFun(); //JIT Warmup

var start = new Date();
val = valFun();
var end = new Date() - start;
print(val);
print("Took:", end, "ms")
