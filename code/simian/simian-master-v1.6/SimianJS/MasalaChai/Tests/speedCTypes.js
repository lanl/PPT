#!/usr/bin/env js --enable-avx --baseline-eager --baseline-warmup-threshold=5 --unboxed-arrays --ion-regalloc=testbed --code-coverage --ion-shared-stubs=on

const INT_MAX = Math.pow(2, 31)-1;
const COUNT = 10000000;

// Import native functions
// open a library
//var libc = ctypes.open("libc.so.6");
var libc = ctypes.open("libSystem.dylib");
var sin = libc.declare("sin", // function name
                        ctypes.default_abi, // call ABI
                        ctypes.double, // return type
                        ctypes.double // argument type
);
var rand = libc.declare("rand", // function name
                        ctypes.default_abi, // call ABI
                        ctypes.int // return type
);

function valFun() {
    var val = 0.0;
    for (var i = 0; i < COUNT; i++) {
        val += sin(i*rand()/INT_MAX);
    }
    return val;
}

var val = valFun(); //JIT Warmup

var start = new Date();
val = valFun();
var end = new Date() - start;
console.log(val);
console.log("Took:", end, "ms")
