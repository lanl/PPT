const print = masala.io.print, sin = Math.sin, random = masala.random.uniform, now = masala.time.now;
const COUNT = 10000000;

function valFun() {
    var val = 0.0;
    for (var i = 0; i < COUNT; i++) val += sin(i*random());
    return val;
}

if (masala.jit.status("baseline") && masala.jit.status("ion")) print("JIT is enabled (baseline + ion)");

var val = valFun(); //JIT Warmup

var start = now();
var val = valFun();
var end = now() - start;
print(val);
print("Took:", end, "ms")
