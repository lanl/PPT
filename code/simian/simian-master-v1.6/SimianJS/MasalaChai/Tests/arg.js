const print = masala.io.print;
print(masala.io.arg.length);
print(masala.io.arg);
for (let i=0; i<masala.io.arg.length; i++)
    print(masala.io.arg[i]);

const LEN = 100000;

masala.jit.off("baseline");
print(masala.jit.status("baseline"));
var t1 = masala.time.now();
var sum = 0;
for (i=0; i<LEN; i++) sum += i*i;
var t2 = masala.time.now();
print("Time taken: ", (t2-t1)/1000);

masala.jit.on("baseline");
print(masala.jit.status("baseline"));
var t1 = masala.time.now();
var sum = 0;
for (i=0; i<LEN; i++) {
    /*
    if (i==100) {
        masala.jit.off("baseline");
        print(masala.jit.status("baseline"));
    }
    */
    sum += i*i;
}
var t2 = masala.time.now();
print("Time taken: ", (t2-t1)/1000);


masala.jit.off("ion");
print(masala.jit.status("ion"));
var t1 = masala.time.now();
var sum = 0;
for (i=0; i<LEN; i++) sum += i*i;
var t2 = masala.time.now();
print("Time taken: ", (t2-t1)/1000);

masala.jit.on("ion");
print(masala.jit.status("ion"));
var t1 = masala.time.now();
var sum = 0;
for (i=0; i<LEN; i++) sum += i*i;
var t2 = masala.time.now();
print("Time taken: ", (t2-t1)/1000);


masala.io.exit(100);
print("Done!");
