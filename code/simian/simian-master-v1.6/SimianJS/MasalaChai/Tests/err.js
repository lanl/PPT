var j = 1;
function test(type) {
    let i = 0.1;
    i += 2;
    if (type == 0) {
        masala.dbg.bt();
        masala.io.print(masala.dbg.getbt({args: true, locals: true, thisprops: true}));
    }
    if (type == 1) throw(new Error("Test 1"));
    if (type == 2) throw("Error");
    if (type == 3) print("Hello");
};
test(0);
test(1);
//test(2);
//test(3);
