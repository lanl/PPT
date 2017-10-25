const print = masala.io.print;
var x;

class _fp {
    constructor(filename, mode) {
        this.fp = masala.io.fopen(filename, mode);
    }

    close() {
        var state = masala.io.fclose(this.fp);
        return state;
    }

    write(...theArgs) {
        return masala.io.fwrite(this.fp, ...theArgs);
    }

    read(...theArgs) {
        return masala.io.fread(this.fp, ...theArgs);
    }
}

var file = {
    "open" : function(filename, mode) {
        return new _fp(filename, mode);
    }
}

var fp = file.open("o1.txt", "w");
var state = fp.write("Hello world\n" + "1 ", 2, " ", x+"\n");
print(state);
state = fp.close();
print(state);

fp = file.open("o1.txt", "a");
state = fp.write("Hello world 2\n");
print(state);
state = fp.close();
print(state);
