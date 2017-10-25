const print = masala.io.print, fopen = masala.io.fopen, fclose = masala.io.fclose, fwrite = masala.io.fwrite;

var x;

var fp = fopen("o1.txt", "w");
var state = fwrite(fp, "Hello world\n"+ "1", 2, x+"\n");
print(state);
state = fclose(fp);
print(state);

fp = fopen("o1.txt", "a");
state = fwrite(fp, "Hello world 2\n");
print(state);
state = fclose(fp);
print(state);
