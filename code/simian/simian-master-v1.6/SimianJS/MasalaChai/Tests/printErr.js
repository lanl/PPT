masala.io.print("Hello 0");
var fp = masala.io.fopen("Test", "w");
masala.io.print("Hello 1");
masala.io.fwrite(fp, "Hello")
masala.io.print("Hello 2");
masala.io.fclose();
masala.io.print("Hello 3");
masala.io.fclose(fp);
masala.io.print("Hello 4");
