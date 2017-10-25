const print = masala.io.print;

//masala.io.load("msgpack.min.js");
masala.io.load("msgpack.js");

// encode from JS Object to MessagePack (Buffer)
var buffer = msgpack.encode({"foo": "bar", "undef" : undefined, "nu" : null, "ar" : [1, 2]});
print(buffer);

// decode from MessagePack (Buffer) to JS Object
var data = msgpack.decode(buffer); // => {"foo": "bar"}
print(data, data.foo, data.undef, data.nu, data.ar);
