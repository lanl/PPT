/*
Author: Nandakishore Santhi
Date: 22 May, 2016
Copyright: Open source, must acknowledge original author
Purpose: JITed PDES Engine in MasalaChai JavaScript, a custom dialect of Mozilla Spidermonkey
  Simple string to integer hashing
*/
// djb2 @ http://www.cse.yorku.ca/~oz/hash.html
function hash(str) {
    var res = 5381;
    var i = str.length;
    while(i--) res = 33*res + str.charCodeAt(i); //hash(i) = 33*hash(i-1) + c
    return res;
}
