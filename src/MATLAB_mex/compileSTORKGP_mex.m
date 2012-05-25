%Please change the last include path to suit your system (where eigen is)

mex createSTORKGP.cpp ../../build/libOTL.a -I../libOTL/ -I/usr/local/include/
mex destroySTORKGP.cpp ../../build/libOTL.a -I../libOTL/ -I/usr/local/include/
mex updateSTORKGP.cpp ../../build/libOTL.a -I../libOTL/ -I/usr/local/include/
mex trainSTORKGP.cpp ../../build/libOTL.a -I../libOTL/ -I/usr/local/include/
mex predictSTORKGP.cpp ../../build/libOTL.a -I../libOTL/ -I/usr/local/include/
mex resetSTORKGP.cpp ../../build/libOTL.a -I../libOTL/ -I/usr/local/include/
mex saveSTORKGP.cpp ../../build/libOTL.a -I../libOTL/ -I/usr/local/include/
mex loadSTORKGP.cpp ../../build/libOTL.a -I../libOTL/ -I/usr/local/include/