call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cl.exe /EHsc /O2 /LD matmul_lib.cpp /link /OUT:directthon\directcompute.dll
