all: gccnosan	clangnosan gccsan clangsan

buildgcc/nosan/:
	CXXFLAGS="" CXX=g++ cmake -S test -B buildgcc/nosan/ -DCMAKE_BUILD_TYPE=Debug -DUSEMIMALLOC=ON

buildgcc/test/:
	CXXFLAGS="" CXX=g++ cmake -S test -B buildgcc/test/ -DCMAKE_BUILD_TYPE=Debug -DUSE_SANITIZER='Address;Undefined' -DUSEMIMALLOC=ON

buildclang/nosan/:
	CXXFLAGS="" CXX=/usr/bin/clang++ cmake -S test -B buildclang/nosan/ -DCMAKE_BUILD_TYPE=Debug -DUSEMIMALLOC=ON

buildclang/test/:
	CXXFLAGS="" CXX=/usr/bin/clang++ cmake -S test -B buildclang/test/ -DCMAKE_BUILD_TYPE=Debug -DUSE_SANITIZER='Address;Undefined' -DUSEMIMALLOC=ON

gccnosan: buildgcc/nosan/
	cmake --build buildgcc/nosan/
	cmake --build buildgcc/nosan/ --target test

gccsan: buildgcc/test/
	cmake --build buildgcc/test/ 
	cmake --build buildgcc/test/ --target test

clangnosan: buildclang/nosan/
	cmake --build buildclang/nosan/
	cmake --build buildclang/nosan/ --target test

clangsan: buildclang/test/
	cmake --build buildclang/test/ 
	cmake --build buildclang/test/ --target test

clean:
	rm -r buildgcc buildclang
