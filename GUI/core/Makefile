CC := clang
SRC := inference.c
OUT_DYLIB := libhybrid_engine.dylib
OUT_SO := libhybrid_engine.so
ORT_ROOT := ../assets/onnxruntime-osx-arm64-1.24.4
ORT_INC := $(ORT_ROOT)/include
ORT_LIB := $(ORT_ROOT)/lib

CFLAGS := -fPIC -I$(ORT_INC)
LDFLAGS := -dynamiclib -L$(ORT_LIB) -lonnxruntime -Wl,-rpath,@loader_path/../assets/onnxruntime-osx-arm64-1.24.4/lib

.PHONY: all build so clean

all: build

build: $(OUT_DYLIB)

$(OUT_DYLIB): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

so: $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(OUT_SO) $<

clean:
	rm -f $(OUT_DYLIB) $(OUT_SO)
