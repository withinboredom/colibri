#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

#ifndef _WIN32
static void wr32(unsigned char *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (i * 8));
}
static void wr64(unsigned char *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (unsigned char)(v >> (i * 8));
}
static int test_native_index(void) {
    char dir[] = "/tmp/coli-native-XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    char path[512]; snprintf(path, sizeof(path), "%s/model.coli", dir);
    unsigned char file[184]; memset(file, 0, sizeof(file));
    memcpy(file, "COLINAT1", 8); wr32(file + 8, 1); wr64(file + 16, sizeof(file));
    wr64(file + 24, 1); wr64(file + 32, 0); wr64(file + 40, 128);
    wr64(file + 48, 184); wr64(file + 56, 64);
    file[64]=1; file[65]=2; file[66]=3; file[67]=4;
    wr32(file + 128, 4); wr32(file + 132, 3); wr64(file + 136, 64);
    wr64(file + 144, 4); wr64(file + 152, 4); memcpy(file + 176, "tiny", 4);
    int fd=open(path,O_CREAT|O_TRUNC|O_WRONLY,0600); CHECK(fd>=0);
    CHECK(write(fd,file,sizeof(file))==(ssize_t)sizeof(file)); close(fd);
    shards S; st_init(&S,dir);
    st_tensor *tensor=st_find(&S,"tiny"); CHECK(tensor && tensor->fd==S.native_fd);
    unsigned char got[4]={0}; st_read_raw(&S,"tiny",got,0);
    CHECK(!memcmp(got,"\1\2\3\4",4));
    unlink(path); rmdir(dir); return 0;
}
#endif

int main(void) {
    CHECK(bf16_to_f32(0x3f80) == 1.0f);
    CHECK(bf16_to_f32(0xc020) == -2.5f);
    CHECK(f16_to_f32(0x3c00) == 1.0f);
    CHECK(f16_to_f32(0xc100) == -2.5f);
    CHECK(f16_to_f32(0x0001) > 0.0f);
    CHECK(isinf(f16_to_f32(0x7c00)));
    CHECK(st_hash("tensor.weight") == st_hash("tensor.weight"));
    CHECK(st_hash("tensor.weight") != st_hash("tensor.bias"));
#ifndef _WIN32
    CHECK(test_native_index() == 0);
#endif

    puts("model container primitive tests: ok");
    return 0;
}
