/* Indicizzazione e lettura on-demand di tensori da piu' file safetensors.
 * Equivale a Shards in engine.py, ma:
 *   - legge con pread (niente mmap) + posix_fadvise(DONTNEED) -> le pagine NON
 *     restano residenti nel processo. E' la correzione del bug di RSS: cosi' la
 *     RAM di picco resta densa+cache, non l'intero modello. (vedi memoria mmap-rss-bug)
 *   - converte sempre in float32 in uscita (BF16/F16/F32 supportati). */
#ifndef ST_H
#define ST_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include "json.h"
#include "compat.h"

/* tetto sulla dimensione dell'header safetensors: gli header reali sono piccoli
 * (KB..pochi MB). Un file crafted che dichiara un hlen enorme causerebbe una
 * malloc gigante prima ancora di leggere: lo respingiamo. */
#define ST_MAX_HEADER (512ll << 20)

typedef struct {
    char   *name;
    int     fd;
    int64_t off;       /* offset assoluto del dato dentro al file */
    int64_t nbytes;
    int     dtype;     /* 0=BF16 1=F16 2=F32 */
    int64_t numel;
} st_tensor;

/* Native model expert records. Each record is a pair of page-aligned regions:
 * [gate/up/down weights | padding] [gate/up/down scales | padding]. io_uring
 * can READV both regions into the final ESlot buffers with one SQE. */
typedef struct {
    int layer, n_experts, fmt, gs;
    int64_t base, stride;
    int64_t off[6], nbytes[6];
} st_expert_layout;

typedef struct {
    st_tensor *t;
    int        n, cap;
    int        fds[512];
    int        dfds[512];  /* gemelli O_DIRECT (aperti pigramente): -2 = non ancora provato */
    char      *paths[512];
    int        nfd;
    int       *hidx;      /* hash map nome->indice (open addressing): con ~120k tensori
                           * (GLM: 256 expert x 78 layer x 3 x 2) la scansione lineare
                           * costava decine di secondi/token (misurato sul primo run reale) */
    int        hcap;
    int        native_fd;
    st_expert_layout *native_experts;
    int        native_nexperts;
    st_expert_layout *native_layer[256];
} shards;
#define ST_MAX_SHARDS 512
#define ST_NATIVE_HEADER 64
#define ST_NATIVE_TENSOR 48
#define ST_NATIVE_EXPERT 128

static uint32_t st_rd32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t st_rd64(const unsigned char *p) {
    return (uint64_t)st_rd32(p) | (uint64_t)st_rd32(p + 4) << 32;
}

static uint64_t st_hash(const char *s){
    uint64_t h=1469598103934665603ULL;
    while(*s){ h^=(unsigned char)*s++; h*=1099511628211ULL; }
    return h;
}

static int st_dtype_code(const char *s) {
    if (!strcmp(s, "BF16")) return 0;
    if (!strcmp(s, "F16"))  return 1;
    if (!strcmp(s, "F32"))  return 2;
    if (!strcmp(s, "U8"))   return 3;   /* dati quantizzati (int4 packed / int8) */
    if (!strcmp(s, "I8"))   return 3;
    fprintf(stderr, "unsupported dtype: %s\n", s); exit(1);
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {            /* subnormale o zero */
        if (man == 0) u = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3FF; u = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) {  /* inf/nan */
        u = sign | 0x7F800000 | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

static int st_open_fd(shards *S, const char *path) {
    for (int i = 0; i < S->nfd; i++) if (!strcmp(S->paths[i], path)) return S->fds[i];
    int fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    S->paths[S->nfd] = strdup(path); S->fds[S->nfd] = fd;
#ifdef O_DIRECT
    S->dfds[S->nfd] = open(path, COMPAT_O_RDONLY | O_DIRECT);   /* eager: lookup poi thread-safe */
#elif defined(__APPLE__) || defined(_WIN32)
    S->dfds[S->nfd] = compat_open_direct(path);          /* macOS: F_NOCACHE; Windows: NO_BUFFERING */
#else
    S->dfds[S->nfd] = -1;                                /* niente equivalente: solo buffered */
#endif
    S->nfd++;
    return fd;
}

/* fd gemello O_DIRECT dello stesso file (bypassa la page cache: il buffered read su
 * ext4-in-VHDX si strozza a ~0.8 GB/s, O_DIRECT arriva a 2.3+; misurato). -1 se non disponibile. */
static int st_direct_fd(shards *S, int fd) {
    for (int i = 0; i < S->nfd; i++) if (S->fds[i] == fd) return S->dfds[i];
    return -1;
}

static void st_build_hash(shards *S) {
    S->hcap = 1; while (S->hcap < S->n * 2) S->hcap <<= 1;
    S->hidx = malloc((size_t)S->hcap * sizeof(int));
    if (!S->hidx) { perror("malloc tensor hash"); exit(1); }
    for (int i = 0; i < S->hcap; i++) S->hidx[i] = -1;
    for (int i = 0; i < S->n; i++) {
        uint64_t h = st_hash(S->t[i].name) & (uint64_t)(S->hcap - 1);
        while (S->hidx[h] >= 0) {
            if (!strcmp(S->t[S->hidx[h]].name, S->t[i].name)) {
                fprintf(stderr, "duplicate tensor name: %s\n", S->t[i].name); exit(1); }
            h = (h + 1) & (uint64_t)(S->hcap - 1);
        }
        S->hidx[h] = i;
    }
}

static void st_native_init(shards *S, const char *path) {
    int fd = st_open_fd(S, path); S->native_fd = fd;
    struct stat sb;
    if (fstat(fd, &sb)) { perror("fstat native model"); exit(1); }
    int64_t fsz = (int64_t)sb.st_size;
    unsigned char h[ST_NATIVE_HEADER];
    if (pread(fd, h, sizeof(h), 0) != (ssize_t)sizeof(h)) { perror("pread native header"); exit(1); }
    if (memcmp(h, "COLINAT1", 8) || st_rd32(h + 8) != 1) {
        fprintf(stderr, "%s: unsupported native model format\n", path); exit(1); }
    uint64_t declared = st_rd64(h + 16), nt = st_rd64(h + 24), ne = st_rd64(h + 32);
    uint64_t ti = st_rd64(h + 40), ei = st_rd64(h + 48), data_start = st_rd64(h + 56);
    if (declared != (uint64_t)fsz || nt > 10000000 || ne > 1024 ||
        data_start < ST_NATIVE_HEADER || ti < data_start || ei < ti || ei > declared ||
        ei - ti > (uint64_t)ST_MAX_HEADER) {
        fprintf(stderr, "%s: invalid native model header\n", path); exit(1); }
    size_t tibytes = (size_t)(ei - ti);
    unsigned char *idx = malloc(tibytes ? tibytes : 1);
    if (!idx) { perror("malloc native tensor index"); exit(1); }
    if (tibytes && pread(fd, idx, tibytes, (off_t)ti) != (ssize_t)tibytes) {
        perror("pread native tensor index"); exit(1); }
    S->cap = S->n = (int)nt; S->t = calloc(nt ? (size_t)nt : 1, sizeof(st_tensor));
    if (!S->t) { perror("calloc native tensors"); exit(1); }
    size_t p = 0;
    for (int i = 0; i < S->n; i++) {
        if (p > tibytes || tibytes - p < ST_NATIVE_TENSOR) {
            fprintf(stderr, "%s: truncated native tensor index\n", path); exit(1); }
        unsigned char *e = idx + p;
        uint32_t nl = st_rd32(e), dtype = st_rd32(e + 4);
        uint64_t off = st_rd64(e + 8), nb = st_rd64(e + 16), numel = st_rd64(e + 24);
        size_t padded = ((size_t)nl + 7) & ~(size_t)7;
        if (!nl || nl > (1u << 20) || dtype > 3 || padded < nl ||
            p + ST_NATIVE_TENSOR > tibytes || padded > tibytes - p - ST_NATIVE_TENSOR ||
            off < data_start || nb > declared || off > declared - nb || numel > INT64_MAX) {
            fprintf(stderr, "%s: invalid native tensor entry %d\n", path, i); exit(1); }
        st_tensor *t = &S->t[i];
        t->name = malloc((size_t)nl + 1);
        if (!t->name) { perror("malloc native tensor name"); exit(1); }
        memcpy(t->name, e + ST_NATIVE_TENSOR, nl); t->name[nl] = 0;
        if (memchr(t->name, 0, nl)) {
            fprintf(stderr, "%s: NUL in native tensor name\n", path); exit(1); }
        t->fd = fd; t->off = (int64_t)off; t->nbytes = (int64_t)nb;
        t->dtype = (int)dtype; t->numel = (int64_t)numel;
        p += ST_NATIVE_TENSOR + padded;
    }
    if (p != tibytes) { fprintf(stderr, "%s: native tensor index size mismatch\n", path); exit(1); }
    free(idx);

    if (ne > SIZE_MAX / sizeof(st_expert_layout) ||
        ne > ((uint64_t)fsz - ei) / ST_NATIVE_EXPERT) {
        fprintf(stderr, "%s: invalid native expert index\n", path); exit(1); }
    S->native_experts = calloc(ne ? (size_t)ne : 1, sizeof(st_expert_layout));
    if (!S->native_experts) { perror("calloc native expert layouts"); exit(1); }
    S->native_nexperts = (int)ne;
    for (int i = 0; i < S->native_nexperts; i++) {
        unsigned char e[ST_NATIVE_EXPERT];
        if (pread(fd, e, sizeof(e), (off_t)(ei + (uint64_t)i * sizeof(e))) != (ssize_t)sizeof(e)) {
            perror("pread native expert layout"); exit(1); }
        st_expert_layout *x = &S->native_experts[i];
        x->layer = (int)st_rd32(e); x->n_experts = (int)st_rd32(e + 4);
        x->base = (int64_t)st_rd64(e + 8); x->stride = (int64_t)st_rd64(e + 16);
        for (int k = 0; k < 6; k++) x->off[k] = (int64_t)st_rd64(e + 24 + k * 8);
        for (int k = 0; k < 6; k++) x->nbytes[k] = (int64_t)st_rd64(e + 72 + k * 8);
        x->fmt = (int)st_rd32(e + 120); x->gs = (int)st_rd32(e + 124);
        if (x->layer < 0 || x->layer >= 256 || S->native_layer[x->layer] ||
            x->n_experts < 1 || x->base < (int64_t)data_start ||
            x->stride < 4096 || (x->base & 4095) || (x->stride & 4095) ||
            x->base > fsz - x->stride || x->n_experts > (fsz - x->base) / x->stride ||
            x->fmt < 1 || x->fmt > 4) {
            fprintf(stderr, "%s: invalid native expert layout %d\n", path, i); exit(1); }
        for (int k = 0; k < 6; k++) if (x->off[k] < 0 || x->nbytes[k] < 0 ||
            x->off[k] > x->stride - x->nbytes[k]) {
            fprintf(stderr, "%s: native expert field outside record\n", path); exit(1); }
        if ((x->off[3] & 4095) || x->off[0] + x->nbytes[0] > x->off[3] ||
            x->off[1] + x->nbytes[1] > x->off[3] || x->off[2] + x->nbytes[2] > x->off[3] ||
            x->off[4] < x->off[3] || x->off[5] < x->off[3]) {
            fprintf(stderr, "%s: invalid native expert weight/scale regions\n", path); exit(1); }
        S->native_layer[x->layer] = x;
    }
    st_build_hash(S);
    fprintf(stderr, "[NATIVE] %d tensors, %d expert layouts from model.coli\n", S->n, S->native_nexperts);
}

static st_expert_layout *st_native_expert(shards *S, int layer) {
    return layer >= 0 && layer < 256 ? S->native_layer[layer] : NULL;
}

/* indicizza tutti i model-*.safetensors in snap_dir */
static void st_init(shards *S, const char *snap_dir) {
    memset(S, 0, sizeof(*S));
    S->native_fd = -1;
    char native_path[2048];
    if (snprintf(native_path, sizeof(native_path), "%s/model.coli", snap_dir) >= (int)sizeof(native_path)) {
        fprintf(stderr, "model path too long\n"); exit(1); }
    struct stat native_stat;
    if (!stat(native_path, &native_stat) && S_ISREG(native_stat.st_mode)) {
        st_native_init(S, native_path); return;
    }
    S->cap = 4096; S->t = calloc(S->cap, sizeof(st_tensor));
    /* raccoglie ordinatamente i nomi dei file shard */
    static char files[ST_MAX_SHARDS][1024]; int nf = 0;
    DIR *d = opendir(snap_dir); struct dirent *e;
    if (!d) { perror(snap_dir); exit(1); }
    while ((e = readdir(d))) {
        const char *dot = strrchr(e->d_name, '.');
        if (dot && !strcmp(dot, ".safetensors")) {  /* model.safetensors o model-0000N-of-... */
            if (nf >= ST_MAX_SHARDS) { fprintf(stderr, "too many shards (>%d): raise ST_MAX_SHARDS\n", ST_MAX_SHARDS); exit(1); }
            snprintf(files[nf++], 1024, "%s/%s", snap_dir, e->d_name);
        }
    }
    closedir(d);
    for (int a = 0; a < nf; a++) for (int b = a+1; b < nf; b++)
        if (strcmp(files[a], files[b]) > 0) { char tmp[1024]; strcpy(tmp, files[a]); strcpy(files[a], files[b]); strcpy(files[b], tmp); }

    for (int fi = 0; fi < nf; fi++) {
        int fd = st_open_fd(S, files[fi]);
        struct stat sst;
        if (fstat(fd, &sst) != 0) { perror("fstat shard"); exit(1); }
        int64_t fsz = (int64_t)sst.st_size;
        uint64_t hlen;
        if (pread(fd, &hlen, 8, 0) != 8) { perror("pread hlen"); exit(1); }
        /* file malevolo/troncato: hlen deve stare nel file dopo gli 8 byte di
         * prefisso e sotto il tetto. Senza questo bound hlen+1 puo' andare in
         * overflow (malloc(0) e poi hdr[hlen]=0 fuori limiti) o forzare una
         * malloc gigante. */
        if (fsz < 8 || hlen > (uint64_t)(fsz - 8) || hlen > (uint64_t)ST_MAX_HEADER) {
            fprintf(stderr, "%s: bad safetensors header length %llu (file %lld bytes)\n",
                    files[fi], (unsigned long long)hlen, (long long)fsz); exit(1); }
        char *hdr = malloc(hlen + 1);
        if (!hdr) { perror("malloc safetensors header"); exit(1); }
        if (pread(fd, hdr, hlen, 8) != (ssize_t)hlen) { perror("pread hdr"); exit(1); }
        hdr[hlen] = 0;
        int64_t data_start = 8 + (int64_t)hlen;
        char *arena = NULL;
        jval *root = json_parse(hdr, &arena);
        if (!root || root->t != J_OBJ) {
            fprintf(stderr, "%s: safetensors header is not a JSON object\n", files[fi]); exit(1); }
        for (int i = 0; i < root->len; i++) {
            const char *name = root->keys[i];
            if (!strcmp(name, "__metadata__")) continue;
            jval *m = root->kids[i];
            jval *dt = json_get(m, "dtype");
            jval *off = json_get(m, "data_offsets");
            jval *shp = json_get(m, "shape");
            /* un header crafted puo' omettere i campi o dare tipi sbagliati:
             * senza questi guard si dereferenzia NULL (json_get) o si legge
             * off->kids[0/1] oltre i limiti dell'array. */
            if (!dt || dt->t != J_STR || !off || off->t != J_ARR || off->len < 2 ||
                !shp || shp->t != J_ARR) {
                fprintf(stderr, "%s: tensor '%s' has malformed dtype/data_offsets/shape\n",
                        files[fi], name); exit(1); }
            int64_t a0 = (int64_t)off->kids[0]->num, b0 = (int64_t)off->kids[1]->num;
            /* offset dichiarati dal file: non-negativi, ordinati e dentro al
             * file. Altrimenti nbytes=b0-a0 diventa negativo -> malloc((size_t))
             * gigante e la memcpy in st_read_f32 sfora il buffer del chiamante;
             * oppure off punta fuori dal file. */
            if (a0 < 0 || b0 < a0 || data_start + b0 > fsz) {
                fprintf(stderr, "%s: tensor '%s' data_offsets [%lld,%lld] out of file bounds (%lld)\n",
                        files[fi], name, (long long)a0, (long long)b0, (long long)fsz); exit(1); }
            int64_t numel = 1; for (int k = 0; k < shp->len; k++) numel *= (int64_t)shp->kids[k]->num;
            if (S->n == S->cap) { S->cap *= 2; S->t = realloc(S->t, S->cap*sizeof(st_tensor)); }
            st_tensor *t = &S->t[S->n++];
            t->name = strdup(name); t->fd = fd; t->off = data_start + a0;
            t->nbytes = b0 - a0; t->dtype = st_dtype_code(dt->str); t->numel = numel;
        }
        free(arena); /* i jval restano leakati: ok, una tantum all'avvio */
        free(hdr);
    }
    /* indice hash costruito a fine indicizzazione (gli indici restano validi dopo i realloc) */
    st_build_hash(S);
}

static st_tensor *st_find(shards *S, const char *name) {
    if (S->hidx) {
        uint64_t h = st_hash(name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) {
            st_tensor *t = &S->t[S->hidx[h]];
            if (!strcmp(t->name, name)) return t;
            h = (h + 1) & (S->hcap - 1);
        }
        return NULL;
    }
    for (int i = 0; i < S->n; i++) if (!strcmp(S->t[i].name, name)) return &S->t[i];
    return NULL;
}
static int st_has(shards *S, const char *name) { return st_find(S, name) != NULL; }

/* prefetch ASINCRONO: dice al kernel di iniziare a leggere le pagine del tensore in
 * background (readahead). Serve a sovrapporre l'I/O degli expert col calcolo: si
 * prefetcha tutto il set di expert di un layer, poi le pread sincrone trovano la cache
 * gia' calda. No-op se il tensore non esiste (es. il primo .qs prima della lettura). */
static void st_prefetch(shards *S, const char *name) {
    st_tensor *t = st_find(S, name);
    if (t) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_WILLNEED);
}

/* legge un tensore in un buffer float32 fornito dal chiamante (numel float).
 * drop=1 -> consiglia al kernel di scartare le pagine (per gli expert in streaming). */
static int64_t st_read_f32(shards *S, const char *name, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    void *raw = malloc(t->nbytes);
    if (!raw) { fprintf(stderr, "malloc %lld bytes for tensor %s failed\n", (long long)t->nbytes, name); exit(1); }
    if (pread(t->fd, raw, t->nbytes, t->off) != t->nbytes) { perror("pread data"); exit(1); }
    if (t->dtype == 2) {
        memcpy(out, raw, t->nbytes);
    } else if (t->dtype == 0) {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = bf16_to_f32(p[i]);
    } else {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = f16_to_f32(p[i]);
    }
    free(raw);
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
    return t->numel;
}

static int64_t st_numel(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->numel : -1;
}
static int64_t st_nbytes(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->nbytes : -1;
}

/* legge i byte GREZZI di un tensore (nessuna conversione di dtype): per i pesi gia'
 * quantizzati int4/int8 del nostro container (dtype U8). drop=1 -> fadvise DONTNEED. */
static void st_read_raw(shards *S, const char *name, void *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (pread(t->fd, out, t->nbytes, t->off) != t->nbytes) { perror("pread raw"); exit(1); }
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
}

/* legge una FETTA di un tensore: n_elems a partire dall'elemento elem_off.
 * Serve per gli expert fusi di GLM (un tensore = blocco [E, ...]): si legge il
 * solo expert richiesto via pread del sotto-range, niente lettura dell'intero blocco. */
static void st_read_slice_f32(shards *S, const char *name, int64_t elem_off, int64_t n_elems, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    int esz = (t->dtype == 2) ? 4 : 2;
    int64_t boff = t->off + elem_off * esz, nb = n_elems * esz;
    void *raw = malloc(nb);
    if (pread(t->fd, raw, nb, boff) != nb) { perror("pread slice"); exit(1); }
    if (t->dtype == 2) memcpy(out, raw, nb);
    else if (t->dtype == 0) { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = bf16_to_f32(p[i]); }
    else { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = f16_to_f32(p[i]); }
    free(raw);
    if (drop) posix_fadvise(t->fd, boff, nb, POSIX_FADV_DONTNEED);
}

#endif
