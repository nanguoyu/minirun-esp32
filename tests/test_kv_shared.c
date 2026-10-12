#include "kv_file.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
static uint32_t rng=17;
static float value(void) { rng=rng*1664525u+1013904223u; return ((int)(rng>>16)-32768)/8192.0f; }
int main(void) {
    assert(LFM_KV_SHARED);
    char dir[]="/tmp/lfm-shared-XXXXXX"; assert(mkdtemp(dir));
    LfmKVFile *a=calloc(1,sizeof(*a)), *b=calloc(1,sizeof(*b));
    Lfm *ma=calloc(1,sizeof(*ma)), *mb=calloc(1,sizeof(*mb));
    unsigned char *ia=malloc(LFM_IO), *ib=malloc(LFM_IO);
    float *q=malloc(8*1024*4), *kv=malloc(8*1024*4), *x=malloc(8*1024*4), *y=malloc(8*1024*4);
    assert(a&&b&&ma&&mb&&ia&&ib&&q&&kv&&x&&y);
    assert(!lfm_kv_file_open(a,dir,ia)&&!lfm_kv_file_open(b,dir,ib));
    ma->attention_opaque=a;mb->attention_opaque=b;
    const int sizes[]={1,7,3,8};int round=0;
    for(int base=0;base<LFM_CONTEXT;) {
        int n=sizes[round++%4];if(n>LFM_CONTEXT-base)n=LFM_CONTEXT-base;
        for(int j=0;j<n*1024;j++){q[j]=value();kv[j]=value();}
        for(int j=0;j<n;j++)
            assert(!lfm_kv_file_attention(ma,0,base+j,q+j*1024,kv+j*1024,kv+j*1024+512,x+j*1024));
        assert(!lfm_kv_file_attention_many(mb,0,base,n,q,1024,kv,kv+512,1024,y,1024));
        assert(!memcmp(x,y,n*1024*4));
        base+=n;
    }
    assert(a->write_bytes==b->write_bytes&&b->read_bytes<a->read_bytes);
    printf("PASS bits=%d every query causal and bitwise; scalar_read=%llu shared_read=%llu\n",LFM_KV_BITS,(unsigned long long)a->read_bytes,(unsigned long long)b->read_bytes);
    assert(!lfm_kv_file_close(a)&&!lfm_kv_file_close(b));
    /* Appending a later page must not conceal a truncated earlier page with a sparse hole. */
    assert(!lfm_kv_file_open(b,dir,ib));
    for(int base=0;base<(int)LFM_KV_PAGE_POS;base+=8)
        assert(!lfm_kv_file_attention_many(mb,0,base,8,q,1024,kv,kv+512,1024,y,1024));
    assert(!ftruncate(b->fd,0));
    assert(lfm_kv_file_attention_many(mb,0,LFM_KV_PAGE_POS,8,q,1024,kv,kv+512,1024,y,1024)==-1);
    struct stat st;assert(!fstat(b->fd,&st)&&st.st_size==0&&b->failed&&mb->failed);
    assert(!lfm_kv_file_close(b));assert(!rmdir(dir));
    puts("PASS truncation cannot be hidden by subsequent shared-batch writes");
    free(a);free(b);free(ma);free(mb);free(ia);free(ib);free(q);free(kv);free(x);free(y);
}
