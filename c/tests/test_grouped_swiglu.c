/* The fused grouped-int4 gate/up/SwiGLU kernel must reproduce the existing
 * grouped matmuls plus elementwise activation exactly for decode and batches. */
#define main coli_glm_main_unused
#include "../glm.c"
#undef main

static uint32_t rng_state=0x9e3779b9u;
static uint32_t xr(void){
    rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5;
    return rng_state;
}

static int check_case(int S,int I,int O,int gs){
    int rb=(I+1)/2,ng=(I+gs-1)/gs;
    size_t wn=(size_t)O*rb,sn=(size_t)O*ng,xn=(size_t)S*I,yn=(size_t)S*O;
    uint8_t *qg=malloc(wn),*qu=malloc(wn);
    float *sg=malloc(sn*sizeof(float)),*su=malloc(sn*sizeof(float));
    float *x=malloc(xn*sizeof(float)),*g=malloc(yn*sizeof(float));
    float *u=malloc(yn*sizeof(float)),*want=malloc(yn*sizeof(float)),*got=malloc(yn*sizeof(float));
    if(!qg||!qu||!sg||!su||!x||!g||!u||!want||!got){ fprintf(stderr,"OOM\n"); return 1; }
    for(size_t i=0;i<wn;i++){ qg[i]=(uint8_t)xr(); qu[i]=(uint8_t)xr(); }
    for(size_t i=0;i<sn;i++){
        sg[i]=0.0005f+(float)(xr()%1000)*1e-6f;
        su[i]=0.0005f+(float)(xr()%1000)*1e-6f;
    }
    for(size_t i=0;i<xn;i++) x[i]=((float)(xr()%4001)-2000.f)/400.f;

    matmul_i4_grouped(g,x,qg,sg,S,I,O,gs);
    matmul_i4_grouped(u,x,qu,su,S,I,O,gs);
    for(size_t i=0;i<yn;i++) want[i]=siluf(g[i])*u[i];
    matmul_i4_grouped_swiglu(got,x,qg,sg,qu,su,S,I,O,gs);

    int rc=0;
    for(size_t i=0;i<yn;i++) if(memcmp(&got[i],&want[i],sizeof(float))!=0){
        fprintf(stderr,"FAIL S=%d I=%d O=%d gs=%d idx=%zu: %.9g != %.9g\n",
                S,I,O,gs,i,(double)got[i],(double)want[i]);
        rc=1; break;
    }
    free(qg); free(qu); free(sg); free(su); free(x); free(g); free(u); free(want); free(got);
    return rc;
}

int main(void){
    static const int Ss[]={1,2,5};
    static const int Is[]={16,127,128,255,1408};
    static const int Os[]={1,7,64,65};
    static const int Gs[]={16,128};
    for(unsigned s=0;s<sizeof Ss/sizeof Ss[0];s++)
        for(unsigned i=0;i<sizeof Is/sizeof Is[0];i++)
            for(unsigned o=0;o<sizeof Os/sizeof Os[0];o++)
                for(unsigned g=0;g<sizeof Gs/sizeof Gs[0];g++)
                    if(check_case(Ss[s],Is[i],Os[o],Gs[g])) return 1;
    puts("grouped gate/up/SwiGLU exactness: ok");
    return 0;
}
