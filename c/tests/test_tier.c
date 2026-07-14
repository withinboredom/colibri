#include <stdio.h>
#include "../tier.h"

static int fail(const char *message){
    fprintf(stderr,"tier test failed: %s\n",message);
    return 1;
}

int main(void){
    uint32_t heat[6]={20,2,8,3,30,1};
    int pinned[2]={0,1}, slot=-1, eid=-1; long gain=0;
    if(!tier_pick_swap(heat,6,pinned,2,&slot,&eid,&gain)) return fail("hot expert not promoted");
    if(slot!=1 || eid!=4 || gain!=28) return fail("wrong promotion candidate");

    uint32_t stable[4]={20,18,24,4}; int resident[2]={0,1};
    if(tier_pick_swap(stable,4,resident,2,&slot,&eid,&gain)) return fail("hysteresis did not block churn");

    tier_decay(heat,6);
    if(heat[0]!=10 || heat[1]!=1 || heat[4]!=15) return fail("heat decay");

    uint32_t freq[5]={10,10,2,18,18}, last[5]={10,90,95,20,99};
    int live[2]={0,1};
    if(!tier_pick_lfru(freq,last,100,5,live,2,&slot,&eid,&gain)) return fail("LFRU promotion");
    if(slot!=0||eid!=4) return fail("LFRU did not prefer recent ties");

    if(tier_cache_score(1,999)>=tier_cache_score(5,10))
        return fail("cache score: one-shot did not rank below established");
    if(tier_cache_score(1000,5)>=tier_cache_score(20,6))
        return fail("cache score: saturated heat tie not broken by LRU");
    if(tier_cache_score(15,7)!=tier_cache_score(99,7))
        return fail("cache score: heat not capped at 15");
    if(tier_cache_score(0,3)!=tier_cache_score(1,3))
        return fail("cache score: unrouted resident not floored to one-shot class");
    puts("tier tests: ok");
    return 0;
}
