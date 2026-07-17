#ifndef COLIBRI_TIER_H
#define COLIBRI_TIER_H

#include <stdint.h>

/* Pick one RAM/VRAM hot-store slot to replace from recent routing heat.
 * The fixed margin handles tiny samples; the 25% margin prevents ping-pong. */
static int tier_pick_swap(const uint32_t *heat, int nexpert,
                          const int *pinned, int npin,
                          int *slot, int *eid, long *gain){
    if(!heat || !pinned || npin<1 || nexpert<1) return 0;
    int cold=0;
    for(int z=1;z<npin;z++) if(heat[pinned[z]]<heat[pinned[cold]]) cold=z;
    int hot=-1; uint32_t fh=0;
    for(int e=0;e<nexpert;e++){
        int resident=0;
        for(int z=0;z<npin;z++) if(pinned[z]==e){ resident=1; break; }
        if(!resident && heat[e]>fh){ fh=heat[e]; hot=e; }
    }
    if(hot<0) return 0;
    uint32_t fc=heat[pinned[cold]];
    if(fh<=fc+(fc>>2)+4) return 0;
    *slot=cold; *eid=hot; *gain=(long)fh-(long)fc;
    return 1;
}

/* LFRU: frequency is the primary signal; recency breaks close calls. A recent
 * access contributes at most 255 points while one frequency count is worth
 * 256, so a merely recent expert cannot displace a genuinely hotter one. */
static uint64_t tier_lfru_score(uint32_t heat, uint32_t last, uint32_t clock){
    uint32_t age=clock-last, recent=age<255?255-age:0;
    return ((uint64_t)heat<<8)|recent;
}

static int tier_pick_lfru(const uint32_t *heat, const uint32_t *last, uint32_t clock,
                          int nexpert, const int *pinned, int npin,
                          int *slot, int *eid, long *gain){
    if(!heat||!last||!pinned||npin<1||nexpert<1) return 0;
    int cold=0;
    for(int z=1;z<npin;z++)
        if(tier_lfru_score(heat[pinned[z]],last[pinned[z]],clock)<
           tier_lfru_score(heat[pinned[cold]],last[pinned[cold]],clock)) cold=z;
    int hot=-1; uint64_t hs=0;
    for(int e=0;e<nexpert;e++){
        int resident=0; for(int z=0;z<npin;z++) if(pinned[z]==e){resident=1;break;}
        uint64_t score=tier_lfru_score(heat[e],last[e],clock);
        if(!resident&&(hot<0||score>hs)){ hot=e; hs=score; }
    }
    if(hot<0) return 0;
    uint64_t cs=tier_lfru_score(heat[pinned[cold]],last[pinned[cold]],clock);
    /* Retain the existing 25%+4-frequency hysteresis in score units. */
    if(hs<=cs+(cs>>2)+(4u<<8)) return 0;
    *slot=cold; *eid=hot; *gain=(long)((hs-cs)>>8); return 1;
}

/* ---- streaming-cache policy: adaptive frequency protection (SLRU) ----
 * Per-expert frequency with a deliberate lifecycle: freq saturates at
 * TIER_FMAX and is NEVER decayed while live — forgetting happens only through
 * the bounded ghost turnover in tier_evict, at a rate proportional to real
 * eviction traffic (never wall-clock or periodic halving). Victims are LRU
 * among experts at freq <= k; k self-tunes from the graduation rate (of the
 * experts evicted under pressure, what fraction first proved reuse by
 * crossing k), and the graduation-rate thresholds themselves self-tune from
 * a hit-rate gradient. Only the adaptation statistics are ever halved. */

#define TIER_FMAX          15   /* freq saturation */
#define TIER_K0             2   /* initial protection bar */
#define TIER_ADAPT_EVERY   16u  /* per-layer evictions between k re-checks */
#define TIER_WIN_OPS       32u  /* probes per hit-rate learning window — keeps the
                                 * reference's 1:2 eviction:probe ratio so ~one k
                                 * step lands per measured window even at 50% miss */
#define TIER_RATE_LOW0   2500u  /* graduation-rate thresholds, x10000 */
#define TIER_RATE_HIGH0  5000u
#define TIER_RATE_LOW_MIN   500u
#define TIER_RATE_LOW_MAX  4000u
#define TIER_RATE_HIGH_MIN 3000u
#define TIER_RATE_HIGH_MAX 8000u
#define TIER_RATE_STEP     1000u

typedef struct {
    int8_t k, last_dir;
    uint32_t ev_unprot, ev_prot, graduated, last_check;
    uint32_t rate_low, rate_high;      /* learned thresholds, x10000 */
    uint32_t win_hits, win_ops, prev_hitrate;
} TierAdapt;

static void tier_adapt_init(TierAdapt *a){
    a->k=TIER_K0; a->last_dir=0;
    a->ev_unprot=a->ev_prot=a->graduated=a->last_check=0;
    a->rate_low=TIER_RATE_LOW0; a->rate_high=TIER_RATE_HIGH0;
    a->win_hits=a->win_ops=a->prev_hitrate=0;
}

/* every streamed-cache probe (hit or miss) feeds the hit-rate window */
static void tier_probe(TierAdapt *a, int hit){ a->win_ops++; if(hit) a->win_hits++; }

/* cache hit: bump freq, saturating. Crossing k while the cache is at
 * capacity is a graduation — the expert proved reuse under eviction
 * pressure, the signal that drives k. */
static void tier_touch(TierAdapt *a, int8_t *f, int at_cap){
    int8_t v=*f;
    if(v<1||v>=TIER_FMAX) return;
    if(v==a->k && at_cap) a->graduated++;
    *f=(int8_t)(v+1);
}

/* admission: a returning ghost resumes at remembered+1, a new expert at 1 */
static void tier_admit(int8_t *f){
    int v=*f;
    *f = v<0 ? (int8_t)(-v+1>TIER_FMAX?TIER_FMAX:-v+1) : (int8_t)1;
}

/* eviction: the freq is negated, not erased — a ghost that remembers how much
 * reuse the expert had proven. Ghost memory is bounded by census: at ghost_cap
 * the least-recently-routed EXISTING ghost is forgotten BEFORE the victim is
 * ghosted. Order matters: the victim is picked as LRU, so its recency is
 * near-oldest in the layer — trimming after ghosting would forget the new
 * ghost immediately, freezing ghost memory on stale entries. */
static void tier_evict(TierAdapt *a, int8_t *freq, int nexpert,
                       const uint32_t *last, int eid, int ghost_cap){
    if(freq[eid]>a->k) a->ev_prot++; else a->ev_unprot++;
    int ng=0, old=-1, first=1; uint32_t oldl=0;
    for(int e=0;e<nexpert;e++) if(e!=eid && freq[e]<0){
        ng++;
        if(first || last[e]<oldl){ old=e; oldl=last[e]; first=0; }
    }
    if(ng>=ghost_cap && old>=0) freq[old]=0;
    if(freq[eid]>0) freq[eid]=(int8_t)-freq[eid];
}

/* Two feedback loops: the graduation rate drives k, and the hit-rate gradient
 * tunes the rate thresholds so the policy adapts to the workload instead of
 * fixed cutoffs. The check is armed every TIER_ADAPT_EVERY evictions, but k
 * MOVES only on a completed hit-rate window: strict change -> measure ->
 * learn alternation, so every k step is evaluated over one full window
 * before the next (an armed check on an incomplete window does nothing —
 * eviction pressure alone can no longer outrun the measurement).
 * Note the deliberate unsigned wrap: the halving at the bottom leaves
 * last_check ABOVE the halved total, so the next eviction wraps to a huge
 * delta and re-arms the check early; the window gate still paces the actual
 * k steps. It reads like a bug; it is not. Don't "fix" it. */
static void tier_maybe_adapt(TierAdapt *a){
    uint32_t total=a->ev_unprot+a->ev_prot;
    if(total-a->last_check<TIER_ADAPT_EVERY) return;
    a->last_check=total;
    uint32_t grad=a->graduated;
    if(!total){ if(grad>100) a->graduated=grad/2; return; }
    if(a->win_ops<TIER_WIN_OPS) return;
    uint32_t hits=a->win_hits>a->win_ops?a->win_ops:a->win_hits;
    uint32_t hr=(uint32_t)((uint64_t)hits*10000u/a->win_ops);
    if(a->prev_hitrate>0 && a->last_dir!=0){
        int up=hr>a->prev_hitrate;
        if(a->last_dir>0){         /* last change raised k */
            if(up){ if(a->rate_high>TIER_RATE_HIGH_MIN+TIER_RATE_STEP) a->rate_high-=TIER_RATE_STEP; }
            else if(a->rate_high<TIER_RATE_HIGH_MAX-TIER_RATE_STEP) a->rate_high+=TIER_RATE_STEP;
        }else{                     /* last change lowered k */
            if(up){ if(a->rate_low<TIER_RATE_LOW_MAX-TIER_RATE_STEP) a->rate_low+=TIER_RATE_STEP; }
            else if(a->rate_low>TIER_RATE_LOW_MIN+TIER_RATE_STEP) a->rate_low-=TIER_RATE_STEP;
        }
    }
    a->prev_hitrate=hr; a->win_ops=0; a->win_hits=0;
    uint32_t rate=(uint32_t)((uint64_t)grad*10000u/total);
    int8_t dir=0;
    if(rate<a->rate_low && a->k>1){ a->k--; dir=-1; }
    else if(rate>a->rate_high && a->k<TIER_FMAX-1){ a->k++; dir=1; }
    a->last_dir=dir;
    if(grad>100) a->graduated=grad/2;
    if(total>100){ a->ev_unprot/=2; a->ev_prot/=2; }  /* last_check kept: arms the wrap re-check */
}

static void tier_decay(uint32_t *heat, int nexpert){
    for(int e=0;e<nexpert;e++) heat[e]>>=1;
}

#endif
