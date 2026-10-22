// Opt-in convex-cell search. Safety/planes are supplied by the frozen Numba
// predicates through C ABI callbacks. Build WITHOUT fast-math or FMA contraction.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <vector>
#include <chrono>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace {
constexpr int V=128, P=2048;
constexpr double INF=std::numeric_limits<double>::infinity();
constexpr double NANV=std::numeric_limits<double>::quiet_NaN();
#ifdef EA_PROFILE
struct Profile { bool enabled=false;std::array<uint64_t,8>calls{},ns{}; };
thread_local Profile profile;
struct Scope {int key;bool active;std::chrono::steady_clock::time_point start;
    explicit Scope(int k):key(k),active(profile.enabled){if(active)start=std::chrono::steady_clock::now();}
    ~Scope(){if(active){++profile.calls[key];profile.ns[key]+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();}}
};
#else
struct Scope {explicit Scope(int){}};
#endif
using Cert=int(*)(double*,double,double,double,double,double,double,int64_t,double*,int64_t*);
using Planes=void(*)(double*,double,double,double,double,double*);
using Refine=double(*)(double*,double,double,double,double,double,double);
using Sep=double(*)(double*,double,double,double,double,double);
using Circle=int(*)(double,double*,double*,int64_t*,int64_t,double*);
using Inside=int(*)(double,double*,int64_t,double*);
struct Point { double x,y; };
struct Seeds {
    std::array<double,14> state;
    double horizon,cap,tolerance,guard;
    std::vector<double> times;
    Point upper{NANV,NANV};bool has_upper=false;
    double wa=0.,wb=0.;bool has_pair=false;
    Seeds(const double*s,double h,double c,double t,double g):horizon(h),cap(c),tolerance(t),guard(g){
        std::copy_n(s,14,state.data());times.reserve(P);
    }
    bool matches(const double*s,double h,double c,double t,double g)const{
        return h==horizon&&c==cap&&t==tolerance&&g==guard&&std::equal(state.begin(),state.end(),s);
    }
};
using Mask=std::array<uint64_t,P/64>;
struct Node { std::array<Point,V> v; int count; double lower,hint_time; };
struct Item { double lower; int id; int64_t serial; };
struct Compare { bool operator()(const Item&a,const Item&b)const {
    return a.lower!=b.lower ? a.lower>b.lower : a.serial>b.serial;
}};
int clip(const Point*v,int count,double nx,double ny,double b,Point*out){
    Scope measured(1);
    int size=0;
    for(int j=0;j<count;++j){
        const Point p=v[j?j-1:count-1],q=v[j];
        const double fp=nx*p.x+ny*p.y-b,fq=nx*q.x+ny*q.y-b;
        const bool pin=fp<=0.,qin=fq<=0.;
        if(pin!=qin){const double a=fp/(fp-fq);out[size++]={p.x+a*(q.x-p.x),p.y+a*(q.y-p.y)};}
        if(qin)out[size++]=q;
    }
    return size;
}
double nearest(const Point*v,int count,Point&where){
    Scope measured(0);
    bool inside=true,strict_side=false;double best_squared=INF,best_robust=INF;Point robust_where{NANV,NANV};
    for(int j=0;j<count;++j){
        const Point p=v[j],q=v[(j+1)%count];
        const double dx=q.x-p.x,dy=q.y-p.y,squared=dx*dx+dy*dy;
        const double side=dx*(-p.y)-dy*(-p.x);
        if(side<0.)inside=false;
        if(side!=0.)strict_side=true;
        const double a=squared?std::max(0.,std::min(1.,-(p.x*dx+p.y*dy)/squared)):0.;
        const Point x{p.x+a*dx,p.y+a*dy};const double norm_squared=x.x*x.x+x.y*x.y;
        // Squared distances preserve the ordering on this bounded polygon.
        // Take ONE square root after selecting an edge, not hypot per edge.
        // Keep a robust path when squaring overflows on a nonstandard cap.
        if(std::isfinite(norm_squared)){if(norm_squared<best_squared){best_squared=norm_squared;where=x;}}
        else {const double norm=std::hypot(x.x,x.y);if(norm<best_robust){best_robust=norm;robust_where=x;}}
    }
    // A collapsed offset point/line has all side tests zero. It is not an
    // enclosing polygon; use its segment distance rather than claiming zero.
    if(inside&&strict_side){where={0.,0.};return 0.;}
    if(std::isfinite(best_squared))return std::sqrt(best_squared);
    where=robust_where;return best_robust;
}
struct Pool {
    std::array<double,P*16> n;
    std::array<double,P*8> b;
    std::array<double,P> times;
    std::array<int64_t,P> counts;
    std::array<int,P> faces;
    std::array<double,P> hint_x,hint_y,hint_b;
    int count=0;
    int store(double*s,double wa,double wb,double t,double guard,Planes planes,Point hint={0.,0.}){
        Scope measured(3);
        if(!(t>0.))return -1;
        for(int k=0;k<count;++k)if(std::abs(times[k]-t)<1e-12)return k;
        if(count==P)return -1;
        double raw[24];planes(s,wa,wb,t,guard,raw);
        const int k=count++;times[k]=t;counts[k]=8;
        std::copy(raw,raw+16,n.data()+k*16);std::copy(raw+16,raw+24,b.data()+k*8);
        // Predicate ordering only: the facet nearest the rejecting point is
        // a better first separating-axis hint than always choosing facet0.
        int face=0;double gap=-INF;
        for(int j=0;j<8;++j){const double value=raw[2*j]*hint.x+raw[2*j+1]*hint.y-raw[16+j];
            if(value>gap){gap=value;face=j;}}
        faces[k]=face;hint_x[k]=raw[2*face];hint_y[k]=raw[2*face+1];hint_b[k]=raw[16+face];return k;
    }
    int cached(Point x,double err,const Mask*excluded=nullptr){
        Scope measured(2);
        const double best=-err;
        auto contains=[&](int k){
            const int first=faces[k];int face=first;
            double gap=n[k*16+first*2]*x.x+n[k*16+first*2+1]*x.y-b[k*8+first];
            if(gap>=best)return false;
            for(int j=0;j<8;++j){
                if(j==first)continue;
                const double value=n[k*16+j*2]*x.x+n[k*16+j*2+1]*x.y-b[k*8+j];
                if(value>gap){gap=value;face=j;}
                if(gap>=best)break;
            }
            faces[k]=face;
            hint_x[k]=n[k*16+face*2];hint_y[k]=n[k*16+face*2+1];hint_b[k]=b[k*8+face];
            // Any STRICTLY containing polygon supplies a necessary escape
            // disjunction. Prefer a recently learned cut and avoid scanning
            // all older cuts just to maximize a heuristic depth. The global
            // frontier and every safety/precision check stay unchanged.
            return gap<best;
        };
        int end=count;
#if defined(__AVX512F__)
        // Identical float64 multiply/add/subtract operations on eight cached
        // face hints at a time. No FMA, approximate instruction, or relaxed
        // predicate. Every possible hit still checks all eight exact planes.
        const __m512d xx=_mm512_set1_pd(x.x),yy=_mm512_set1_pd(x.y),threshold=_mm512_set1_pd(best);
        while(end>=8){
            const int start=end-8;
            unsigned allowed=255;
            if(excluded){
                const int shift=start%64,word=start/64;
                uint64_t bits=(*excluded)[word]>>shift;
                if(shift>56)bits|=(*excluded)[word+1]<<(64-shift);
                allowed&=~unsigned(bits);
                if(!allowed){end=start;continue;}
            }
            const __m512d gap=_mm512_sub_pd(_mm512_add_pd(_mm512_mul_pd(_mm512_loadu_pd(hint_x.data()+start),xx),
                _mm512_mul_pd(_mm512_loadu_pd(hint_y.data()+start),yy)),_mm512_loadu_pd(hint_b.data()+start));
            const unsigned hits=_mm512_cmp_pd_mask(gap,threshold,_CMP_LT_OQ)&allowed;
            for(int j=7;j>=0;--j)if((hits&(1u<<j))&&contains(start+j))return start+j;
            end=start;
        }
#endif
        for(int k=end-1;k>=0;--k){
            if(excluded&&((*excluded)[k/64]&(uint64_t(1)<<(k%64))))continue;
            if(contains(k))return k;
        }
        return -1;
    }
};
struct Callbacks { Cert initial,proposal,main;Planes planes;Refine refine;Sep sep;Circle circle;Inside inside; };
struct Result {
    double value=INF,ux=NANV,uy=NANV,lower=0.;bool verified=false,converged=false;
    int64_t iterations=0,cuts=0,rays=0,nodes=0,cn=0;int status=5;
    void output(double*out)const{
        const double a[12]={value,ux,uy,lower,double(verified),double(converged),double(iterations),
            double(cuts),double(rays),double(nodes),double(cn),double(status)};std::copy(a,a+12,out);
    }
};
int certify(Cert f,double*s,double wa,double wb,Point a,double horizon,double guard,int64_t limit,
            double&witness,Result&r){Scope measured(4);int64_t visited=0;const int status=f(s,wa,wb,a.x,a.y,horizon,guard,limit,&witness,&visited);r.cn+=visited;return status;}
int circle_call(Circle f,double radius,Pool&p,double*out){Scope measured(5);return f(radius,p.n.data(),p.b.data(),p.counts.data(),p.count,out);}
int inside_call(Inside f,double radius,Point*vertices,int count,double*out){Scope measured(6);return f(radius,reinterpret_cast<double*>(vertices),count,out);}
double refine_call(Refine f,double*s,double wa,double wb,Point a,double t,double horizon){Scope measured(7);return f(s,wa,wb,a.x,a.y,t,horizon);}
}

extern "C" int ea_cells_native_abi(){return 1;}
extern "C" int ea_cells_native_seed_abi(){return 1;}
extern "C" void*ea_cells_native_seed_create(double*s,double h,double cap,double tol,double guard){
    try{return new Seeds(s,h,cap,tol,guard);}catch(...){return nullptr;}
}
extern "C" void ea_cells_native_seed_destroy(void*ptr){delete reinterpret_cast<Seeds*>(ptr);}
extern "C" void ea_cells_profile_enable(int enabled){
#ifdef EA_PROFILE
    profile.enabled=enabled!=0;profile.calls.fill(0);profile.ns.fill(0);
#else
    (void)enabled;
#endif
}
extern "C" void ea_cells_profile_read(uint64_t*out){
#ifdef EA_PROFILE
    std::copy(profile.calls.begin(),profile.calls.end(),out);std::copy(profile.ns.begin(),profile.ns.end(),out+8);
#else
    std::fill(out,out+16,0);
#endif
}
extern "C" int ea_cells_native_clip(double*vertices,int count,double nx,double ny,double b,double*out){
    if(count<3||count>V)return -1;
    return clip(reinterpret_cast<Point*>(vertices),count,nx,ny,b,reinterpret_cast<Point*>(out));
}
extern "C" double ea_cells_native_nearest(double*vertices,int count,double*out){
    if(count<3||count>V)return NANV;
    Point p;const double value=nearest(reinterpret_cast<Point*>(vertices),count,p);out[0]=p.x;out[1]=p.y;return value;
}
void solve_native(double*s,double wa,double wb,double horizon,double cap,double tolerance,
    double guard,int max_nodes,int max_vertices,int max_iterations,int cert_limit,
    void*initial,void*proposal,void*main,void*planes,void*refine,void*sep,void*circle,void*inside,Seeds*seeds,double*out){
    Result r;
    try{
        // A context lives for ONE current-input four-mode call only. A
        // mismatch disables hints; no different state's answer can be read.
        if(seeds&&!seeds->matches(s,horizon,cap,tolerance,guard))seeds=nullptr;
        // Do not pay for old hints when the nominal models differ strongly.
        // This only selects a proposal strategy, not a different EA problem.
        const double model_change=seeds&&seeds->has_pair?std::max(std::abs(wa-seeds->wa),std::abs(wb-seeds->wb)):0.;
        const bool very_close=model_change*horizon<=.01;
        if(seeds&&seeds->has_pair&&!very_close){
            const double relevant=seeds->times.empty()?horizon:*std::max_element(seeds->times.begin(),seeds->times.end());
            if(model_change*relevant>.1)seeds=nullptr;
        }
        const Callbacks f{reinterpret_cast<Cert>(initial),reinterpret_cast<Cert>(proposal),reinterpret_cast<Cert>(main),
            reinterpret_cast<Planes>(planes),reinterpret_cast<Refine>(refine),reinterpret_cast<Sep>(sep),
            reinterpret_cast<Circle>(circle),reinterpret_cast<Inside>(inside)};
        double witness=NANV;const int initial_status=certify(f.initial,s,wa,wb,{0.,0.},horizon,guard,cert_limit,witness,r);
        if(initial_status==1){r.value=0.;r.verified=r.converged=true;r.status=0;r.output(out);return;}
        if(f.sep(s,wa,wb,0.,0.,0.)<=guard){r.lower=cap;r.status=3;r.output(out);return;}
        if(max_nodes<1||max_vertices<12||max_vertices>V||max_iterations<1){r.status=5;r.output(out);return;}
        auto storage=std::unique_ptr<Node[]>(new Node[max_nodes]);
        std::vector<int>free;free.reserve(max_nodes);for(int i=max_nodes-1;i>=1;--i)free.push_back(i);
        std::priority_queue<Item,std::vector<Item>,Compare>pending;
        storage[0].count=4;storage[0].lower=0.;storage[0].hint_time=witness;
        storage[0].v[0]={-cap,-cap};storage[0].v[1]={cap,-cap};storage[0].v[2]={cap,cap};storage[0].v[3]={-cap,cap};
        int64_t serial=0;pending.push({0.,0,serial++});r.nodes=1;
        const double err=2048.*std::numeric_limits<double>::epsilon()*(cap+1.);
        auto pool=std::unique_ptr<Pool>(new Pool);
        const double fractions[]={.25,.5,.75,1.,1.25,1.5,2.};
        for(double fraction:fractions)pool->store(s,wa,wb,std::min(horizon,witness*fraction),guard,f.planes);
        for(double fraction:{.25,.5,.75,1.})pool->store(s,wa,wb,horizon*fraction,guard,f.planes);
        const double omega=std::max(std::abs(wa),std::abs(wb));const bool fast=omega>5.;
        const int initial_count=fast?256:omega>.1?24:8;
        for(int j=0;j<=initial_count;++j){const double fraction=fast?.75+.5*j/initial_count:.25+2.*j/initial_count;
            pool->store(s,wa,wb,std::min(horizon,witness*fraction),guard,f.planes);}
        const int learned_begin=pool->count;
        std::vector<double> inherited;
        if(seeds){
            if(very_close||seeds->times.size()<=64)inherited=seeds->times;
            else {
                auto sorted=seeds->times;std::sort(sorted.begin(),sorted.end());
                for(int j=0;j<64;++j)inherited.push_back(sorted[j*(sorted.size()-1)/63]);
            }
        }
        for(double t:inherited){
            // ONLY exact times are inherited. Rebuild ALL halfplanes for
            // this pairing's own wa/wb, never inherit geometry or a bound.
            if(t>0.&&t<=horizon)pool->store(s,wa,wb,t,guard,f.planes,seeds->has_upper?seeds->upper:Point{0.,0.});
        }
        std::array<Point,V+2>cell,work,piece;
        auto update=[&](Point a){const double value=std::hypot(a.x,a.y);if(value<r.value){r.value=value;r.ux=value?a.x/value:NANV;r.uy=value?a.y/value:NANV;r.verified=true;}};
        if(seeds&&seeds->has_upper){
            const double norm=std::hypot(seeds->upper.x,seeds->upper.y);
            for(double extra:{0.,.9*tolerance}){
                if(!(norm>0.)||norm+extra>cap)continue;
                const double scale=(norm+extra)/norm;const Point candidate{seeds->upper.x*scale,seeds->upper.y*scale};
                double t=NANV;const int status=certify(f.proposal,s,wa,wb,candidate,horizon,guard,cert_limit,t,r);
                if(status==1){update(candidate);break;}
                if(status==0&&t>0.)pool->store(s,wa,wb,t,guard,f.planes,candidate);
            }
        }
        double recent_time=witness,cell_hint=witness;
        auto query=[&](Cert oracle,Point a,double&t){
            // A local search can only REJECT with an actual penetrating
            // instant. A failed hunt ALWAYS falls through to the unchanged
            // whole-horizon oracle, so sampling never certifies safety.
            if(fast&&recent_time>0.){
                // The parent collision time belongs to THIS frontier cell,
                // unlike the last query from an unrelated global branch.
                const double hint=cell_hint>0.?cell_hint:recent_time;
                // Only the heuristic's local search window is tightened.
                // The oracle below still certifies the full horizon.
                const double local=refine_call(f.refine,s,wa,wb,a,hint,std::min(horizon,10.*hint));
                if(local>0.&&f.sep(s,wa,wb,local,a.x,a.y)<-guard){
                    t=recent_time=local;++r.cn;return 0;
                }
            }
            const int status=certify(oracle,s,wa,wb,a,horizon,guard,cert_limit,t,r);
            if(status==0&&t>0.){
                const double critical=refine_call(f.refine,s,wa,wb,a,t,horizon);
                if(critical>0.)t=critical;recent_time=t;
            }
            return status;
        };
        auto finish=[&](int status,bool converged){r.status=status;r.converged=converged;
            if(converged&&std::isfinite(r.value))r.lower=std::max(0.,r.value-tolerance);
            if(seeds){
                seeds->wa=wa;seeds->wb=wb;seeds->has_pair=true;
                seeds->times.assign(pool->times.begin()+learned_begin,pool->times.begin()+pool->count);
                if(r.verified&&std::isfinite(r.value)&&r.value>0.){
                    seeds->upper={r.value*r.ux,r.value*r.uy};seeds->has_upper=true;
                }
            }
            r.output(out);};
        for(int iteration=0;iteration<max_iterations;++iteration){
            r.iterations=iteration;
            if(pending.empty()){if(std::isfinite(r.value))finish(1,true);else{r.lower=cap;finish(7,true);}return;}
            const Item item=pending.top();pending.pop();r.lower=item.lower;
            if(std::isfinite(r.value)&&r.value-r.lower<=tolerance){finish(1,true);return;}
            const int count=storage[item.id].count;cell_hint=storage[item.id].hint_time;
            std::copy_n(storage[item.id].v.data(),count,cell.data());free.push_back(item.id);
            Point point;const double radius=nearest(cell.data(),count,point);if(radius-err>cap)continue;
            int known=pool->cached(point,err);bool need_rescue=false;
            if(known<0){
                for(double fraction:{.9}){
                    const double target=radius+fraction*tolerance;double q[2];bool ok=false;
                    if(radius<1.&&omega>.1&&omega<=5.)ok=circle_call(f.circle,target,*pool,q)!=0;
                    if(!ok)ok=inside_call(f.inside,target,cell.data(),count,q)!=0;
                    if(!ok)continue;const Point candidate{q[0],q[1]};const double value=std::hypot(q[0],q[1]);
                    if(value>cap||value>=r.value||pool->cached(candidate,err)>=0)continue;
                    double t=NANV;const int status=query(f.proposal,candidate,t);
                    if(status==1)update(candidate);
                    else if(status==0&&t>0.)pool->store(s,wa,wb,t,guard,f.planes,candidate);
                    if(std::isfinite(r.value)&&r.value-std::max(0.,radius-err)<=tolerance)break;
                }
                if(std::isfinite(r.value)&&r.value-std::max(0.,radius-err)<=tolerance)continue;
                known=pool->cached(point,err);
            }
            double n[16],b[8],cut_time=NANV;
            if(known<0){
                double t=NANV;const int status=query(f.main,point,t);
                if(status==1){update(point);continue;}
                if(status!=0||!(t>0.))need_rescue=true;
                else{
                    known=pool->store(s,wa,wb,t,guard,f.planes,point);
                    cut_time=t;
                    if(known>=0){std::copy_n(pool->n.data()+known*16,16,n);std::copy_n(pool->b.data()+known*8,8,b);}
                    else {double raw[24];f.planes(s,wa,wb,t,guard,raw);std::copy_n(raw,16,n);std::copy_n(raw+16,8,b);}
                }
            }else{cut_time=pool->times[known];std::copy_n(pool->n.data()+known*16,16,n);std::copy_n(pool->b.data()+known*8,8,b);}
            if(!need_rescue)for(int j=0;j<8;++j)if(n[2*j]*point.x+n[2*j+1]*point.y>=b[j]-err){need_rescue=true;break;}
            if(need_rescue){
                bool restart=false;
                for(double fraction:{.9,.5,1.8}){
                    const double target=radius+fraction*tolerance;if(target>cap)continue;double q[2];
                    if(!circle_call(f.circle,target,*pool,q))continue;
                    const Point candidate{q[0],q[1]};double t=NANV;const int status=query(f.main,candidate,t);
                    if(status==1){update(candidate);if(r.value-r.lower<=tolerance){r.iterations=iteration+1;finish(1,true);return;}}
                    else if(status==0&&t>0.){
                        const int old=pool->count;
                        pool->store(s,wa,wb,t,guard,f.planes,candidate);known=pool->cached(point,err);
                        if(known>=0){cut_time=pool->times[known];std::copy_n(pool->n.data()+known*16,16,n);std::copy_n(pool->b.data()+known*8,8,b);need_rescue=false;break;}
                        if(pool->count>old){restart=true;break;}
                    }
                }
                if(restart){if(free.empty()){r.iterations=iteration+1;finish(5,false);return;}
                    const int id=free.back();free.pop_back();storage[id].count=count;storage[id].lower=r.lower;storage[id].hint_time=cell_hint;
                    std::copy_n(cell.data(),count,storage[id].v.data());pending.push({r.lower,id,serial++});continue;}
                if(need_rescue){r.iterations=iteration+1;finish(5,false);return;}
            }
            ++r.cuts;int remaining=count;
            for(int j=0;j<8;++j){
                if(remaining<3)break;const double nx=n[2*j],ny=n[2*j+1],offset=b[j];
                const int size=clip(cell.data(),remaining,-nx,-ny,-offset,piece.data());
                if(size>=3){Point candidate;const double norm=nearest(piece.data(),size,candidate);
                    const double bound=std::max(r.lower,std::max(0.,norm-err));
                    if(bound<=cap&&(!std::isfinite(r.value)||r.value-bound>tolerance)){
                        if(free.empty()||size>max_vertices){r.iterations=iteration+1;finish(5,false);return;}
                        const int id=free.back();free.pop_back();storage[id].count=size;storage[id].lower=bound;storage[id].hint_time=cut_time;
                        std::copy_n(piece.data(),size,storage[id].v.data());pending.push({bound,id,serial++});++r.nodes;
                    }
                }
                const int size_inside=clip(cell.data(),remaining,nx,ny,offset,work.data());
                if(size_inside>max_vertices){r.iterations=iteration+1;finish(5,false);return;}
                remaining=size_inside;std::copy_n(work.data(),remaining,cell.data());
            }
        }
        r.iterations=max_iterations;finish(5,false);
    }catch(...){r.status=5;r.converged=false;r.output(out);}
}

extern "C" void ea_cells_native_solve(double*s,double wa,double wb,double horizon,double cap,double tolerance,
    double guard,int max_nodes,int max_vertices,int max_iterations,int cert_limit,
    void*initial,void*proposal,void*main,void*planes,void*refine,void*sep,void*circle,void*inside,double*out){
    solve_native(s,wa,wb,horizon,cap,tolerance,guard,max_nodes,max_vertices,max_iterations,cert_limit,
        initial,proposal,main,planes,refine,sep,circle,inside,nullptr,out);
}
extern "C" void ea_cells_native_solve_seeded(double*s,double wa,double wb,double horizon,double cap,double tolerance,
    double guard,int max_nodes,int max_vertices,int max_iterations,int cert_limit,
    void*initial,void*proposal,void*main,void*planes,void*refine,void*sep,void*circle,void*inside,void*seed,double*out){
    solve_native(s,wa,wb,horizon,cap,tolerance,guard,max_nodes,max_vertices,max_iterations,cert_limit,
        initial,proposal,main,planes,refine,sep,circle,inside,reinterpret_cast<Seeds*>(seed),out);
}
