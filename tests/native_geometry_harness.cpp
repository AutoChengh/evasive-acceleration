// Independent scalar check of the SIMD face-hint filter. Include in one
// translation unit to exercise the actual private Pool implementation.
#include "native_cells.cpp" // add the EA solver source directory with -I
#include <iostream>
#include <random>

int main(){
    std::mt19937_64 rng(20261022);
    std::uniform_real_distribution<double> u(-100.,100.),angle(-3.14,3.14),width(.001,20.);
    auto p=std::unique_ptr<Pool>(new Pool);p->count=2048;
    for(int k=0;k<p->count;++k){
        const double x=u(rng),y=u(rng),theta=angle(rng),c=std::cos(theta),s=std::sin(theta),a=width(rng),b=width(rng);
        const double nx[8]={c,-c,-s,s,c,-c,-s,s},ny[8]={s,-s,c,-c,s,-s,c,-c};
        for(int j=0;j<8;++j){p->n[k*16+j*2]=nx[j];p->n[k*16+j*2+1]=ny[j];
            p->b[k*8+j]=nx[j]*x+ny[j]*y+((j%4<2)?a:b);}
        p->faces[k]=0;p->hint_x[k]=nx[0];p->hint_y[k]=ny[0];p->hint_b[k]=p->b[k*8];
    }
    const double err=2048.*std::numeric_limits<double>::epsilon()*101.;
    int checks=0;
    for(int iteration=0;iteration<10000;++iteration){
        const Point x{u(rng),u(rng)};int expected=-1;
        for(int k=p->count-1;k>=0;--k){
            double gap=-INF;
            for(int j=0;j<8;++j)gap=std::max(gap,p->n[k*16+j*2]*x.x+p->n[k*16+j*2+1]*x.y-p->b[k*8+j]);
            if(gap<-err){expected=k;break;}
        }
        const int actual=p->cached(x,err);++checks;
        if(actual!=expected){std::cerr<<"cached predicate mismatch "<<iteration<<" "<<actual<<" "<<expected<<"\n";return 1;}
        // Vary both full blocks and partial tails; no answer is reused.
        p->count=iteration%2048+1;
    }
    for(int iteration=0;iteration<10000;++iteration){
        const Point x{u(rng),u(rng)};Mask excluded;
        for(auto&word:excluded)word=rng();int expected=-1;
        for(int k=p->count-1;k>=0;--k){
            if(excluded[k/64]&(uint64_t(1)<<(k%64)))continue;
            double gap=-INF;
            for(int j=0;j<8;++j)gap=std::max(gap,p->n[k*16+j*2]*x.x+p->n[k*16+j*2+1]*x.y-p->b[k*8+j]);
            if(gap<-err){expected=k;break;}
        }
        const int actual=p->cached(x,err,&excluded);++checks;
        if(actual!=expected){std::cerr<<"masked SIMD mismatch "<<iteration<<" "<<actual<<" "<<expected<<"\n";return 1;}
        p->count=iteration%2048+1;
    }
    std::cout<<"scalar/SIMD exact predicate checks "<<checks<<" passed; AVX512="
#if defined(__AVX512F__)
        <<1
#else
        <<0
#endif
        <<"\n";
}
