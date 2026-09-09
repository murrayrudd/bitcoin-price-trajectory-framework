#include <cmath>
#include <algorithm>
#include <limits>

// Methods v0.5. All prices, endowments and periods are synthetic.
// Kinds: 0=O, 1=T, 2=hybrid (O for A/B), 3=T with eta=3 from month 43.
// Parameters per row: funding, probability shift, anchor exponent, eta, tau.
// State/output: log price, holdings A/B/C, cash A/B/C, lagged log return.
static const double a0[3]={.75,.375,.25};
static const double xi[3]={.25,.6,1.};
static const double sign_[3]={1.,0.,-1.};

static double orders(int kind,const double* par,int t,double A,double P,
                     const double* h,const double* c,const double* k,double lag,double* q){
    double z=0.;
    for(int i=0;i<3;i++){
        double a;
        bool opt=(kind==0 || (kind==2 && i<2));
        if(opt){
            double pi=(1.+a0[i])/3.+(i==0?par[1]:0.);
            double L=.5*A,H=2.*A,E=(1.-pi)*L+pi*H;
            if(P<=L) a=1.;
            else if(P>=H) a=0.;
            else a=std::clamp(P*(E-P)/((P-L)*(H-P)),0.,1.);
        }else{
            double eta=(kind==3 && t>=43)?3.:par[3];
            double v=std::log(a0[i]/(1.-a0[i]))+sign_[i]*par[4]*lag+eta*std::log(A/P);
            a=(v>=0.)?1./(1.+std::exp(-v)):std::exp(v)/(1.+std::exp(v));
        }
        double raw=(opt?1.:xi[i])*(a*(h[i]+c[i]/P)-h[i]);
        q[i]=std::clamp(raw,k[i]-h[i],c[i]/P); z+=q[i];
    }
    return z;
}

extern "C" void simulate(int kind,int n,const double* pars,int start,int end,
                          const double* initial,double chi,double pulse,
                          double* out,double* volumes,double* metrics){
    int length=end-start+1;
    for(int j=0;j<n;j++){
        const double* par=pars+j*5;
        double h[3],c[3],k[3]={0,0,0},q[3],lp=initial[j*8],lag=initial[j*8+7];
        double* rec=out+j*length*8; double* met=metrics+j*6;
        met[0]=met[1]=met[2]=0.; met[3]=met[4]=1e300;met[5]=0.;
        for(int a=0;a<8;a++)rec[a]=initial[j*8+a];
        for(int i=0;i<3;i++){h[i]=initial[j*8+1+i];c[i]=initial[j*8+4+i];}
        volumes[j*length]=0.;
        for(int t=start+1;t<=end;t++){
            h[2]+=.1;c[0]+=par[0];c[1]+=.08;c[2]+=.02;
            if(t==start+1){for(int i=0;i<3;i++)k[i]=chi*h[i];c[0]+=pulse;}
            double A=std::pow((120.+t)/120.,par[2]);
            double lo=A/16.,hi=16.*A;
            double zl=orders(kind,par,t,A,lo,h,c,k,lag,q), zh=orders(kind,par,t,A,hi,h,c,k,lag,q);
            int expansion=0;
            while(!(zl>0. && zh<0.) && expansion<20){
                lo/=2.;hi*=2.;expansion++;
                zl=orders(kind,par,t,A,lo,h,c,k,lag,q);zh=orders(kind,par,t,A,hi,h,c,k,lag,q);
            }
            if(!(zl>0. && zh<0.)){
                met[5]=1.;
                for(int tt=t-start;tt<length;tt++){for(int a=0;a<8;a++)rec[tt*8+a]=NAN;volumes[j*length+tt]=NAN;}
                break;
            }
            double P=0,z=0,denom=0;
            for(int b=0;b<65;b++){
                P=(lo+hi)/2.;z=orders(kind,par,t,A,P,h,c,k,lag,q);
                denom=1.;for(int i=0;i<3;i++)denom+=h[i]+c[i]/P;
                if(std::abs(z)/denom<2e-15)break;
                if(z>0)lo=P;else hi=P;
            }
            met[0]=std::max(met[0],std::abs(z)/denom);
            met[1]=std::max(met[1],std::abs(z));
            double volume=0.;
            for(int i=0;i<3;i++){
                double wealth=c[i]+P*h[i];h[i]+=q[i];c[i]-=P*q[i];
                met[2]=std::max(met[2],std::abs(c[i]+P*h[i]-wealth));
                met[3]=std::min(met[3],h[i]-k[i]);met[4]=std::min(met[4],c[i]);
                volume+=.5*std::abs(q[i]);
            }
            double ln=std::log(P);lag=ln-lp;lp=ln;
            int offset=(t-start)*8;rec[offset]=lp;rec[offset+7]=lag;
            for(int i=0;i<3;i++){rec[offset+1+i]=h[i];rec[offset+4+i]=c[i];}
            volumes[j*length+t-start]=volume;
        }
    }
}
