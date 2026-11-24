// Generic SSE with heat-bath directed loops for number-conserving bond Hamiltonians
// H = sum_b H_b,  H_b = sum_{sites in b} eps_i(n_i)/deg_i + Vint(n_i,n_j) - [A-hopping]
// off-diagonal weight (a,b)->(a+1,b-1) = A[a][b] >= 0 (sign-free)
// Measures, for site 0: p_k = <Pi_k>, X_kl = Kubo integral of Pi_k Pi_l (not connected).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <string>
using namespace std;

int N, Nb, d; double beta_; unsigned long long seed;
int nbins, spb, therm; double cpad;
vector<int> bs1, bs2, deg; vector<double> gb;
vector<double> eps;   // N*d
vector<double> Vint, Aw; // d*d
vector<double> Cb;    // per bond constant
vector<double> Aobs, taus; int ntau=0, Ksamp=64;
mt19937_64 rng;
uniform_real_distribution<double> U01(0.0,1.0);
inline double rnd(){ return U01(rng); }

int M, nops;
vector<int> opb;      // bond index of operator at position p, -1 identity
vector<int> legs;     // 4*M leg states: 0 lower i, 1 lower j, 2 upper i, 3 upper j
vector<int> st;       // state at tau=0
vector<int> vtx;      // list of positions of ops (compressed), size nops
vector<int> link_;
vector<int> firstv, lastv;

inline double Hd(int b, int a, int c){
  int i=bs1[b], j=bs2[b];
  return eps[i*d+a]/deg[i] + eps[j*d+c]/deg[j] + gb[b]*Vint[a*d+c];
}
inline double W(int b, int a, int bb, int c, int dd){
  if(a<0||a>=d||bb<0||bb>=d||c<0||c>=d||dd<0||dd>=d) return 0.0;
  if(a==c && bb==dd) return Cb[b]-Hd(b,a,bb);
  if(c==a+1 && dd==bb-1) return gb[b]*Aw[a*d+bb];
  if(c==a-1 && dd==bb+1) return gb[b]*Aw[c*d+dd];
  return 0.0;
}

void diag_update(){
  vector<int> s = st;
  double fac = beta_*Nb;
  for(int p=0;p<M;p++){
    int b = opb[p];
    if(b<0){
      int bn = (int)(rnd()*Nb); if(bn>=Nb) bn=Nb-1;
      int a=s[bs1[bn]], c=s[bs2[bn]];
      double w = Cb[bn]-Hd(bn,a,c);
      if(w>0 && rnd()*(M-nops) < fac*w){
        opb[p]=bn; legs[4*p]=a; legs[4*p+1]=c; legs[4*p+2]=a; legs[4*p+3]=c; nops++;
      }
    } else {
      int i=bs1[b], j=bs2[b];
      bool isdiag = (legs[4*p]==legs[4*p+2] && legs[4*p+1]==legs[4*p+3]);
      if(isdiag){
        double w = Cb[b]-Hd(b,s[i],s[j]);
        if(rnd()*fac*w < (M-nops+1)){ opb[p]=-1; nops--; continue; }
      }
      s[i]=legs[4*p+2]; s[j]=legs[4*p+3];
    }
  }
}

void build_links(){
  vtx.clear();
  for(int p=0;p<M;p++) if(opb[p]>=0) vtx.push_back(p);
  int n = vtx.size();
  link_.assign(4*n,-1);
  firstv.assign(N,-1); lastv.assign(N,-1);
  for(int v=0; v<n; v++){
    int b=opb[vtx[v]]; int si[2]={bs1[b],bs2[b]};
    for(int k=0;k<2;k++){
      int i=si[k]; int lo=4*v+k, up=4*v+2+k;
      if(lastv[i]>=0){ link_[lastv[i]]=lo; link_[lo]=lastv[i]; }
      else firstv[i]=lo;
      lastv[i]=up;
    }
  }
  for(int i=0;i<N;i++) if(firstv[i]>=0){ link_[firstv[i]]=lastv[i]; link_[lastv[i]]=firstv[i]; }
}

double Nl=5; double looplen_acc=0; long nloops_acc=0;

void loop_update(){
  int n = vtx.size();
  if(n==0){ for(int i=0;i<N;i++) st[i]=(int)(rnd()*d)%d; return; }
  int nl = max(1,(int)Nl);
  for(int l=0;l<nl;l++){
    int j0 = (int)(rnd()*4*n); if(j0>=4*n) j0=4*n-1;
    int k0 = link_[j0];
    int Delta = (rnd()<0.5)?1:-1;
    // apply change to leg j0
    int e = j0; int De = Delta;
    long len=0;
    legs[4*vtx[j0/4]+(j0%4)] += Delta;
    while(true){
      int v = e/4, le = e%4; int p = vtx[v]; int b=opb[p];
      int *s = &legs[4*p];
      // candidates
      double w[4]; int dl[4]; double tot=0;
      for(int y=0;y<4;y++){
        int dy;
        if(y==le) dy=-De;
        else dy = ((y<2)==(le<2)) ? -De : De;
        dl[y]=dy;
        s[y]+=dy;
        w[y]=W(b,s[0],s[1],s[2],s[3]);
        s[y]-=dy;
        tot+=w[y];
      }
      double r=rnd()*tot; int y=0;
      for(y=0;y<3;y++){ if(r<w[y]) break; r-=w[y]; }
      while(w[y]==0.0 && y>0) y--; // safety
      s[y]+=dl[y];
      len++;
      int x = 4*v+y;
      if(x==j0 || x==k0) break;
      int z = link_[x];
      legs[4*vtx[z/4]+(z%4)] += dl[y];
      e = z; De = dl[y];
      if(len>100000000L){ fprintf(stderr,"loop too long\n"); exit(1);} 
    }
    looplen_acc += len; nloops_acc++;
  }
  // new initial state
  for(int i=0;i<N;i++){
    if(firstv[i]>=0){ int f=firstv[i]; st[i]=legs[4*vtx[f/4]+(f%4)]; }
    else st[i]=(int)(rnd()*d)%d;
  }
}

int main(int argc, char** argv){
  FILE* f=fopen(argv[1],"r");
  if(!f){fprintf(stderr,"no input\n");return 1;}
  if(fscanf(f,"%d %d %d %lf %llu %d %d %d %lf",&N,&Nb,&d,&beta_,&seed,&nbins,&spb,&therm,&cpad)!=9) return 2;
  bs1.resize(Nb); bs2.resize(Nb); gb.resize(Nb); deg.assign(N,0);
  for(int b=0;b<Nb;b++){ if(fscanf(f,"%d %d %lf",&bs1[b],&bs2[b],&gb[b])!=3) return 3; deg[bs1[b]]++; deg[bs2[b]]++; }
  eps.resize(N*d); for(auto&x:eps) if(fscanf(f,"%lf",&x)!=1) return 4;
  Vint.resize(d*d); for(auto&x:Vint) if(fscanf(f,"%lf",&x)!=1) return 5;
  Aw.resize(d*d); for(auto&x:Aw) if(fscanf(f,"%lf",&x)!=1) return 6;
  Aobs.resize(d); for(auto&x:Aobs) if(fscanf(f,"%lf",&x)!=1) return 7;
  if(fscanf(f,"%d",&ntau)!=1) return 8; taus.resize(ntau); for(auto&x:taus) if(fscanf(f,"%lf",&x)!=1) return 9;
  if(fscanf(f,"%d",&Ksamp)!=1) return 10;
  fclose(f);
  rng.seed(seed);
  Cb.resize(Nb); double Csum=0;
  for(int b=0;b<Nb;b++){ double mx=-1e300; for(int a=0;a<d;a++)for(int c=0;c<d;c++) mx=max(mx,Hd(b,a,c)); Cb[b]=mx+cpad; Csum+=Cb[b]; }
  M = max(20, (int)(beta_*Nb*0.1)); nops=0; opb.assign(M,-1); legs.assign(4*M,0);
  st.resize(N); for(int i=0;i<N;i++) st[i]=(int)(rnd()*d)%d;
  // thermalization with M and Nl adjustment
  for(int s=0;s<therm;s++){
    diag_update(); build_links(); loop_update();
    if(nops*4/3 > M){ int Mn = nops*4/3+10; opb.resize(Mn,-1); legs.resize(4*Mn,0); M=Mn; }
    if(s%100==99 && nloops_acc>0){
      double avlen = looplen_acc/nloops_acc; Nl = max(1.0, 2.0*max(1,nops)/avlen);
      looplen_acc=0; nloops_acc=0;
    }
  }
  printf("# Csum %.12g M %d Nl %g\n", Csum, M, Nl);
  vector<double> cnt(d);
  for(int bin=0;bin<nbins;bin++){
    double an=0; vector<double> P(d,0), X(d*d,0), G(ntau,0); vector<double> aseq;
    for(int s=0;s<spb;s++){
      diag_update(); build_links(); loop_update();
      // measure site 0 time series
      int s0 = st[0]; int n=nops;
      for(int k=0;k<d;k++) cnt[k]=0;
      if(n==0){
        P[s0]+=1; X[s0*d+s0]+=beta_;
        for(int k=0;k<ntau;k++) G[k]+=Aobs[s0]*Aobs[s0];
      } else {
        aseq.resize(n); int q=0;
        for(int p=0;p<M;p++){ int b=opb[p]; if(b<0) continue;
          if(bs1[b]==0) s0=legs[4*p+2]; else if(bs2[b]==0) s0=legs[4*p+3];
          cnt[s0]+=1; aseq[q++]=Aobs[s0]; }
        // imaginary-time correlation <A(tau)A(0)>: m ~ Binomial(n, tau/beta), p uniform (cyclic)
        for(int k=0;k<ntau;k++){
          binomial_distribution<int> bd(n, taus[k]); double acc=0;
          for(int r=0;r<Ksamp;r++){ int p=(int)(rnd()*n); if(p>=n)p=n-1; int m=bd(rng); acc+=aseq[p]*aseq[(p+m)%n]; }
          G[k]+=acc/Ksamp;
        }
        double nn=n, fac=beta_/(nn*(nn+1));
        for(int k=0;k<d;k++){ P[k]+=cnt[k]/nn; for(int l=0;l<d;l++) X[k*d+l]+=fac*(cnt[k]*cnt[l]+(k==l?cnt[k]:0)); }
      }
      an+=n;
    }
    printf("%.10g",an/spb);
    for(int k=0;k<d;k++) printf(" %.10g",P[k]/spb);
    for(int k=0;k<d*d;k++) printf(" %.10g",X[k]/spb);
    for(int k=0;k<ntau;k++) printf(" %.10g",G[k]/spb);
    printf("\n"); fflush(stdout);
  }
  return 0;
}
