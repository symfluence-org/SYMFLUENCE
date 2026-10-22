#include "pihm.h"
int nelem=0, nriver=0;
int main(void) {
    elem_struct a={0}, b={0};
    a.topo.edge[0]=b.topo.edge[0]=50.;
    a.topo.dist_nabr[0]=b.topo.dist_nabr[0]=100.;
    a.topo.zmax=100.;b.topo.zmax=99.5;
    a.ws.surfh=b.ws.surfh=.03;
    a.lc.rough=b.lc.rough=.03;
    double ab=OvlFlowElemToElem(0,.01,&a,&b);
    double ba=OvlFlowElemToElem(0,.01,&b,&a);
    printf("sloped %.17g %.17g %.17g\n",ab,ba,ab+ba);
    if (!(ab>0. && ba<0. && fabs(ab+ba)<1e-12)) return 1;
    b.topo.zmax=a.topo.zmax;
    ab=OvlFlowElemToElem(0,.01,&a,&b);ba=OvlFlowElemToElem(0,.01,&b,&a);
    printf("level %.17g %.17g %.17g\n",ab,ba,ab+ba);
    if(ab!=0. || ba!=0.)return 2;
    a.ws.surfh=b.ws.surfh=0.;
    ab=OvlFlowElemToElem(0,.01,&a,&b);ba=OvlFlowElemToElem(0,.01,&b,&a);
    if(ab!=0. || ba!=0.)return 3;
    return 0;
}
