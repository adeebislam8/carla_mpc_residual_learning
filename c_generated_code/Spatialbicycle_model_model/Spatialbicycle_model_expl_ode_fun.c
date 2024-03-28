/* This file was automatically generated by CasADi 3.6.5.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Spatialbicycle_model_expl_ode_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_clear_casadi_int CASADI_PREFIX(clear_casadi_int)
#define casadi_de_boor CASADI_PREFIX(de_boor)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_fill_casadi_int CASADI_PREFIX(fill_casadi_int)
#define casadi_low CASADI_PREFIX(low)
#define casadi_nd_boor_eval CASADI_PREFIX(nd_boor_eval)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_de_boor(casadi_real x, const casadi_real* knots, casadi_int n_knots, casadi_int degree, casadi_real* boor) {
  casadi_int d, i;
  for (d=1;d<degree+1;++d) {
    for (i=0;i<n_knots-d-1;++i) {
      casadi_real b, bottom;
      b = 0;
      bottom = knots[i + d] - knots[i];
      if (bottom) b = (x - knots[i]) * boor[i] / bottom;
      bottom = knots[i + d + 1] - knots[i + 1];
      if (bottom) b += (knots[i + d + 1] - x) * boor[i + 1] / bottom;
      boor[i] = b;
    }
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_fill_casadi_int(casadi_int* x, casadi_int n, casadi_int alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_clear_casadi_int(casadi_int* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

casadi_int casadi_low(casadi_real x, const casadi_real* grid, casadi_int ng, casadi_int lookup_mode) {
  switch (lookup_mode) {
    case 1:
      {
        casadi_real g0, dg;
        casadi_int ret;
        g0 = grid[0];
        dg = grid[ng-1]-g0;
        ret = (casadi_int) ((x-g0)*(ng-1)/dg);
        if (ret<0) ret=0;
        if (ret>ng-2) ret=ng-2;
        return ret;
      }
    case 2:
      {
        casadi_int start, stop, pivot;
        if (ng<2 || x<grid[1]) return 0;
        if (x>grid[ng-1]) return ng-2;
        start = 0;
        stop  = ng-1;
        while (1) {
          pivot = (stop+start)/2;
          if (x < grid[pivot]) {
            if (pivot==stop) return pivot;
            stop = pivot;
          } else {
            if (pivot==start) return pivot;
            start = pivot;
          }
        }
      }
    default:
      {
        casadi_int i;
        for (i=0; i<ng-2; ++i) {
          if (x < grid[i+1]) break;
        }
        return i;
      }
  }
}

void casadi_nd_boor_eval(casadi_real* ret, casadi_int n_dims, const casadi_real* all_knots, const casadi_int* offset, const casadi_int* all_degree, const casadi_int* strides, const casadi_real* c, casadi_int m, const casadi_real* all_x, const casadi_int* lookup_mode, casadi_int* iw, casadi_real* w) {
  casadi_int n_iter, k, i, pivot;
  casadi_int *boor_offset, *starts, *index, *coeff_offset;
  casadi_real *cumprod, *all_boor;
  boor_offset = iw; iw+=n_dims+1;
  starts = iw; iw+=n_dims;
  index = iw; iw+=n_dims;
  coeff_offset = iw;
  cumprod = w; w+= n_dims+1;
  all_boor = w;
  boor_offset[0] = 0;
  cumprod[n_dims] = 1;
  coeff_offset[n_dims] = 0;
  n_iter = 1;
  for (k=0;k<n_dims;++k) {
    casadi_real *boor;
    const casadi_real* knots;
    casadi_real x;
    casadi_int degree, n_knots, n_b, L, start;
    boor = all_boor+boor_offset[k];
    degree = all_degree[k];
    knots = all_knots + offset[k];
    n_knots = offset[k+1]-offset[k];
    n_b = n_knots-degree-1;
    x = all_x[k];
    L = casadi_low(x, knots+degree, n_knots-2*degree, lookup_mode[k]);
    start = L;
    if (start>n_b-degree-1) start = n_b-degree-1;
    starts[k] = start;
    casadi_clear(boor, 2*degree+1);
    if (x>=knots[0] && x<=knots[n_knots-1]) {
      if (x==knots[1]) {
        casadi_fill(boor, degree+1, 1.0);
      } else if (x==knots[n_knots-1]) {
        boor[degree] = 1;
      } else if (knots[L+degree]==x) {
        boor[degree-1] = 1;
      } else {
        boor[degree] = 1;
      }
    }
    casadi_de_boor(x, knots+start, 2*degree+2, degree, boor);
    boor+= degree+1;
    n_iter*= degree+1;
    boor_offset[k+1] = boor_offset[k] + degree+1;
  }
  casadi_clear_casadi_int(index, n_dims);
  for (pivot=n_dims-1;pivot>=0;--pivot) {
    cumprod[pivot] = (*(all_boor+boor_offset[pivot]))*cumprod[pivot+1];
    coeff_offset[pivot] = starts[pivot]*strides[pivot]+coeff_offset[pivot+1];
  }
  for (k=0;k<n_iter;++k) {
    casadi_int pivot = 0;
    for (i=0;i<m;++i) ret[i] += c[coeff_offset[0]+i]*cumprod[0];
    index[0]++;
    {
      while (index[pivot]==boor_offset[pivot+1]-boor_offset[pivot]) {
        index[pivot] = 0;
        if (pivot==n_dims-1) break;
        index[++pivot]++;
      }
      while (pivot>0) {
        cumprod[pivot] = (*(all_boor+boor_offset[pivot]+index[pivot]))*cumprod[pivot+1];
        coeff_offset[pivot] = (starts[pivot]+index[pivot])*strides[pivot]+coeff_offset[pivot+1];
        pivot--;
      }
    }
    cumprod[0] = (*(all_boor+index[0]))*cumprod[1];
    coeff_offset[0] = (starts[0]+index[0])*m+coeff_offset[1];
  }
}

static const casadi_int casadi_s0[2] = {0, 124};
static const casadi_int casadi_s1[1] = {3};
static const casadi_int casadi_s2[1] = {1};
static const casadi_int casadi_s3[1] = {2};
static const casadi_int casadi_s4[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s5[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s6[3] = {0, 0, 0};

static const casadi_real casadi_c0[124] = {-3.1307295000000002e+00, -3.1307295000000002e+00, -3.1307295000000002e+00, -3.1307295000000002e+00, -2.8561955999999999e+00, -2.5816618000000000e+00, -2.4589926000000002e+00, -2.3363233999999999e+00, -2.2136543000000000e+00, -2.0909851000000002e+00, -1.9683158999999999e+00, -1.8456467000000001e+00, -1.5711128999999999e+00, -1.2586128999999999e+00, -9.4611290000000003e-01, -7.4751129999999999e-01, -6.2484220000000001e-01, -5.0217299999999998e-01, -2.8571429999999998e-01, 6.2500000000000000e-02, 3.7500000000000000e-01, 6.8750000000000000e-01, 1., 1.1226692000000000e+00, 1.2453384000000001e+00, 1.3680076000000001e+00, 1.4906767000000001e+00, 1.6133459000000001e+00, 1.7360150999999999e+00, 1.8586843000000000e+00, 1.9813535000000000e+00, 2.1040226999999998e+00, 2.2266919000000001e+00, 2.3493610000000000e+00, 2.4720301999999998e+00, 2.5946994000000001e+00, 2.7173685999999999e+00, 2.8400378000000002e+00, 2.9627070000000000e+00, 3.0853761999999998e+00, 3.2080453000000002e+00, 3.3307145000000000e+00, 3.6052483999999998e+00, 3.8797822000000002e+00, 4.0024514000000000e+00, 4.1251205999999998e+00, 4.2477897999999996e+00, 4.5602897999999996e+00, 4.7968574000000004e+00, 4.9195266000000002e+00, 5.0421958000000000e+00, 5.1648649999999998e+00, 5.2875341999999996e+00, 5.4102034000000003e+00, 5.5328726000000001e+00, 5.8453726000000001e+00, 6.0819402000000000e+00, 6.2046093999999998e+00, 6.3272785999999996e+00, 6.4499478000000003e+00, 6.5726170000000002e+00, 6.6952862000000000e+00, 6.8179553000000004e+00, 7.1304553000000004e+00, 7.4429553000000004e+00, 7.7554553000000004e+00, 7.9160906999999998e+00, 8.0387599000000005e+00, 8.1614290999999994e+00, 8.4247823999999998e+00, 8.7729967000000002e+00, 9.0854967000000002e+00, 9.3979967000000002e+00, 9.7104967000000002e+00, 9.8331658999999991e+00, 9.9558350999999998e+00, 1.0078504300000001e+01, 1.0201173400000000e+01, 1.0323842600000001e+01, 1.0446511800000000e+01, 1.0569181000000000e+01, 1.0691850199999999e+01, 1.0814519400000000e+01, 1.0937188600000001e+01, 1.1059857700000000e+01, 1.1182526899999999e+01, 1.1305196100000000e+01, 1.1427865300000001e+01, 1.1550534499999999e+01, 1.1673203700000000e+01, 1.1795872900000001e+01, 1.1918542000000000e+01, 1.2041211199999999e+01, 1.2315745099999999e+01, 1.2590278899999999e+01, 1.2712948100000000e+01, 1.2835617299999999e+01, 1.2958286500000000e+01, 1.3270786500000000e+01, 1.3507354100000001e+01, 1.3630023300000000e+01, 1.3752692500000000e+01, 1.3875361699999999e+01, 1.3998030900000000e+01, 1.4120700100000001e+01, 1.4243369299999999e+01, 1.4555869299999999e+01, 1.4792436900000000e+01, 1.4915106099999999e+01, 1.5037775300000000e+01, 1.5160444500000001e+01, 1.5283113699999999e+01, 1.5405782900000000e+01, 1.5528452000000000e+01, 1.5840952000000000e+01, 1.6153452000000001e+01, 1.6465952000000001e+01, 1.6626587399999998e+01, 1.6749256599999999e+01, 1.6871925800000000e+01, 1.7135279100000002e+01, 1.7135279100000002e+01, 1.7135279100000002e+01, 1.7135279100000002e+01};
static const casadi_real casadi_c1[120] = {0., 0., 1.7317874799999999e+00, -4.6242881200000001e+00, -3.8959919699999999e+00, -4.0044931699999999e+00, -4.0860351799999997e+00, -3.6513657700000000e+00, -5.3085016400000002e+00, 3.4413798600000001e+00, -1.1998194099999999e+00, -4.1570498000000000e-01, 2.4934219500000001e+00, -4.8729867400000000e+00, -3.6848349200000001e+00, -4.8218146900000001e+00, 2.6391975900000002e+00, -6.2009112000000000e-01, -2.4410477000000000e-01, 1.5827125399999999e+00, -4.5337861100000003e+00, -3.9100898100000001e+00, -4.0380966899999997e+00, -3.9375234500000000e+00, -4.2118094399999997e+00, -3.2152380500000000e+00, -6.9272381000000003e+00, 6.9241904500000002e+00, 3.2304762999999999e+00, 4.1539043600000003e+00, 4.1539062500000004e+00, 3.2304712699999998e+00, 6.9242115599999998e+00, -6.9273165099999998e+00, -3.2149455100000002e+00, -4.2129014700000003e+00, -3.9334486300000000e+00, -4.0533039700000000e+00, -3.8533355000000000e+00, -4.8139762199999998e+00, 3.4018347499999999e+00, -5.0297580100000001e+00, -3.5112621000000002e+00, -5.3687752399999997e+00, 4.2645589499999996e+00, -3.2964052599999998e+00, 4.8204949199999998e+00, 3.8481155500000002e+00, 4.0168645700000001e+00, 4.0844261599999996e+00, 3.6454307799999999e+00, 5.3338507100000001e+00, -4.2381619199999996e+00, 3.2892281099999998e+00, -4.8198058200000000e+00, -3.8480407200000002e+00, -4.0177078100000001e+00, -4.0811280500000002e+00, -3.6577799899999999e+00, -5.2877515500000003e+00, 3.9535840400000000e+00, -1.1478139800000000e+00, -6.5755748000000003e-01, 3.0595256900000001e+00, -5.0273070999999998e+00, -3.6616933500000002e+00, -4.7563763400000001e+00, 2.2079379900000000e+00, -4.5952605000000002e-01, -2.8523047000000001e-01, 1.5940517700000001e+00, -4.5348645200000002e+00, -3.9099151100000000e+00, -4.0381435000000003e+00, -3.9375108999999999e+00, -4.2118127999999997e+00, -3.2152371500000001e+00, -6.9272383399999997e+00, 6.9241905199999998e+00, 3.2304762800000000e+00, 4.1539043700000002e+00, 4.1539062500000004e+00, 3.2304712699999998e+00, 6.9242115599999998e+00, -6.9273165099999998e+00, -3.2149455100000002e+00, -4.2129014700000003e+00, -3.9334486300000000e+00, -4.0533039700000000e+00, -3.8533355000000000e+00, -4.8139762199999998e+00, 3.4018347499999999e+00, -5.0297580100000001e+00, -3.5112621000000002e+00, -5.3687752399999997e+00, 4.2645589499999996e+00, -3.2964052599999998e+00, 4.8204949199999998e+00, 3.8481155500000002e+00, 4.0168645700000001e+00, 4.0844261599999996e+00, 3.6454307799999999e+00, 5.3338507100000001e+00, -4.2381619199999996e+00, 3.2892281300000001e+00, -4.8198058499999998e+00, -3.8480406600000001e+00, -4.0177080500000004e+00, -4.0811271299999996e+00, -3.6577834500000002e+00, -5.2877386499999997e+00, 3.9535043999999999e+00, -1.1469765900000000e+00, -6.6059456999999999e-01, 3.0685124400000001e+00, -5.0418522699999997e+00, -3.6207573599999998e+00, -4.9854330100000004e+00, 0., 0.};

/* kapparef_s:(i0)->(o0) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1;
  /* #0: @0 = input[0][0] */
  w0 = arg[0] ? arg[0][0] : 0;
  /* #1: @1 = BSpline(@0) */
  casadi_clear((&w1), 1);
  CASADI_PREFIX(nd_boor_eval)((&w1),1,casadi_c0,casadi_s0,casadi_s1,casadi_s2,casadi_c1,1,(&w0),casadi_s3, iw, w);
  /* #2: output[0][0] = @1 */
  if (res[0]) res[0][0] = w1;
  return 0;
}

/* Spatialbicycle_model_expl_ode_fun:(i0[6],i1[2],i2[])->(o0[6]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real **res1=res+1;
  const casadi_real **arg1=arg+3;
  casadi_real w0, w1, w2, w3, w4, w5, w6, w7;
  /* #0: @0 = input[0][3] */
  w0 = arg[0] ? arg[0][3] : 0;
  /* #1: @1 = input[0][2] */
  w1 = arg[0] ? arg[0][2] : 0;
  /* #2: @2 = 0.5 */
  w2 = 5.0000000000000000e-01;
  /* #3: @3 = input[0][5] */
  w3 = arg[0] ? arg[0][5] : 0;
  /* #4: @2 = (@2*@3) */
  w2 *= w3;
  /* #5: @2 = (@1+@2) */
  w2  = (w1+w2);
  /* #6: @2 = cos(@2) */
  w2 = cos( w2 );
  /* #7: @2 = (@0*@2) */
  w2  = (w0*w2);
  /* #8: @4 = 1 */
  w4 = 1.;
  /* #9: @5 = input[0][0] */
  w5 = arg[0] ? arg[0][0] : 0;
  /* #10: @6 = kapparef_s(@5) */
  arg1[0]=(&w5);
  res1[0]=(&w6);
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #11: @7 = input[0][1] */
  w7 = arg[0] ? arg[0][1] : 0;
  /* #12: @6 = (@6*@7) */
  w6 *= w7;
  /* #13: @4 = (@4-@6) */
  w4 -= w6;
  /* #14: @2 = (@2/@4) */
  w2 /= w4;
  /* #15: output[0][0] = @2 */
  if (res[0]) res[0][0] = w2;
  /* #16: @4 = 0.5 */
  w4 = 5.0000000000000000e-01;
  /* #17: @4 = (@4*@3) */
  w4 *= w3;
  /* #18: @1 = (@1+@4) */
  w1 += w4;
  /* #19: @1 = sin(@1) */
  w1 = sin( w1 );
  /* #20: @1 = (@0*@1) */
  w1  = (w0*w1);
  /* #21: output[0][1] = @1 */
  if (res[0]) res[0][1] = w1;
  /* #22: @1 = 15.5 */
  w1 = 1.5500000000000000e+01;
  /* #23: @1 = (@1*@0) */
  w1 *= w0;
  /* #24: @1 = (@1*@3) */
  w1 *= w3;
  /* #25: @4 = kapparef_s(@5) */
  arg1[0]=(&w5);
  res1[0]=(&w4);
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #26: @4 = (@4*@2) */
  w4 *= w2;
  /* #27: @1 = (@1-@4) */
  w1 -= w4;
  /* #28: output[0][2] = @1 */
  if (res[0]) res[0][2] = w1;
  /* #29: @1 = 0.28 */
  w1 = 2.8000000000000003e-01;
  /* #30: @4 = 0.05 */
  w4 = 5.0000000000000003e-02;
  /* #31: @4 = (@4*@0) */
  w4 *= w0;
  /* #32: @1 = (@1-@4) */
  w1 -= w4;
  /* #33: @4 = input[0][4] */
  w4 = arg[0] ? arg[0][4] : 0;
  /* #34: @1 = (@1*@4) */
  w1 *= w4;
  /* #35: @4 = 0.006 */
  w4 = 6.0000000000000001e-03;
  /* #36: @4 = (@4*@0) */
  w4 *= w0;
  /* #37: @4 = (@4*@0) */
  w4 *= w0;
  /* #38: @1 = (@1-@4) */
  w1 -= w4;
  /* #39: @4 = 0.011 */
  w4 = 1.0999999999999999e-02;
  /* #40: @2 = 5 */
  w2 = 5.;
  /* #41: @2 = (@2*@0) */
  w2 *= w0;
  /* #42: @2 = tanh(@2) */
  w2 = tanh( w2 );
  /* #43: @4 = (@4*@2) */
  w4 *= w2;
  /* #44: @1 = (@1-@4) */
  w1 -= w4;
  /* #45: @4 = 0.043 */
  w4 = 4.2999999999999997e-02;
  /* #46: @1 = (@1/@4) */
  w1 /= w4;
  /* #47: @4 = 0.5 */
  w4 = 5.0000000000000000e-01;
  /* #48: @4 = (@4*@3) */
  w4 *= w3;
  /* #49: @4 = cos(@4) */
  w4 = cos( w4 );
  /* #50: @1 = (@1*@4) */
  w1 *= w4;
  /* #51: output[0][3] = @1 */
  if (res[0]) res[0][3] = w1;
  /* #52: @1 = input[1][0] */
  w1 = arg[1] ? arg[1][0] : 0;
  /* #53: output[0][4] = @1 */
  if (res[0]) res[0][4] = w1;
  /* #54: @1 = input[1][1] */
  w1 = arg[1] ? arg[1][1] : 0;
  /* #55: output[0][5] = @1 */
  if (res[0]) res[0][5] = w1;
  return 0;
}

CASADI_SYMBOL_EXPORT int Spatialbicycle_model_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Spatialbicycle_model_expl_ode_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Spatialbicycle_model_expl_ode_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Spatialbicycle_model_expl_ode_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Spatialbicycle_model_expl_ode_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Spatialbicycle_model_expl_ode_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Spatialbicycle_model_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Spatialbicycle_model_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Spatialbicycle_model_expl_ode_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int Spatialbicycle_model_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Spatialbicycle_model_expl_ode_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Spatialbicycle_model_expl_ode_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Spatialbicycle_model_expl_ode_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Spatialbicycle_model_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Spatialbicycle_model_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Spatialbicycle_model_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 6;
  if (sz_w) *sz_w = 19;
  return 0;
}

CASADI_SYMBOL_EXPORT int Spatialbicycle_model_expl_ode_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 3*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 6*sizeof(casadi_int);
  if (sz_w) *sz_w = 19*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
