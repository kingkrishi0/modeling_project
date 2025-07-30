/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__ode_neuron
#define _nrn_initial _nrn_initial__ode_neuron
#define nrn_cur _nrn_cur__ode_neuron
#define _nrn_current _nrn_current__ode_neuron
#define nrn_jacob _nrn_jacob__ode_neuron
#define nrn_state _nrn_state__ode_neuron
#define _net_receive _net_receive__ode_neuron 
#define states states__ode_neuron 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg(int);
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define g_leak _p[0]
#define g_leak_columnindex 0
#define e_leak _p[1]
#define e_leak_columnindex 1
#define v_threshold_spike _p[2]
#define v_threshold_spike_columnindex 2
#define ksP _p[3]
#define ksP_columnindex 3
#define k_cleave _p[4]
#define k_cleave_columnindex 4
#define k_p75_pro_on _p[5]
#define k_p75_pro_on_columnindex 5
#define k_p75_pro_off _p[6]
#define k_p75_pro_off_columnindex 6
#define k_degP _p[7]
#define k_degP_columnindex 7
#define k_TrkB_pro_on _p[8]
#define k_TrkB_pro_on_columnindex 8
#define k_TrkB_pro_off _p[9]
#define k_TrkB_pro_off_columnindex 9
#define k_TrkB_B_on _p[10]
#define k_TrkB_B_on_columnindex 10
#define k_TrkB_B_off _p[11]
#define k_TrkB_B_off_columnindex 11
#define k_degB _p[12]
#define k_degB_columnindex 12
#define k_p75_B_on _p[13]
#define k_p75_B_on_columnindex 13
#define k_p75_B_off _p[14]
#define k_p75_B_off_columnindex 14
#define k_degR1 _p[15]
#define k_degR1_columnindex 15
#define k_degR2 _p[16]
#define k_degR2_columnindex 16
#define k_int_p75_pro _p[17]
#define k_int_p75_pro_columnindex 17
#define k_int_p75_B _p[18]
#define k_int_p75_B_columnindex 18
#define k_int_TrkB_B _p[19]
#define k_int_TrkB_B_columnindex 19
#define k_int_TrkB_pro _p[20]
#define k_int_TrkB_pro_columnindex 20
#define aff_p75_pro _p[21]
#define aff_p75_pro_columnindex 21
#define aff_p75_B _p[22]
#define aff_p75_B_columnindex 22
#define aff_TrkB_pro _p[23]
#define aff_TrkB_pro_columnindex 23
#define aff_TrkB_B _p[24]
#define aff_TrkB_B_columnindex 24
#define k_deg_tPA _p[25]
#define k_deg_tPA_columnindex 25
#define ks_tPA _p[26]
#define ks_tPA_columnindex 26
#define ks_p75 _p[27]
#define ks_p75_columnindex 27
#define ks_TrkB _p[28]
#define ks_TrkB_columnindex 28
#define tau_activity _p[29]
#define tau_activity_columnindex 29
#define activity_gain _p[30]
#define activity_gain_columnindex 30
#define growth_strength _p[31]
#define growth_strength_columnindex 31
#define apop_strength _p[32]
#define apop_strength_columnindex 32
#define syn_input_activity _p[33]
#define syn_input_activity_columnindex 33
#define P _p[34]
#define P_columnindex 34
#define B _p[35]
#define B_columnindex 35
#define p75 _p[36]
#define p75_columnindex 36
#define TrkB _p[37]
#define TrkB_columnindex 37
#define p75_pro _p[38]
#define p75_pro_columnindex 38
#define p75_B _p[39]
#define p75_B_columnindex 39
#define TrkB_B _p[40]
#define TrkB_B_columnindex 40
#define TrkB_pro _p[41]
#define TrkB_pro_columnindex 41
#define tPA _p[42]
#define tPA_columnindex 42
#define activity_level _p[43]
#define activity_level_columnindex 43
#define ks_P_variable _p[44]
#define ks_P_variable_columnindex 44
#define ks_tPA_variable _p[45]
#define ks_tPA_variable_columnindex 45
#define DP _p[46]
#define DP_columnindex 46
#define DB _p[47]
#define DB_columnindex 47
#define Dp75 _p[48]
#define Dp75_columnindex 48
#define DTrkB _p[49]
#define DTrkB_columnindex 49
#define Dp75_pro _p[50]
#define Dp75_pro_columnindex 50
#define Dp75_B _p[51]
#define Dp75_B_columnindex 51
#define DTrkB_B _p[52]
#define DTrkB_B_columnindex 52
#define DTrkB_pro _p[53]
#define DTrkB_pro_columnindex 53
#define DtPA _p[54]
#define DtPA_columnindex 54
#define Dactivity_level _p[55]
#define Dactivity_level_columnindex 55
#define v _p[56]
#define v_columnindex 56
#define _g _p[57]
#define _g_columnindex 57
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_Hill(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_ode_neuron", _hoc_setdata,
 "Hill_ode_neuron", _hoc_Hill,
 0, 0
};
#define Hill Hill_ode_neuron
 extern double Hill( _threadargsprotocomma_ double , double , double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "g_leak_ode_neuron", "S/cm2",
 "e_leak_ode_neuron", "mV",
 "v_threshold_spike_ode_neuron", "mV",
 "ksP_ode_neuron", "uM/s",
 "k_cleave_ode_neuron", "/s",
 "k_p75_pro_on_ode_neuron", "MS",
 "k_p75_pro_off_ode_neuron", "/s",
 "k_degP_ode_neuron", "/s",
 "k_TrkB_pro_on_ode_neuron", "MS",
 "k_TrkB_pro_off_ode_neuron", "/s",
 "k_TrkB_B_on_ode_neuron", "MS",
 "k_TrkB_B_off_ode_neuron", "/s",
 "k_degB_ode_neuron", "/s",
 "k_p75_B_on_ode_neuron", "MS",
 "k_p75_B_off_ode_neuron", "/s",
 "k_degR1_ode_neuron", "/s",
 "k_degR2_ode_neuron", "/s",
 "k_int_p75_pro_ode_neuron", "/s",
 "k_int_p75_B_ode_neuron", "/s",
 "k_int_TrkB_B_ode_neuron", "/s",
 "k_int_TrkB_pro_ode_neuron", "/s",
 "k_deg_tPA_ode_neuron", "/s",
 "ks_tPA_ode_neuron", "uM/s",
 "ks_p75_ode_neuron", "uM/s",
 "ks_TrkB_ode_neuron", "uM/s",
 "tau_activity_ode_neuron", "ms",
 "P_ode_neuron", "uM",
 "B_ode_neuron", "uM",
 "p75_ode_neuron", "uM",
 "TrkB_ode_neuron", "uM",
 "p75_pro_ode_neuron", "uM",
 "p75_B_ode_neuron", "uM",
 "TrkB_B_ode_neuron", "uM",
 "TrkB_pro_ode_neuron", "uM",
 "tPA_ode_neuron", "uM",
 0,0
};
 static double B0 = 0;
 static double P0 = 0;
 static double TrkB_pro0 = 0;
 static double TrkB_B0 = 0;
 static double TrkB0 = 0;
 static double activity_level0 = 0;
 static double delta_t = 0.01;
 static double p75_B0 = 0;
 static double p75_pro0 = 0;
 static double p750 = 0;
 static double tPA0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[0]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ode_neuron",
 "g_leak_ode_neuron",
 "e_leak_ode_neuron",
 "v_threshold_spike_ode_neuron",
 "ksP_ode_neuron",
 "k_cleave_ode_neuron",
 "k_p75_pro_on_ode_neuron",
 "k_p75_pro_off_ode_neuron",
 "k_degP_ode_neuron",
 "k_TrkB_pro_on_ode_neuron",
 "k_TrkB_pro_off_ode_neuron",
 "k_TrkB_B_on_ode_neuron",
 "k_TrkB_B_off_ode_neuron",
 "k_degB_ode_neuron",
 "k_p75_B_on_ode_neuron",
 "k_p75_B_off_ode_neuron",
 "k_degR1_ode_neuron",
 "k_degR2_ode_neuron",
 "k_int_p75_pro_ode_neuron",
 "k_int_p75_B_ode_neuron",
 "k_int_TrkB_B_ode_neuron",
 "k_int_TrkB_pro_ode_neuron",
 "aff_p75_pro_ode_neuron",
 "aff_p75_B_ode_neuron",
 "aff_TrkB_pro_ode_neuron",
 "aff_TrkB_B_ode_neuron",
 "k_deg_tPA_ode_neuron",
 "ks_tPA_ode_neuron",
 "ks_p75_ode_neuron",
 "ks_TrkB_ode_neuron",
 "tau_activity_ode_neuron",
 "activity_gain_ode_neuron",
 0,
 "growth_strength_ode_neuron",
 "apop_strength_ode_neuron",
 "syn_input_activity_ode_neuron",
 0,
 "P_ode_neuron",
 "B_ode_neuron",
 "p75_ode_neuron",
 "TrkB_ode_neuron",
 "p75_pro_ode_neuron",
 "p75_B_ode_neuron",
 "TrkB_B_ode_neuron",
 "TrkB_pro_ode_neuron",
 "tPA_ode_neuron",
 "activity_level_ode_neuron",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 58, _prop);
 	/*initialize range parameters*/
 	g_leak = 0.0001;
 	e_leak = -65;
 	v_threshold_spike = -20;
 	ksP = 0.005;
 	k_cleave = 0.01;
 	k_p75_pro_on = 1;
 	k_p75_pro_off = 0.9;
 	k_degP = 0.0005;
 	k_TrkB_pro_on = 0.2;
 	k_TrkB_pro_off = 0.1;
 	k_TrkB_B_on = 1;
 	k_TrkB_B_off = 0.9;
 	k_degB = 0.005;
 	k_p75_B_on = 0.3;
 	k_p75_B_off = 0.1;
 	k_degR1 = 0.0001;
 	k_degR2 = 1e-05;
 	k_int_p75_pro = 0.0005;
 	k_int_p75_B = 0.0005;
 	k_int_TrkB_B = 0.0005;
 	k_int_TrkB_pro = 0.0005;
 	aff_p75_pro = 0.9;
 	aff_p75_B = 0.1;
 	aff_TrkB_pro = 0.1;
 	aff_TrkB_B = 0.9;
 	k_deg_tPA = 0.0011;
 	ks_tPA = 0.0001;
 	ks_p75 = 0.0001;
 	ks_TrkB = 1e-05;
 	tau_activity = 50;
 	activity_gain = 0.1;
 	_prop->param = _p;
 	_prop->param_size = 58;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ode_neuron_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 58, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ode_neuron /Users/ethan/Documents/BURise/modeling_project/ode_neuron.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[10], _dlist1[10];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DP = 0.0 ;
   DB = 0.0 ;
   Dp75 = 0.0 ;
   DTrkB = 0.0 ;
   Dp75_pro = 0.0 ;
   Dp75_B = 0.0 ;
   DTrkB_B = 0.0 ;
   DTrkB_pro = 0.0 ;
   DtPA = 0.0 ;
   Dactivity_level = 0.0 ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DP = DP  / (1. - dt*( 0.0 )) ;
 DB = DB  / (1. - dt*( 0.0 )) ;
 Dp75 = Dp75  / (1. - dt*( 0.0 )) ;
 DTrkB = DTrkB  / (1. - dt*( 0.0 )) ;
 Dp75_pro = Dp75_pro  / (1. - dt*( 0.0 )) ;
 Dp75_B = Dp75_B  / (1. - dt*( 0.0 )) ;
 DTrkB_B = DTrkB_B  / (1. - dt*( 0.0 )) ;
 DTrkB_pro = DTrkB_pro  / (1. - dt*( 0.0 )) ;
 DtPA = DtPA  / (1. - dt*( 0.0 )) ;
 Dactivity_level = Dactivity_level  / (1. - dt*( 0.0 )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    P = P - dt*(- ( 0.0 ) ) ;
    B = B - dt*(- ( 0.0 ) ) ;
    p75 = p75 - dt*(- ( 0.0 ) ) ;
    TrkB = TrkB - dt*(- ( 0.0 ) ) ;
    p75_pro = p75_pro - dt*(- ( 0.0 ) ) ;
    p75_B = p75_B - dt*(- ( 0.0 ) ) ;
    TrkB_B = TrkB_B - dt*(- ( 0.0 ) ) ;
    TrkB_pro = TrkB_pro - dt*(- ( 0.0 ) ) ;
    tPA = tPA - dt*(- ( 0.0 ) ) ;
    activity_level = activity_level - dt*(- ( 0.0 ) ) ;
   }
  return 0;
}
 
double Hill ( _threadargsprotocomma_ double _lC , double _lKD , double _ln ) {
   double _lHill;
 _lHill = pow( _lC , _ln ) / ( pow( _lKD , _ln ) + pow( _lC , _ln ) ) ;
   
return _lHill;
 }
 
static void _hoc_Hill(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  Hill ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 10;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 10; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  B = B0;
  P = P0;
  TrkB_pro = TrkB_pro0;
  TrkB_B = TrkB_B0;
  TrkB = TrkB0;
  activity_level = activity_level0;
  p75_B = p75_B0;
  p75_pro = p75_pro0;
  p75 = p750;
  tPA = tPA0;
 {
   P = 0.2 ;
   B = 0.0 ;
   p75 = 1.0 ;
   TrkB = 1.0 ;
   p75_pro = 0.0 ;
   p75_B = 0.0 ;
   TrkB_B = 0.0 ;
   TrkB_pro = 0.0 ;
   tPA = 0.1 ;
   activity_level = 0.0 ;
   syn_input_activity = 0.0 ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{
} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {   states(_p, _ppvar, _thread, _nt);
  } {
   ks_P_variable = ksP * ( 1.0 + activity_level ) ;
   ks_tPA_variable = ks_tPA * ( 1.0 + activity_level ) ;
   growth_strength = 0.0 ;
   apop_strength = 0.0 ;
   }
}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = P_columnindex;  _dlist1[0] = DP_columnindex;
 _slist1[1] = B_columnindex;  _dlist1[1] = DB_columnindex;
 _slist1[2] = p75_columnindex;  _dlist1[2] = Dp75_columnindex;
 _slist1[3] = TrkB_columnindex;  _dlist1[3] = DTrkB_columnindex;
 _slist1[4] = p75_pro_columnindex;  _dlist1[4] = Dp75_pro_columnindex;
 _slist1[5] = p75_B_columnindex;  _dlist1[5] = Dp75_B_columnindex;
 _slist1[6] = TrkB_B_columnindex;  _dlist1[6] = DTrkB_B_columnindex;
 _slist1[7] = TrkB_pro_columnindex;  _dlist1[7] = DTrkB_pro_columnindex;
 _slist1[8] = tPA_columnindex;  _dlist1[8] = DtPA_columnindex;
 _slist1[9] = activity_level_columnindex;  _dlist1[9] = Dactivity_level_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/ethan/Documents/BURise/modeling_project/ode_neuron.mod";
static const char* nmodl_file_text = 
  "NEURON {\n"
  "    SUFFIX ode_neuron\n"
  "    RANGE g_leak, e_leak\n"
  "    RANGE P, B, p75, TrkB, p75_pro, p75_B, TrkB_B, TrkB_pro, tPA\n"
  "    RANGE ksP, k_cleave, k_p75_pro_on, k_p75_pro_off, k_degP, k_TrkB_pro_on, k_TrkB_pro_off\n"
  "    RANGE k_TrkB_B_on, k_TrkB_B_off, k_degB, k_p75_B_on, k_p75_B_off, k_degR1, k_degR2\n"
  "    RANGE k_int_p75_pro, k_int_p75_B, k_int_TrkB_B, k_int_TrkB_pro, aff_p75_pro\n"
  "    RANGE aff_p75_B, aff_TrkB_pro, aff_TrkB_B, k_deg_tPA, ks_tPA, ks_p75, ks_TrkB\n"
  "    RANGE activity_level\n"
  "    RANGE v_threshold_spike\n"
  "    RANGE growth_strength, apop_strength\n"
  "    RANGE syn_input_activity\n"
  "    RANGE tau_activity, activity_gain\n"
  "    THREADSAFE\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (molar) = (1)\n"
  "    (mM) = (millimolar)\n"
  "    (nM) = (nanomolar)\n"
  "    (uM) = (micromolar)\n"
  "    (pM) = (picomolar)\n"
  "    (mS) = (millisiemens)\n"
  "    (MS) = (/micromolar/second)\n"
  "    (nA) = (nanoamp)\n"
  "    (mV) = (millivolt)\n"
  "    (uF) = (microfarad)\n"
  "    (S) = (siemens)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    : Parameters associated with membrane properties\n"
  "    g_leak = 0.0001 (S/cm2) : Leak conductance\n"
  "    e_leak = -65 (mV)       : Leak reversal potential\n"
  "    v_threshold_spike = -20 (mV) : Threshold for spike detection (constant)\n"
  "\n"
  "    : Your existing ODE parameters (ensure these match your Python param list order)\n"
  "    ksP = 5.0e-3 (uM/s)   : synthesis rate of proBDNF\n"
  "    k_cleave = 0.01 (/s)    : rate of proBDNF cleavage into BDNF\n"
  "    k_p75_pro_on = 1.0 (MS)    : proBDNF binding to p75\n"
  "    k_p75_pro_off = 0.9 (/s)    : proBDNF unbinding from p75\n"
  "    k_degP = 5.0e-4 (/s)   : proBDNF degradation\n"
  "    k_TrkB_pro_on = 0.2 (MS)    : proBDNF binding to TrkB\n"
  "    k_TrkB_pro_off = 0.1 (/s)   : proBDNF unbinding from TrkB\n"
  "    k_TrkB_B_on = 1.0 (MS)    : BDNF binding to TrkB\n"
  "    k_TrkB_B_off = 0.9 (/s)    : BDNF unbinding from TrkB\n"
  "    k_degB = 0.005 (/s)    : BDNF degradation\n"
  "    k_p75_B_on = 0.3 (MS)    : BDNF binding to p75\n"
  "    k_p75_B_off = 0.1 (/s)   : BDNF unbinding from p75\n"
  "    k_degR1 = 0.0001 (/s)   : p75 degradation\n"
  "    k_degR2 = 0.00001 (/s)   : TrkB degradation\n"
  "    k_int_p75_pro = 0.0005 (/s)    : proBDNF-p75 internalization\n"
  "    k_int_p75_B = 0.0005 (/s)    : BDNF-p75 internalization\n"
  "    k_int_TrkB_B = 0.0005 (/s)    : BDNF-TrkB internalization\n"
  "    k_int_TrkB_pro = 0.0005 (/s)    : proBDNF-TrkB internalization\n"
  "    aff_p75_pro = 0.9    : affinity of proBDNF for p75\n"
  "    aff_p75_B = 0.1    : affinity of BDNF for p75\n"
  "    aff_TrkB_pro = 0.1    : affinity of proBDNF for TrkB\n"
  "    aff_TrkB_B = 0.9    : affinity of BDNF for TrkB\n"
  "    k_deg_tPA = 0.0011 (/s)   : degradation rate of tPA - slow degradation\n"
  "    ks_tPA = 0.0001 (uM/s)    : synthesis rate of tPA\n"
  "    ks_p75 = 0.0001 (uM/s)    : synthesis rate of p75 - small value to maintain baseline\n"
  "    ks_TrkB = 0.00001 (uM/s)    : synthesis rate of TrkB - small value to maintain baseline\n"
  "    \n"
  "    : Parameters for activity level dynamics\n"
  "    tau_activity = 50 (ms) : Time constant for activity_level decay\n"
  "    activity_gain = 0.1 : How much one synaptic event boosts activity_level\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    v (mV) : Membrane potential\n"
  "    \n"
  "    ks_P_variable (uM/s)\n"
  "    ks_tPA_variable (uM/s)\n"
  "    growth_strength \n"
  "    apop_strength \n"
  "    syn_input_activity : Input from synapses\n"
  "}\n"
  "\n"
  "STATE {\n"
  "    P (uM)    : proBDNF concentration\n"
  "    B (uM)    : BDNF concentration\n"
  "    p75 (uM)    : p75 receptor concentration\n"
  "    TrkB (uM)    : TrkB receptor concentration\n"
  "    p75_pro (uM)    : proBDNF bound to p75\n"
  "    p75_B (uM)    : BDNF bound to p75\n"
  "    TrkB_B (uM)    : BDNF bound to TrkB\n"
  "    TrkB_pro (uM)    : proBDNF bound to TrkB\n"
  "    tPA (uM)    : tPA concentration\n"
  "    activity_level : Activity level factor\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    P = 0.2    : Initial concentration of proBDNF\n"
  "    B = 0.0    : Initial concentration of BDNF\n"
  "    p75 = 1.0    : Initial concentration of p75 receptor\n"
  "    TrkB = 1.0    : Initial concentration of TrkB receptor\n"
  "    p75_pro = 0.0    : Initial concentration of proBDNF bound to p75\n"
  "    p75_B = 0.0    : Initial concentration of BDNF bound to p75\n"
  "    TrkB_B = 0.0    : Initial concentration of BDNF bound to TrkB\n"
  "    TrkB_pro = 0.0    : Initial concentration of proBDNF bound to TrkB\n"
  "    tPA = 0.1    : Initial concentration of tPA\n"
  "    activity_level = 0.0 : Initial activity level\n"
  "    syn_input_activity = 0.0 : Initialize syn_input_activity\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE states METHOD cnexp\n"
  "    \n"
  "    : Calculate current contributed by this mechanism (leak current)\n"
  "    \n"
  "    ks_P_variable = ksP * (1 + activity_level)\n"
  "    ks_tPA_variable = ks_tPA * (1 + activity_level)\n"
  "\n"
  "    : growth_strength = (Hill(TrkB_B, 0.05, 2) + Hill(TrkB_pro, 0.02, 2))/2\n"
  "    : apop_strength = (Hill(p75_pro, 0.02, 2) + Hill(p75_B, 0.02, 2))/2\n"
  "    growth_strength = 0\n"
  "    apop_strength = 0\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "    :P' = ks_P_variable - k_cleave * tPA * P - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degP * P\n"
  "        \n"
  "    :B' = k_cleave * tPA * P - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degB * B\n"
  "    \n"
  "    :p75' = ks_p75 - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degR1 * p75\n"
  "        \n"
  "    :TrkB' = ks_TrkB - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degR2 * TrkB\n"
  "        \n"
  "    :p75_pro' = k_p75_pro_on * aff_p75_pro * P * p75 - k_p75_pro_off * p75_pro - k_int_p75_pro * p75_pro\n"
  "      \n"
  "    :p75_B' = k_p75_B_on * aff_p75_B * B * p75 - k_p75_B_off * p75_B - k_int_p75_B * p75_B\n"
  "        \n"
  "    :TrkB_B' = k_TrkB_B_on * aff_TrkB_B * B * TrkB - k_TrkB_B_off * TrkB_B - k_int_TrkB_B * TrkB_B\n"
  "        \n"
  "    : TrkB_pro' = k_TrkB_pro_on * aff_TrkB_pro * P * TrkB - k_TrkB_pro_off * TrkB_pro - k_int_TrkB_pro * TrkB_pro\n"
  "        \n"
  "    : tPA' = ks_tPA_variable - k_deg_tPA * tPA\n"
  "\n"
  "    P' = 0\n"
  "    B' = 0\n"
  "    \n"
  "    p75' = 0\n"
  "        \n"
  "    TrkB' = 0\n"
  "        \n"
  "    p75_pro' = 0\n"
  "      \n"
  "    p75_B' = 0\n"
  "        \n"
  "    TrkB_B' = 0\n"
  "        \n"
  "    TrkB_pro' = 0\n"
  "        \n"
  "    tPA' = 0\n"
  "\n"
  "    : activity_level' = -activity_level / tau_activity + syn_input_activity\n"
  "    activity_level' = 0\n"
  "}\n"
  "\n"
  "FUNCTION Hill(C, KD, n) {\n"
  "    Hill = C^n / (KD^n + C^n)\n"
  "}\n"
  ;
#endif
