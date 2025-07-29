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
#define integrate integrate__ode_neuron 
 
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
#define ksP _p[0]
#define ksP_columnindex 0
#define k_cleave _p[1]
#define k_cleave_columnindex 1
#define k_p75_pro_on _p[2]
#define k_p75_pro_on_columnindex 2
#define k_p75_pro_off _p[3]
#define k_p75_pro_off_columnindex 3
#define k_degP _p[4]
#define k_degP_columnindex 4
#define k_TrkB_pro_on _p[5]
#define k_TrkB_pro_on_columnindex 5
#define k_TrkB_pro_off _p[6]
#define k_TrkB_pro_off_columnindex 6
#define k_TrkB_B_on _p[7]
#define k_TrkB_B_on_columnindex 7
#define k_TrkB_B_off _p[8]
#define k_TrkB_B_off_columnindex 8
#define k_degB _p[9]
#define k_degB_columnindex 9
#define k_p75_B_on _p[10]
#define k_p75_B_on_columnindex 10
#define k_p75_B_off _p[11]
#define k_p75_B_off_columnindex 11
#define k_degR1 _p[12]
#define k_degR1_columnindex 12
#define k_degR2 _p[13]
#define k_degR2_columnindex 13
#define k_int_p75_pro _p[14]
#define k_int_p75_pro_columnindex 14
#define k_int_p75_B _p[15]
#define k_int_p75_B_columnindex 15
#define k_int_TrkB_B _p[16]
#define k_int_TrkB_B_columnindex 16
#define k_int_TrkB_pro _p[17]
#define k_int_TrkB_pro_columnindex 17
#define aff_p75_pro _p[18]
#define aff_p75_pro_columnindex 18
#define aff_p75_B _p[19]
#define aff_p75_B_columnindex 19
#define aff_TrkB_pro _p[20]
#define aff_TrkB_pro_columnindex 20
#define aff_TrkB_B _p[21]
#define aff_TrkB_B_columnindex 21
#define k_deg_tPA _p[22]
#define k_deg_tPA_columnindex 22
#define ks_tPA _p[23]
#define ks_tPA_columnindex 23
#define ks_p75 _p[24]
#define ks_p75_columnindex 24
#define ks_TrkB _p[25]
#define ks_TrkB_columnindex 25
#define activity_level _p[26]
#define activity_level_columnindex 26
#define growth_strength _p[27]
#define growth_strength_columnindex 27
#define apop_strength _p[28]
#define apop_strength_columnindex 28
#define i _p[29]
#define i_columnindex 29
#define P _p[30]
#define P_columnindex 30
#define B _p[31]
#define B_columnindex 31
#define p75 _p[32]
#define p75_columnindex 32
#define TrkB _p[33]
#define TrkB_columnindex 33
#define p75_pro _p[34]
#define p75_pro_columnindex 34
#define p75_B _p[35]
#define p75_B_columnindex 35
#define TrkB_B _p[36]
#define TrkB_B_columnindex 36
#define TrkB_pro _p[37]
#define TrkB_pro_columnindex 37
#define tPA _p[38]
#define tPA_columnindex 38
#define ks_P_variable _p[39]
#define ks_P_variable_columnindex 39
#define ks_tPA_variable _p[40]
#define ks_tPA_variable_columnindex 40
#define ica _p[41]
#define ica_columnindex 41
#define DP _p[42]
#define DP_columnindex 42
#define DB _p[43]
#define DB_columnindex 43
#define Dp75 _p[44]
#define Dp75_columnindex 44
#define DTrkB _p[45]
#define DTrkB_columnindex 45
#define Dp75_pro _p[46]
#define Dp75_pro_columnindex 46
#define Dp75_B _p[47]
#define Dp75_B_columnindex 47
#define DTrkB_B _p[48]
#define DTrkB_B_columnindex 48
#define DTrkB_pro _p[49]
#define DTrkB_pro_columnindex 49
#define DtPA _p[50]
#define DtPA_columnindex 50
#define v _p[51]
#define v_columnindex 51
#define _g _p[52]
#define _g_columnindex 52
#define _ion_ica	*_ppvar[0]._pval
#define _ion_dicadv	*_ppvar[1]._pval
 
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
 "ksP_ode_neuron", "uM/s",
 "k_cleave_ode_neuron", "1/s",
 "k_p75_pro_on_ode_neuron", "MS",
 "k_p75_pro_off_ode_neuron", "1/s",
 "k_degP_ode_neuron", "1/s",
 "k_TrkB_pro_on_ode_neuron", "MS",
 "k_TrkB_pro_off_ode_neuron", "1/s",
 "k_TrkB_B_on_ode_neuron", "MS",
 "k_TrkB_B_off_ode_neuron", "1/s",
 "k_degB_ode_neuron", "1/s",
 "k_p75_B_on_ode_neuron", "MS",
 "k_p75_B_off_ode_neuron", "1/s",
 "k_degR1_ode_neuron", "1/s",
 "k_degR2_ode_neuron", "1/s",
 "k_int_p75_pro_ode_neuron", "1/s",
 "k_int_p75_B_ode_neuron", "1/s",
 "k_int_TrkB_B_ode_neuron", "1/s",
 "k_int_TrkB_pro_ode_neuron", "1/s",
 "k_deg_tPA_ode_neuron", "1/s",
 "ks_tPA_ode_neuron", "uM/s",
 "ks_p75_ode_neuron", "uM/s",
 "ks_TrkB_ode_neuron", "uM/s",
 "P_ode_neuron", "uM",
 "B_ode_neuron", "uM",
 "p75_ode_neuron", "uM",
 "TrkB_ode_neuron", "uM",
 "p75_pro_ode_neuron", "uM",
 "p75_B_ode_neuron", "uM",
 "TrkB_B_ode_neuron", "uM",
 "TrkB_pro_ode_neuron", "uM",
 "tPA_ode_neuron", "uM",
 "i_ode_neuron", "mA/cm2",
 0,0
};
 static double B0 = 0;
 static double P0 = 0;
 static double TrkB_pro0 = 0;
 static double TrkB_B0 = 0;
 static double TrkB0 = 0;
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
 
#define _cvode_ieq _ppvar[2]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ode_neuron",
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
 "activity_level_ode_neuron",
 0,
 "growth_strength_ode_neuron",
 "apop_strength_ode_neuron",
 "i_ode_neuron",
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
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 53, _prop);
 	/*initialize range parameters*/
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
 	activity_level = 1;
 	_prop->param = _p;
 	_prop->param_size = 53;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 	_ppvar[0]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[1]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _thread_mem_init(Datum*);
 static void _thread_cleanup(Datum*);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ode_neuron_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 5);
  _extcall_thread = (Datum*)ecalloc(4, sizeof(Datum));
  _thread_mem_init(_extcall_thread);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 53, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "cvodeieq");
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
 
#define _deriv1_advance _thread[0]._i
#define _dith1 1
#define _recurse _thread[2]._i
#define _newtonspace1 _thread[3]._pvoid
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist2[9];
  static int _slist1[9], _dlist1[9];
 static int integrate(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DP = ks_P_variable - k_cleave * tPA * P - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degP * P ;
   DB = k_cleave * tPA * P - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degB * B ;
   Dp75 = ks_p75 - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degR1 * p75 ;
   DTrkB = ks_TrkB - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degR2 * TrkB ;
   Dp75_pro = k_p75_pro_on * aff_p75_pro * P * p75 - k_p75_pro_off * p75_pro - k_int_p75_pro * p75_pro ;
   Dp75_B = k_p75_B_on * aff_p75_B * B * p75 - k_p75_B_off * p75_B - k_int_p75_B * p75_B ;
   DTrkB_B = k_TrkB_B_on * aff_TrkB_B * B * TrkB - k_TrkB_B_off * TrkB_B - k_int_TrkB_B * TrkB_B ;
   DTrkB_pro = k_TrkB_pro_on * aff_TrkB_pro * P * TrkB - k_TrkB_pro_off * TrkB_pro - k_int_TrkB_pro * TrkB_pro ;
   DtPA = ks_tPA_variable - k_deg_tPA * tPA ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DP = DP  / (1. - dt*( ( - ( k_cleave * tPA )*( 1.0 ) ) - ( ( k_p75_pro_on * aff_p75_pro )*( 1.0 ) )*( p75 ) - ( ( k_TrkB_pro_on * aff_TrkB_pro )*( 1.0 ) )*( TrkB ) - ( k_degP )*( 1.0 ) )) ;
 DB = DB  / (1. - dt*( ( - ( ( k_TrkB_B_on * aff_TrkB_B )*( 1.0 ) )*( TrkB ) ) - ( ( k_p75_B_on * aff_p75_B )*( 1.0 ) )*( p75 ) - ( k_degB )*( 1.0 ) )) ;
 Dp75 = Dp75  / (1. - dt*( ( - ( k_p75_pro_on * aff_p75_pro * P )*( 1.0 ) ) - ( k_p75_B_on * aff_p75_B * B )*( 1.0 ) - ( k_degR1 )*( 1.0 ) )) ;
 DTrkB = DTrkB  / (1. - dt*( ( - ( k_TrkB_B_on * aff_TrkB_B * B )*( 1.0 ) ) - ( k_TrkB_pro_on * aff_TrkB_pro * P )*( 1.0 ) - ( k_degR2 )*( 1.0 ) )) ;
 Dp75_pro = Dp75_pro  / (1. - dt*( ( - ( k_p75_pro_off )*( 1.0 ) ) - ( k_int_p75_pro )*( 1.0 ) )) ;
 Dp75_B = Dp75_B  / (1. - dt*( ( - ( k_p75_B_off )*( 1.0 ) ) - ( k_int_p75_B )*( 1.0 ) )) ;
 DTrkB_B = DTrkB_B  / (1. - dt*( ( - ( k_TrkB_B_off )*( 1.0 ) ) - ( k_int_TrkB_B )*( 1.0 ) )) ;
 DTrkB_pro = DTrkB_pro  / (1. - dt*( ( - ( k_TrkB_pro_off )*( 1.0 ) ) - ( k_int_TrkB_pro )*( 1.0 ) )) ;
 DtPA = DtPA  / (1. - dt*( ( - ( k_deg_tPA )*( 1.0 ) ) )) ;
  return 0;
}
 /*END CVODE*/
 
static int integrate (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0; int error = 0;
 { double* _savstate1 = _thread[_dith1]._pval;
 double* _dlist2 = _thread[_dith1]._pval + 9;
 int _counte = -1;
 if (!_recurse) {
 _recurse = 1;
 {int _id; for(_id=0; _id < 9; _id++) { _savstate1[_id] = _p[_slist1[_id]];}}
 error = nrn_newton_thread(_newtonspace1, 9,_slist2, _p, integrate, _dlist2, _ppvar, _thread, _nt);
 _recurse = 0; if(error) {abort_run(error);}}
 {
   DP = ks_P_variable - k_cleave * tPA * P - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degP * P ;
   DB = k_cleave * tPA * P - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degB * B ;
   Dp75 = ks_p75 - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degR1 * p75 ;
   DTrkB = ks_TrkB - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degR2 * TrkB ;
   Dp75_pro = k_p75_pro_on * aff_p75_pro * P * p75 - k_p75_pro_off * p75_pro - k_int_p75_pro * p75_pro ;
   Dp75_B = k_p75_B_on * aff_p75_B * B * p75 - k_p75_B_off * p75_B - k_int_p75_B * p75_B ;
   DTrkB_B = k_TrkB_B_on * aff_TrkB_B * B * TrkB - k_TrkB_B_off * TrkB_B - k_int_TrkB_B * TrkB_B ;
   DTrkB_pro = k_TrkB_pro_on * aff_TrkB_pro * P * TrkB - k_TrkB_pro_off * TrkB_pro - k_int_TrkB_pro * TrkB_pro ;
   DtPA = ks_tPA_variable - k_deg_tPA * tPA ;
   {int _id; for(_id=0; _id < 9; _id++) {
if (_deriv1_advance) {
 _dlist2[++_counte] = _p[_dlist1[_id]] - (_p[_slist1[_id]] - _savstate1[_id])/dt;
 }else{
_dlist2[++_counte] = _p[_slist1[_id]] - _savstate1[_id];}}}
 } }
 return _reset;}
 
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
 
static int _ode_count(int _type){ return 9;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ica = _ion_ica;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 9; ++_i) {
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
  ica = _ion_ica;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
   _thread[_dith1]._pval = (double*)ecalloc(18, sizeof(double));
   _newtonspace1 = nrn_cons_newtonspace(9);
 }
 
static void _thread_cleanup(Datum* _thread) {
   free((void*)(_thread[_dith1]._pval));
   nrn_destroy_newtonspace(_newtonspace1);
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  B = B0;
  P = P0;
  TrkB_pro = TrkB_pro0;
  TrkB_B = TrkB_B0;
  TrkB = TrkB0;
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
   ica = 0.0 ;
   i = 0.0 ;
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
  ica = _ion_ica;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ks_P_variable = ksP * activity_level ;
   ks_tPA_variable = ks_tPA * activity_level ;
   growth_strength = ( Hill ( _threadargscomma_ TrkB_B , 0.05 , 2.0 ) + Hill ( _threadargscomma_ TrkB_pro , 0.02 , 2.0 ) ) / 2.0 ;
   apop_strength = ( Hill ( _threadargscomma_ p75_pro , 0.02 , 2.0 ) + Hill ( _threadargscomma_ p75_B , 0.02 , 2.0 ) ) / 2.0 ;
   }
 _current += ica;
 _current += i;

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
  ica = _ion_ica;
if (_nt->_vcv) { _ode_spec1(_p, _ppvar, _thread, _nt); }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
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
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
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
  ica = _ion_ica;
 {  _deriv1_advance = 1;
 derivimplicit_thread(9, _slist1, _dlist1, _p, integrate, _ppvar, _thread, _nt);
_deriv1_advance = 0;
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 9; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 } }}
 dt = _dtsav;
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
 _slist2[0] = B_columnindex;
 _slist2[1] = P_columnindex;
 _slist2[2] = TrkB_pro_columnindex;
 _slist2[3] = TrkB_B_columnindex;
 _slist2[4] = TrkB_columnindex;
 _slist2[5] = p75_B_columnindex;
 _slist2[6] = p75_pro_columnindex;
 _slist2[7] = p75_columnindex;
 _slist2[8] = tPA_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/ethan/Documents/BURise/modeling_project/ode_neuron.mod";
static const char* nmodl_file_text = 
  "UNITS {\n"
  "    (molar) = (1)\n"
  "    (mM) = (millimolar)\n"
  "    (nM) = (nanomolar)\n"
  "    (uM) = (micromolar)\n"
  "    (pM) = (picomolar)\n"
  "    (mS) = (millisiemens)\n"
  "    (MS) = (1/ micromolar 1/ second)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    ksP = 5.0e-3 (uM/s)   : ksP (synthesis rate of proBDNF)\n"
  "    k_cleave = 0.01 (1/s)    : k_cleave (rate of proBDNF cleavage into BDNF)\n"
  "    k_p75_pro_on = 1.0 (MS)    : k_p75_pro_on (proBDNF binding to p75)\n"
  "    k_p75_pro_off = 0.9 (1/s)    : k_p75_pro_off (proBDNF unbinding from p75)\n"
  "    k_degP = 5.0e-4 (1/s)   : k_degP (proBDNF degradation )\n"
  "    k_TrkB_pro_on = 0.2 (MS)    : k_TrkB_pro_on (proBDNF binding to TrkB)\n"
  "    k_TrkB_pro_off = 0.1 (1/s)   : k_TrkB_pro_off (proBDNF unbinding from TrkB)\n"
  "    k_TrkB_B_on = 1.0 (MS)    : k_TrkB_B_on (BDNF binding to TrkB)\n"
  "    k_TrkB_B_off = 0.9 (1/s)    : k_TrkB_B_off (BDNF unbinding from TrkB)\n"
  "    k_degB = 0.005 (1/s)    : k_degB (BDNF degradation)\n"
  "    k_p75_B_on = 0.3 (MS)    : k_p75_B_on (BDNF binding to p75)\n"
  "    k_p75_B_off = 0.1 (1/s)   : k_p75_B_off (BDNF unbinding from p75)\n"
  "    k_degR1 = 0.0001 (1/s)   : k_degR1 (p75 degradation)\n"
  "    k_degR2 = 0.00001 (1/s)   : k_degR2 (TrkB degradation)\n"
  "    k_int_p75_pro = 0.0005 (1/s)    : k_int_p75_pro (proBDNF-p75 internalization)\n"
  "    k_int_p75_B = 0.0005 (1/s)    : k_int_p75_B (BDNF-p75 internalization)\n"
  "    k_int_TrkB_B = 0.0005 (1/s)    : k_int_TrkB_B (BDNF-TrkB internalization)\n"
  "    k_int_TrkB_pro = 0.0005 (1/s)    : k_int_TrkB_pro (proBDNF-TrkB internalization)\n"
  "    aff_p75_pro = 0.9    : aff_p75_pro (affinity of proBDNF for p75)\n"
  "    aff_p75_B = 0.1    : aff_p75_B (affinity of BDNF for p75)\n"
  "    aff_TrkB_pro = 0.1    : aff_TrkB_pro (affinity of proBDNF for TrkB)\n"
  "    aff_TrkB_B = 0.9    : aff_TrkB_B (affinity of BDNF for TrkB)\n"
  "    k_deg_tPA = 0.0011 (1/s)   :k_deg_tPA (degradation rate of tPA) - slow degradation\n"
  "    ks_tPA = 0.0001 (uM/s)    : ks_tPA (synthesis rate of tPA)\n"
  "    : NEW PARAMETERS FOR BIOLOGICAL ACCURACY\n"
  "    ks_p75 = 0.0001 (uM/s)    : ks_p75 (synthesis rate of p75) - small value to maintain baseline\n"
  "    ks_TrkB = 0.00001 (uM/s)    : ks_TrkB (synthesis rate of TrkB) - small value to maintain baseline\n"
  "    activity_level = 1.0   : activity level factor (default is 1, can be adjusted)\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "    SUFFIX ode_neuron\n"
  "    USEION ca READ ica WRITE ica\n"
  "    NONSPECIFIC_CURRENT i\n"
  "    RANGE P, B, p75, TrkB, p75_pro, p75_B, TrkB_B, TrkB_pro, tPA\n"
  "    RANGE ksP, k_cleave, k_p75_pro_on, k_p75_pro_off, k_degP, k_TrkB_pro_on, k_TrkB_pro_off\n"
  "    RANGE k_TrkB_B_on, k_TrkB_B_off, k_degB, k_p75_B_on, k_p75_B_off, k_degR1, k_degR2\n"
  "    RANGE k_int_p75_pro, k_int_p75_B, k_int_TrkB_B, k_int_TrkB_pro, aff_p75_pro\n"
  "    RANGE aff_p75_B, aff_TrkB_pro, aff_TrkB_B, k_deg_tPA, ks_tPA, ks_p75, ks_TrkB\n"
  "\n"
  "    RANGE activity_level\n"
  "\n"
  "    RANGE growth_strength, apop_strength\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    ks_P_variable (uM/s)  : Adjusted synthesis rate of proBDNF based on activity level\n"
  "    ks_tPA_variable (uM/s)  : Adjusted synthesis rate of tPA based on activity level\n"
  "    growth_strength  : Growth strength factor\n"
  "    apop_strength    : Apoptosis strength factor\n"
  "    ica (mA/cm2)\n"
  "    i (mA/cm2)\n"
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
  "\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    P = 0.2 (uM)    : Initial concentration of proBDNF\n"
  "    B = 0.0 (uM)    : Initial concentration of BDNF\n"
  "    p75 = 1.0 (uM)    : Initial concentration of p75 receptor\n"
  "    TrkB = 1.0 (uM)    : Initial concentration of TrkB receptor\n"
  "    p75_pro = 0.0 (uM)    : Initial concentration of proBDNF bound to p75\n"
  "    p75_B = 0.0 (uM)    : Initial concentration of BDNF bound to p75\n"
  "    TrkB_B = 0.0 (uM)    : Initial concentration of BDNF bound to TrkB\n"
  "    TrkB_pro = 0.0 (uM)    : Initial concentration of proBDNF bound to TrkB\n"
  "    tPA = 0.1 (uM)    : Initial concentration of tPA\n"
  "    ica = 0\n"
  "    i = 0\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE integrate METHOD derivimplicit\n"
  "    ks_P_variable = ksP * activity_level  : Adjusted synthesis rate of proBDNF based on activity level\n"
  "    ks_tPA_variable = ks_tPA * activity_level  : Adjusted synthesis rate of tPA based on activity level\n"
  "\n"
  "    growth_strength = (Hill(TrkB_B, 0.05 (uM), 2) + Hill(TrkB_pro, 0.02 (uM), 2))/2 : Growth strength based on proBDNF and BDNF concentrations\n"
  "    apop_strength = (Hill(p75_pro, 0.02 (uM), 2) + Hill(p75_B, 0.02 (uM), 2))/2 : Apoptosis strength based on proBDNF concentration\n"
  "\n"
  "   \n"
  "}\n"
  "\n"
  "DERIVATIVE integrate {\n"
  "    P' = ks_P_variable - k_cleave * tPA * P - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degP * P\n"
  "        \n"
  "    B' = k_cleave * tPA * P - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degB * B\n"
  "    \n"
  "    p75' = ks_p75 - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degR1 * p75\n"
  "        \n"
  "    TrkB' = ks_TrkB - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degR2 * TrkB\n"
  "        \n"
  "    p75_pro' = k_p75_pro_on * aff_p75_pro * P * p75 - k_p75_pro_off * p75_pro - k_int_p75_pro * p75_pro\n"
  "        \n"
  "    p75_B' = k_p75_B_on * aff_p75_B * B * p75 - k_p75_B_off * p75_B - k_int_p75_B * p75_B\n"
  "        \n"
  "    TrkB_B' = k_TrkB_B_on * aff_TrkB_B * B * TrkB - k_TrkB_B_off * TrkB_B - k_int_TrkB_B * TrkB_B\n"
  "        \n"
  "    TrkB_pro' = k_TrkB_pro_on * aff_TrkB_pro * P * TrkB - k_TrkB_pro_off * TrkB_pro - k_int_TrkB_pro * TrkB_pro\n"
  "        \n"
  "    tPA' = ks_tPA_variable - k_deg_tPA * tPA \n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "FUNCTION Hill(C (uM), KD (uM), n) {\n"
  "    : C: Concentration (uM)\n"
  "    : KD: Half-maximal concentration (uM)\n"
  "    : n: Hill coefficient (dimensionless)\n"
  "    \n"
  "    Hill = C^n / (KD^n + C^n)\n"
  "}\n"
  "\n"
  ;
#endif
