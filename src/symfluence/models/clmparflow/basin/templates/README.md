# CLM driver templates

`drv_clmin.dat` and `drv_vegp.dat` are copied unchanged from ParFlow v3.15.0
(`test/tcl/clm/`, LGPL-2.1, https://github.com/parflow/parflow). Basin mode rewrites the dates, initial
temperature/snow and restart flags of `drv_clmin.dat` per segment (`case.write_clm_files`) and uses the IGBP
vegetation parameters of `drv_vegp.dat` as they are: class 15 (snow/ice, CLM land ice) on glacier cells,
class 16 (barren) elsewhere.
