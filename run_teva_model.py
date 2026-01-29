from classes import TEVAHandler
import pandas as pd


teva_h = TEVAHandler()
wrk_dir = '/root/repositories/sedregfromedz/data/teva/'
suffix = '_edzsonly' # _binned_edzsonly
which = "all" # "confinement" "vertical (dis)connection" "tr vs. cst" "fstcd vs. ust" "cefd vs. dep" "all"


## Lateral confinement
if which == "confinement":
    data = pd.read_csv(wrk_dir + 'Dataset_lat_conf_TEVA' + suffix + '.csv')
    classifications = data.pop('lat_conf')
    out_suffix = 'lat_conf' + suffix

# Vertical connection
if which == "vertical (dis)connection":
    data = pd.read_csv(wrk_dir + 'Dataset_vert_conn_TEVA' + suffix + '.csv')
    classifications = data.pop('vert_conn')
    out_suffix = 'vert_conn' + suffix

## FSTCD vs UST
if which == "fstcd vs. ust":
    data = pd.read_csv(wrk_dir+'Dataset_FSTCD_UST_TEVA' + suffix + '.csv')
    classifications = data.Ph2SedReg.to_numpy()
    out_suffix = 'FSTCD_UST' + suffix

# TR vs CST
if which == "tr vs. cst":
    data = pd.read_csv(wrk_dir+'Dataset_TR_CST_TEVA' + suffix + '.csv')
    classifications = data.Ph2SedReg.to_numpy()
    out_suffix = 'TR_CST' + suffix

## CEFD vs DEP
if which == "cefd vs. dep":
    data = pd.read_csv(wrk_dir+'Dataset_CEFD_DEP_TEVA' + suffix + '.csv')
    classifications = data.Ph2SedReg.to_numpy()
    out_suffix = 'CEFD_DEP' + suffix

## All sed regimes
if which == "all":
    data = pd.read_csv(wrk_dir+'Dataset_lat_conf_TEVA' + suffix + '.csv')  # vertical confinement has all of the input data, too; just need to select the different classifications
    data.pop('lat_conf') # still pop this column off the dataframe
    classifications = data.Ph2SedReg.to_numpy()
    out_suffix = 'all' + suffix

teva_h.run_teva_model(data, classifications, wrk_dir, out_suffix=out_suffix)