from classes import TEVAHandler
import pandas as pd


teva_h = TEVAHandler()
wrk_dir = '/root/repositories/sedregfromedz/data/teva/'
suffix = '_edzsonly' # _binned_edzsonly
which = "confinement" # "confinement" "vertical (dis)connection" "tr vs. cst" "fstcd vs. ust" "all"


## Lateral confinement
if which == "confinement":
    data = pd.read_csv(wrk_dir + 'Dataset_lat_conf_TEVA' + suffix + '.csv')
    classifications = data.pop('lat_conf')

# Vertical connection
if which == "vertical (dis)connection":
    data = pd.read_csv(wrk_dir + 'Dataset_vert_conn_TEVA' + suffix + '.csv')
    classifications = data.pop('vert_conn')

## FSTCD vs UST
if which == "fstcd vs. ust":
    data = pd.read_csv(wrk_dir+'Dataset_FSTCD_UST_TEVA' + suffix + '.csv')
    classifications = data.Ph2SedReg.to_numpy()

# TR vs CST
if which == "tr vs. cst":
    data = pd.read_csv(wrk_dir+'Dataset_TR_CST_TEVA' + suffix + '.csv')
    classifications = data.Ph2SedReg.to_numpy()

## CEFD vs DEP
if which == "cefd vs. dep":
    data = pd.read_csv(wrk_dir+'Dataset_CEFD_DEP_TEVA' + suffix + '.csv')
    classifications = data.Ph2SedReg.to_numpy()

## All sed regimes
if which == "all":
    data = pd.read_csv(wrk_dir+'Dataset_lat_conf_TEVA' + suffix + '.csv')  # vertical confinement has all of the input data, too; just need to select the different classifications
    data.pop('lat_conf') # still pop this column off the dataframe
    classifications = data.Ph2SedReg.to_numpy()

teva_h.run_teva_model(data, classifications, wrk_dir, out_suffix=suffix)