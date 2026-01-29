import teva
import logging
import pandas as pd

# Data
wrk_dir = '/root/repositories/sedregfromedz/data/teva/'
suffix = '_edzsonly'
data = pd.read_csv(wrk_dir+'Dataset_lat_conf_TEVA' + suffix + '.csv')  # vertical confinement has all of the input data, too; just need to select the different classifications
data.pop('lat_conf') # still pop this column off the dataframe
classifications = data.pop('Ph2SedReg')
sheets = ['CCEA_' + str(x) for x in classifications.unique()]

# output directory
out_dir = "/root/repositories/sedregfromedz/data/teva/cc_loops/"

n_loops = 20

# list of input featuers
input_features_list = data.iloc[:, 7:].columns.tolist()

# reformat the data
observation_table = data[input_features_list].to_numpy()

# Other variables
n_observations = classifications.shape[0]
n_features = len(input_features_list)
visualize = False
output_logging_level = logging.INFO

cc_max_order = 4  # n_features
n_cc_gens = 25
use_sensitivity = True


# Algorithm
teva_alg = teva.TEVA(ccea_max_order                     =cc_max_order, # n_features,
                     ccea_offspring_per_gen             =n_features,
                     ccea_num_new_pop                   =n_features,
                     ccea_total_generations             =n_cc_gens,
                     ccea_n_age_layers                  =5,
                    #  ccea_max_novel_order               =4,
                     ccea_gen_per_growth                =3,
                     ccea_layer_size                    =n_features,
                     ccea_archive_offspring_per_gen     =25,
                     ccea_p_crossover                   =0.5,
                     ccea_p_wildcard                    =0.75,
                     ccea_p_mutation                    =1 / n_features,
                     ccea_tournament_size               =3,
                     ccea_selective_mutation            =False,
                     ccea_use_sensitivity               =use_sensitivity,
                     ccea_sensitivity_threshold         =0,
                     ccea_selection_exponent            =5,
                     ccea_fitness_threshold             =1 / n_observations,
                     ccea_archive_bin_size              =20,

                     dnfea_total_generations            =0,
                     dnfea_gen_per_growth               =3,
                     dnfea_n_age_layers                 =5,
                     dnfea_offspring_per_gen            =20,
                     dnfea_p_crossover                  =0.5,
                     dnfea_p_targeted_mutation          =0.2,
                     dnfea_p_targeted_crossover         =0.25,
                     dnfea_tournament_size              =3,
                     dnfea_p_union                      =0.5,
                     dnfea_p_intersection               =0.0,
                     dnfea_selection_exponent           =5,
                     dnfea_max_order                    =0,
                     dnfea_layer_size                   =20)



# Run the algorithm for the data set
unique_classes = teva_alg.fit(observation_table=observation_table,
                              classifications=classifications)

for i in range(n_loops):
    print(f"LOOP {i}")
    teva_alg.run_all_targets(logfile_logging_level=logging.INFO,
                            output_logging_level=output_logging_level,
                            visualize=visualize)

    print(teva_alg._ccea.get_all_archive_values())

    cc_name = out_dir + f"cc_{i}.xlsx"
    teva_alg.export(cc_name, "")
    # TEVA export doesn't preserve feature names in the CC output file, for some reason
    # each feature column is labelled like "feature_0"
    # manually resave excel files with the correct feature names
    # they start in column 14
    tabs = [pd.read_excel(cc_name, sheet_name=x) for x in sheets ]
    for tab in tabs:
        try:
            tab.columns = tab.columns[:14].to_list() + input_features_list
            tab.drop(columns=tab.columns[0], inplace=True)  # drop the first column; just an index with no column name; confuses the teva output explorer
        except:
            continue
    with pd.ExcelWriter(cc_name) as writer:
        for tab,sheet in zip(tabs,sheets):
            tab.to_excel(writer, sheet_name=sheet)

cc_names = [out_dir + f'cc_{i}.xlsx' for i in range(n_loops)]
outs = [pd.DataFrame() for x in sheets]

for name in cc_names:
    for n,sheet in enumerate(sheets):
        outs[n] = pd.concat([outs[n], pd.read_excel(name, sheet_name=sheet)])

# drop first column
for out in outs:
    try:
        out.drop(columns=out.columns[0], inplace=True)
        out.index = range(out.shape[0])
    except:
        continue

with pd.ExcelWriter(out_dir+'cc_all.xlsx') as writer:
    for out,sheet in zip(outs, sheets):
        try:
            out.to_excel(writer, sheet_name=sheet)
        except:
            continue
