store = {}
store['args']={'dataset': 'DatasetEnum.repeated_mnist_w_noise5', 'num_inference_samples': 10, 'available_sample_k': 10, 'seed': 607771, 'type': 'AcquisitionFunction.random', 'acquisition_method': 'AcquisitionMethod.independent', 'experiment_description': 'RMNIST with noise k10 b5 and b10 (and k100 b10), BALD, BatchBALD and heuristic', 'initial_samples': [38043, 40091, 17418, 2094, 39879, 3133, 5011, 40683, 54379, 24287, 9849, 59305, 39508, 39356, 8758, 52579, 13655, 7636, 21562, 41329], 'batch_size': 64, 'scoring_batch_size': 512, 'test_batch_size': 512, 'validation_set_size': 3072, 'early_stopping_patience': 3, 'epochs': 40, 'epoch_samples': 5056, 'target_accuracy': 1.0, 'target_num_acquired_samples': 300, 'initial_percentage': 100, 'reduce_percentage': 0, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 100, 'log_interval': 20, 'experiment_task_id': 'repeated_mnist_w_noise5_independent_random_k10_b10_607771', 'experiments_laaos': 'experiment_configs/rmnist_w_noise_2_5/configs.py', 'no_cuda': False, 'quickquick': False, 'initial_samples_per_class': 2, 'balanced_validation_set': False, 'balanced_test_set': False}
store['cmdline']=['./src/ignite_mnist.py', '--experiment_task_id=repeated_mnist_w_noise5_independent_random_k10_b10_607771', '--experiments_laaos=experiment_configs/rmnist_w_noise_2_5/configs.py']
store['iterations']=[]
store['initial_samples']=[38043, 40091, 17418, 2094, 39879, 3133, 5011, 40683, 54379, 24287, 9849, 59305, 39508, 39356, 8758, 52579, 13655, 7636, 21562, 41329]
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6379, 'nll': 1.557040020347085}, 'chosen_targets': [0, 8, 5, 3, 9, 3, 0, 3, 0, 6], 'chosen_samples': [258112, 177352, 231144, 18612, 219889, 245127, 94356, 130734, 15491, 99742], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.859634599008132, 'batch_acquisition_elapsed_time': 0.0039485500019509345})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.71, 'nll': 1.1990432336910168}, 'chosen_targets': [4, 3, 3, 3, 4, 9, 4, 6, 4, 3], 'chosen_samples': [61744, 244681, 163730, 292227, 252075, 249304, 111155, 227916, 46384, 241551], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.109813215996837, 'batch_acquisition_elapsed_time': 0.003935351007385179})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.7193, 'nll': 1.4139015455638018}, 'chosen_targets': [1, 4, 5, 7, 0, 2, 4, 8, 5, 0], 'chosen_samples': [259968, 183708, 206538, 285428, 227275, 227228, 213926, 228559, 202506, 227470], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.071777827979531, 'batch_acquisition_elapsed_time': 0.003941363014746457})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.7549, 'nll': 1.163339855141039}, 'chosen_targets': [9, 2, 0, 7, 6, 7, 4, 3, 4, 7], 'chosen_samples': [5836, 82021, 269925, 173725, 32241, 183494, 251256, 11511, 161173, 116112], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.141092186997412, 'batch_acquisition_elapsed_time': 0.003828165994491428})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.7864, 'nll': 0.7435043461896199}, 'chosen_targets': [0, 2, 6, 6, 4, 0, 4, 1, 5, 1], 'chosen_samples': [197838, 169976, 265076, 21821, 279341, 222801, 268963, 113001, 204332, 10982], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.088052284001606, 'batch_acquisition_elapsed_time': 0.003939682996133342})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8063, 'nll': 0.6467893002466992}, 'chosen_targets': [5, 1, 5, 2, 2, 4, 8, 7, 6, 8], 'chosen_samples': [127165, 181003, 51097, 151740, 274136, 199032, 113208, 17183, 273341, 113203], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.042597894003848, 'batch_acquisition_elapsed_time': 0.0038794629799667746})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.8601, 'nll': 0.5403373616766376}, 'chosen_targets': [9, 9, 3, 4, 2, 5, 6, 5, 3, 7], 'chosen_samples': [188847, 127251, 27467, 228365, 120109, 276609, 225274, 126027, 34525, 30538], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 14.120625258015934, 'batch_acquisition_elapsed_time': 0.003942301002098247})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8437, 'nll': 0.5759048433153123}, 'chosen_targets': [4, 9, 3, 7, 7, 7, 0, 3, 0, 5], 'chosen_samples': [272262, 83989, 135350, 166928, 116356, 251623, 56606, 114724, 244738, 35062], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.116589754004963, 'batch_acquisition_elapsed_time': 0.0039695799932815135})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.8408, 'nll': 0.596938250925289}, 'chosen_targets': [2, 9, 5, 6, 3, 9, 7, 9, 5, 0], 'chosen_samples': [150769, 41613, 6374, 257814, 62941, 225142, 192408, 159613, 238653, 254629], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.133115595002891, 'batch_acquisition_elapsed_time': 0.0039332279993686825})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.8585, 'nll': 0.5377154256031171}, 'chosen_targets': [7, 2, 9, 1, 1, 9, 6, 7, 5, 0], 'chosen_samples': [22467, 25807, 24082, 123644, 250562, 56690, 132744, 241825, 1575, 18212], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.0745912650018, 'batch_acquisition_elapsed_time': 0.0039042540010996163})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.8619, 'nll': 0.5303956850129021}, 'chosen_targets': [9, 9, 5, 5, 1, 2, 1, 0, 4, 9], 'chosen_samples': [103888, 57281, 159468, 65456, 72434, 176335, 172777, 6681, 276630, 12431], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 14.061592153011588, 'batch_acquisition_elapsed_time': 0.0038923360116314143})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.858, 'nll': 0.5594863094458374}, 'chosen_targets': [7, 3, 6, 2, 2, 4, 3, 4, 5, 6], 'chosen_samples': [191894, 188405, 22425, 146031, 115399, 164530, 149232, 14788, 165602, 265424], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 14.101598642999306, 'batch_acquisition_elapsed_time': 0.0038336620200425386})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.866, 'nll': 0.48691603756367474}, 'chosen_targets': [4, 8, 9, 7, 7, 0, 8, 5, 1, 2], 'chosen_samples': [233307, 63976, 107323, 253188, 68044, 161296, 216747, 95766, 259323, 47922], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.054983948008157, 'batch_acquisition_elapsed_time': 0.003941666014725342})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8705, 'nll': 0.4382671853804579}, 'chosen_targets': [8, 4, 9, 5, 1, 1, 4, 7, 7, 1], 'chosen_samples': [110625, 27351, 112638, 146574, 268969, 39169, 188854, 221297, 226132, 15290], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.001032080996083, 'batch_acquisition_elapsed_time': 0.004132371017476544})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8786, 'nll': 0.4687735929957122}, 'chosen_targets': [5, 0, 0, 2, 8, 0, 5, 1, 8, 3], 'chosen_samples': [121265, 119165, 131577, 236540, 293191, 95473, 18103, 68080, 39135, 246612], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.035332398983883, 'batch_acquisition_elapsed_time': 0.003916159010259435})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.8773, 'nll': 0.5395653607698471}, 'chosen_targets': [1, 5, 4, 2, 6, 5, 7, 9, 9, 5], 'chosen_samples': [84405, 239846, 129821, 31947, 113148, 89066, 267498, 73395, 289226, 16116], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.108431330998428, 'batch_acquisition_elapsed_time': 0.0035980540269520134})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8876, 'nll': 0.4368012147877582}, 'chosen_targets': [8, 9, 3, 1, 8, 0, 8, 5, 3, 8], 'chosen_samples': [128686, 237796, 134420, 231564, 148144, 276107, 37106, 113466, 55440, 3600], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.188274883985287, 'batch_acquisition_elapsed_time': 0.00400188600178808})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.8894, 'nll': 0.47061182372706345}, 'chosen_targets': [3, 7, 2, 0, 8, 6, 1, 1, 0, 7], 'chosen_samples': [295926, 232110, 90975, 272951, 252085, 186453, 37881, 22689, 120596, 80909], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.26417208899511, 'batch_acquisition_elapsed_time': 0.0039637690060772})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.8937, 'nll': 0.4282800903404982}, 'chosen_targets': [6, 3, 1, 5, 6, 2, 3, 3, 6, 2], 'chosen_samples': [234188, 88803, 78656, 1104, 78330, 230676, 111373, 173457, 8870, 43116], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.232714957994176, 'batch_acquisition_elapsed_time': 0.0038712839887011796})
store['iterations'].append({'num_epochs': 17, 'test_metrics': {'accuracy': 0.9062, 'nll': 0.39777402037735254}, 'chosen_targets': [5, 2, 0, 8, 8, 7, 3, 8, 1, 9], 'chosen_samples': [252157, 21537, 49311, 164436, 32553, 256197, 258826, 136893, 30667, 165888], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 18.49411412299378, 'batch_acquisition_elapsed_time': 0.0038767679943703115})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.9129, 'nll': 0.35113081556930975}, 'chosen_targets': [7, 1, 2, 4, 7, 2, 9, 1, 6, 5], 'chosen_samples': [91125, 2907, 2866, 96877, 249911, 147736, 221783, 6083, 74405, 126772], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.32378989298013, 'batch_acquisition_elapsed_time': 0.003906895988620818})
store['iterations'].append({'num_epochs': 16, 'test_metrics': {'accuracy': 0.9134, 'nll': 0.3493182778667461}, 'chosen_targets': [4, 2, 9, 1, 5, 6, 7, 3, 7, 4], 'chosen_samples': [144758, 185890, 43094, 129487, 141609, 151411, 143246, 121994, 231384, 166], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 17.539467337017413, 'batch_acquisition_elapsed_time': 0.003750815987586975})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.911, 'nll': 0.3312473833663197}, 'chosen_targets': [4, 1, 8, 3, 5, 9, 8, 2, 8, 0], 'chosen_samples': [105198, 206717, 146804, 188395, 121328, 5745, 212225, 154559, 61893, 266438], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 14.443544814013876, 'batch_acquisition_elapsed_time': 0.003984256007242948})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9004, 'nll': 0.3555595190553297}, 'chosen_targets': [5, 1, 5, 0, 9, 1, 6, 6, 7, 6], 'chosen_samples': [72697, 276356, 170223, 17064, 86064, 18963, 201126, 4090, 59611, 184572], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.201150440989295, 'batch_acquisition_elapsed_time': 0.0039036350208334625})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.9073, 'nll': 0.35533971630203043}, 'chosen_targets': [4, 0, 6, 2, 6, 3, 4, 7, 4, 0], 'chosen_samples': [187854, 168479, 146563, 249271, 54345, 140797, 289665, 102164, 111860, 69035], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.500433358014561, 'batch_acquisition_elapsed_time': 0.003887103986926377})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9074, 'nll': 0.33280168864289844}, 'chosen_targets': [2, 8, 1, 3, 6, 9, 0, 1, 2, 9], 'chosen_samples': [38326, 7712, 171908, 110717, 279154, 66155, 146040, 37869, 203025, 127946], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.307837583997753, 'batch_acquisition_elapsed_time': 0.003956561005907133})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9192, 'nll': 0.29423732334328373}, 'chosen_targets': [7, 6, 6, 8, 0, 1, 7, 1, 5, 6], 'chosen_samples': [206274, 2756, 47611, 13176, 65272, 56081, 167487, 287338, 216193, 68305], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.307420861994615, 'batch_acquisition_elapsed_time': 0.0039610230014659464})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.9219, 'nll': 0.29263256402525434}, 'chosen_targets': [7, 4, 1, 6, 1, 2, 9, 8, 9, 0], 'chosen_samples': [156531, 254592, 104542, 100751, 218039, 98354, 3991, 96696, 198475, 73583], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.598033974994905, 'batch_acquisition_elapsed_time': 0.003990361001342535})
store['iterations'].append({'num_epochs': 15, 'test_metrics': {'accuracy': 0.9238, 'nll': 0.2917025301849893}, 'chosen_targets': [5, 7, 0, 0, 6, 5, 3, 8, 2, 3], 'chosen_samples': [275867, 66280, 218946, 80283, 274621, 54233, 90304, 142079, 243930, 101537], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 16.75900800700765, 'batch_acquisition_elapsed_time': 0.003933383006369695})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9182, 'nll': 0.2860288672322094}, 'chosen_targets': [0, 7, 9, 5, 6, 7, 2, 5, 1, 3], 'chosen_samples': [69934, 200765, 112989, 10884, 282856, 246176, 134550, 92343, 19094, 241583], 'chosen_samples_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.34872957400512, 'batch_acquisition_elapsed_time': 0.0038854239974170923})
