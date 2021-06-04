store = {}
store['args']={'num_inference_samples': 10, 'available_sample_k': 10, 'seed': 211889, 'type': 'AcquisitionFunction.bald', 'acquisition_method': 'AcquisitionMethod.multibald', 'experiment_description': 'RMNIST with noise k10 b5 and b10 (and k100 b10), BALD, BatchBALD and heuristic', 'initial_samples': [38043, 40091, 17418, 2094, 39879, 3133, 5011, 40683, 54379, 24287, 9849, 59305, 39508, 39356, 8758, 52579, 13655, 7636, 21562, 41329], 'batch_size': 64, 'scoring_batch_size': 512, 'test_batch_size': 512, 'validation_set_size': 3072, 'early_stopping_patience': 3, 'epochs': 40, 'epoch_samples': 5056, 'target_accuracy': 0.95, 'target_num_acquired_samples': 300, 'initial_percentage': 100, 'reduce_percentage': 0, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 100, 'log_interval': 20, 'dataset': 'DatasetEnum.repeated_mnist_w_noise', 'experiment_task_id': 'rmnist_w_noise_multibald_bald_k10_b10_211889', 'experiments_laaos': './experiment_configs/rmnist_w_noise/configs.py', 'no_cuda': False, 'quickquick': False, 'initial_samples_per_class': 2}
store['cmdline']=['./src/ignite_mnist.py', '--experiment_task_id=rmnist_w_noise_multibald_bald_k10_b10_211889', '--experiments_laaos=./experiment_configs/rmnist_w_noise/configs.py']
store['iterations']=[]
store['initial_samples']=[38043, 40091, 17418, 2094, 39879, 3133, 5011, 40683, 54379, 24287, 9849, 59305, 39508, 39356, 8758, 52579, 13655, 7636, 21562, 41329]
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.6281, 'nll': 2.667152810352445}, 'chosen_samples': [138490, 166864, 85359, 163636, 100527, 140059, 41714, 63727, 145359, 121974], 'chosen_samples_score': [1.2237067352916462, 1.9394860675396413, 2.1568936300298427, 2.2406586519379568, 2.2758190075855147, 2.300248227872242, 2.2848848239037776, 2.2805315432889097, 2.264574911442928, 2.3186731727447807], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 14.362575256000127, 'batch_acquisition_elapsed_time': 79.76577000999896})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.6823, 'nll': 2.0552410743260383}, 'chosen_samples': [24077, 145907, 134095, 154010, 80794, 72802, 84600, 35930, 144600, 171838], 'chosen_samples_score': [1.1639320951749483, 1.79906228762262, 2.0911454165540797, 2.215111645147662, 2.265884690488738, 2.2793066692039, 2.270999776436917, 2.30377686253533, 2.296250392836823, 2.3089554628701077], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.644733010998607, 'batch_acquisition_elapsed_time': 79.43380693200015})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.6918, 'nll': 1.8783342500901223}, 'chosen_samples': [50066, 114709, 147610, 54709, 56707, 55244, 28944, 14177, 96251, 69082], 'chosen_samples_score': [1.3338295512276757, 2.065920510261636, 2.2027079110946977, 2.266762699944308, 2.2894635994655816, 2.272312127196901, 2.2929905050393415, 2.322072712542988, 2.2817837073660434, 2.2703681094470385], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.205276003001927, 'batch_acquisition_elapsed_time': 79.45042525200188})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7675, 'nll': 1.2461322005474569}, 'chosen_samples': [168356, 29338, 156783, 65000, 132196, 118476, 96329, 128676, 148305, 89483], 'chosen_samples_score': [0.9423207750014412, 1.6003401191022149, 1.96579385093617, 2.1562997497182916, 2.2332375136783593, 2.2850249561743383, 2.299019175622372, 2.3088194354135396, 2.310454596356581, 2.2581720171546555], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.947648036002647, 'batch_acquisition_elapsed_time': 80.44585081900004})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7509, 'nll': 1.317851260921955}, 'chosen_samples': [21372, 4873, 167214, 9517, 72633, 153224, 32693, 107260, 150332, 12877], 'chosen_samples_score': [1.1007041315371047, 1.7766425869499678, 2.0605085684735025, 2.1897885654463503, 2.246264341839008, 2.2993187804421567, 2.292768198598687, 2.283559783504102, 2.3110830675863827, 2.3072268743435256], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.885589358000288, 'batch_acquisition_elapsed_time': 79.3100225820017})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7827, 'nll': 1.1271603784585}, 'chosen_samples': [124761, 96793, 83738, 56695, 79592, 46356, 54592, 23642, 97069, 171180], 'chosen_samples_score': [0.8064973458748774, 1.3968226125452465, 1.7902792948925066, 1.9937879313009153, 2.125568494658795, 2.1993202657127258, 2.220926707848629, 2.301112747633124, 2.284967894137739, 2.2845862319868937], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.105459856997186, 'batch_acquisition_elapsed_time': 80.06919276299959})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8306, 'nll': 0.8946821364033223}, 'chosen_samples': [54880, 55905, 113504, 107365, 76951, 115905, 83152, 3016, 100466, 107068], 'chosen_samples_score': [0.9089309199180253, 1.538053467117466, 1.8620585554552402, 2.0436899656687135, 2.147991092113646, 2.1868190743877083, 2.2520688916222404, 2.29368499117183, 2.292614495627576, 2.3201501892066974], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.153657658000157, 'batch_acquisition_elapsed_time': 79.8602105050013})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8521, 'nll': 0.9075730348825455}, 'chosen_samples': [65728, 168038, 61812, 7745, 75494, 19344, 19524, 94815, 123644, 173019], 'chosen_samples_score': [1.2056266020034263, 1.835356592792694, 2.1477032672510954, 2.2561928297528144, 2.2838530844917475, 2.296382042459392, 2.2823791636086095, 2.295571028419473, 2.269502002476688, 2.3549488553362132], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.39334506600062, 'batch_acquisition_elapsed_time': 79.79191299399827})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8745, 'nll': 0.751822487926483}, 'chosen_samples': [157171, 117523, 173378, 123056, 24990, 30962, 107249, 96072, 129727, 46832], 'chosen_samples_score': [1.0562634157518234, 1.7724618514230637, 2.0845897912722102, 2.225501749700577, 2.2749910803038187, 2.2767774401883982, 2.3240407452909104, 2.3292800802338007, 2.2649556499425554, 2.3193280546292048], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.280388101000426, 'batch_acquisition_elapsed_time': 79.80822867199822})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.841, 'nll': 0.8409254249823093}, 'chosen_samples': [2748, 146072, 179468, 33401, 114117, 133705, 145538, 826, 37567, 145917], 'chosen_samples_score': [0.9564104624669192, 1.5651629479253268, 1.9358061328610212, 2.1181338562571845, 2.199545582838442, 2.229058508581793, 2.2481936783679988, 2.266494047129929, 2.300091032390074, 2.3138028583601598], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.831783448000351, 'batch_acquisition_elapsed_time': 79.46964326399757})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.885, 'nll': 0.6926374739313126}, 'chosen_samples': [46440, 101602, 152622, 142686, 62933, 37048, 10265, 166440, 102384, 82686], 'chosen_samples_score': [1.0850199354171453, 1.827162438078496, 2.0908722004914413, 2.2117606738888433, 2.265679440998272, 2.277508082838425, 2.320973514519931, 2.3028712303870815, 2.3167748773397334, 2.3039836507569964], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 14.218465330999607, 'batch_acquisition_elapsed_time': 80.13823173399942})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8919, 'nll': 0.6073846378529072}, 'chosen_samples': [160905, 17710, 74260, 14825, 148512, 136084, 97758, 76084, 60306, 136025], 'chosen_samples_score': [1.092636461704068, 1.7557659778804011, 2.0520590313924862, 2.1960140011658615, 2.2575717917552396, 2.2885869677239885, 2.284155332972416, 2.305087842478489, 2.2406009571636645, 2.302719896267579], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 13.045731864000118, 'batch_acquisition_elapsed_time': 79.20329746700008})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.911, 'nll': 0.5701467102491855}, 'chosen_samples': [129440, 174994, 11621, 137958, 123370, 12702, 141040, 60693, 132157, 113844], 'chosen_samples_score': [1.0423877617926312, 1.8106542623701185, 2.1035323755181383, 2.2166684171011353, 2.2654670080016936, 2.265660842224779, 2.2888827028886594, 2.3096436333178767, 2.265064864212711, 2.331979997292618], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 16.295282996001333, 'batch_acquisition_elapsed_time': 79.17086125499918})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.9049, 'nll': 0.5915434142816067}, 'chosen_samples': [159429, 6466, 149132, 77592, 7322, 78003, 9118, 72985, 77048, 66466], 'chosen_samples_score': [1.0041686692800544, 1.6748892339250891, 1.9931720528975698, 2.1593392573382206, 2.230105304740222, 2.2497653594067106, 2.256097173163921, 2.2865596209955124, 2.2752000225267226, 2.2988937247325945], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.04143228400062, 'batch_acquisition_elapsed_time': 80.49982474399803})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.8846, 'nll': 0.6726955378186703}, 'chosen_samples': [71721, 103224, 166081, 131657, 156746, 63719, 122633, 137603, 72179, 175774], 'chosen_samples_score': [1.1510902706575106, 1.8244664045179506, 2.1621435024257254, 2.247480936630951, 2.279202179497841, 2.2780914188411034, 2.299347791891576, 2.3443605293566243, 2.3245033582967083, 2.2760672897912073], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 16.922908303997247, 'batch_acquisition_elapsed_time': 79.19818810099969})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8975, 'nll': 0.5777497076296807}, 'chosen_samples': [37588, 59747, 94520, 162746, 38760, 61129, 86184, 150646, 154520, 50908], 'chosen_samples_score': [0.8889801024807439, 1.5027926018874527, 1.8797426566020166, 2.0734306340072157, 2.180175305321236, 2.239769844248744, 2.2531471548780706, 2.278569856568061, 2.2559813808762277, 2.2951573770053493], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.86086696199709, 'batch_acquisition_elapsed_time': 79.38262877500165})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.91, 'nll': 0.5550371807312966}, 'chosen_samples': [93594, 109545, 43176, 5402, 91308, 157373, 76628, 162437, 131208, 60440], 'chosen_samples_score': [1.2137546445709217, 1.8837222704676284, 2.169641177618707, 2.2489067987986697, 2.2845274777602693, 2.314059488797521, 2.2554891558353805, 2.2926591748147223, 2.3199015315930476, 2.2446407459614566], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 19.30743659899963, 'batch_acquisition_elapsed_time': 79.74926108699947})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8998, 'nll': 0.5635808712989093}, 'chosen_samples': [34765, 109890, 36268, 17057, 37347, 169890, 48507, 84637, 42237, 11645], 'chosen_samples_score': [1.000513840652875, 1.6374540419410923, 1.972135349260352, 2.129636434502877, 2.211174819257974, 2.272972985038067, 2.2398112223324906, 2.2903655379504437, 2.3229601386662493, 2.2961783497147215], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.489548253000976, 'batch_acquisition_elapsed_time': 78.75219136300075})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9225, 'nll': 0.5432313810396194}, 'chosen_samples': [117728, 99818, 155946, 58459, 94406, 37602, 107220, 126905, 156174, 179314], 'chosen_samples_score': [1.262653917989749, 1.9789981763957252, 2.221931736925254, 2.2743541845796074, 2.2939423472251095, 2.295908034418895, 2.2873865684690444, 2.281638835011382, 2.3129708981305583, 2.2986708425880327], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 24.67579895900053, 'batch_acquisition_elapsed_time': 79.7214787289995})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9247, 'nll': 0.47123815119981766}, 'chosen_samples': [76158, 102317, 26412, 118322, 100390, 68532, 164870, 139188, 50320, 58237], 'chosen_samples_score': [1.0393730126238632, 1.7264806702628868, 2.0558883949508937, 2.195053560973226, 2.2605848647542333, 2.2784192339417793, 2.2612105721287667, 2.3254587910109183, 2.3313063356704706, 2.314173666433768], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 17.25416951200168, 'batch_acquisition_elapsed_time': 79.1524147400014})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.9305, 'nll': 0.45643577792465684}, 'chosen_samples': [66474, 96337, 112294, 148674, 76196, 175274, 167076, 148886, 92911, 165502], 'chosen_samples_score': [1.0890110537560562, 1.7647473909403946, 2.0561947019285016, 2.184724479489868, 2.249267380877522, 2.2949462220830705, 2.2744171992014435, 2.2837382794127974, 2.2918451806011664, 2.312422688173686], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.215807913002209, 'batch_acquisition_elapsed_time': 78.99850881100065})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9406, 'nll': 0.42234296903848645}, 'chosen_samples': [125155, 94597, 171986, 138031, 80206, 119361, 162112, 139538, 169895, 102178], 'chosen_samples_score': [1.0842016087577047, 1.756570949215296, 2.0967147395331653, 2.2238674741359397, 2.2691852378840625, 2.2699762751217003, 2.2812350909620465, 2.300484211495493, 2.2857684441109063, 2.283045059931724], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 19.51488365700061, 'batch_acquisition_elapsed_time': 79.41564406500038})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9476, 'nll': 0.37864014720678335}, 'chosen_samples': [31738, 101349, 137079, 88633, 64822, 72934, 128932, 39668, 166313, 1374], 'chosen_samples_score': [1.1397744948905815, 1.8467846164702173, 2.148959935388045, 2.2479969284181944, 2.2839449970705736, 2.2744411275462415, 2.310438648148107, 2.307172841787411, 2.2870753020544337, 2.2846670501708237], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 22.35394309700132, 'batch_acquisition_elapsed_time': 79.16294133699921})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9462, 'nll': 0.3986832028853893}, 'chosen_samples': [51759, 172048, 11482, 27503, 113156, 95864, 28102, 10373, 36281, 121642], 'chosen_samples_score': [1.0687377695703046, 1.7138680415889573, 2.095229381905355, 2.2185859295070762, 2.2721985655424013, 2.3051289316875545, 2.301670007333075, 2.270949589222017, 2.3068355937719005, 2.281058260193702], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 21.78312887300126, 'batch_acquisition_elapsed_time': 78.66168170600213})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9415, 'nll': 0.3790403688943386}, 'chosen_samples': [124955, 109916, 157256, 21880, 74765, 51993, 75190, 19934, 28783, 94122], 'chosen_samples_score': [1.0203247933175374, 1.6964376801957526, 2.0516419516102475, 2.212204435189273, 2.2683320100517586, 2.307523499190662, 2.3061638968379143, 2.2755731008440208, 2.2942606178712674, 2.3270919832619947], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 23.901664114000596, 'batch_acquisition_elapsed_time': 79.79708292799842})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9503, 'nll': 0.35803151987075804}, 'chosen_samples': [149839, 168762, 48360, 120080, 131572, 89839, 98246, 46368, 74749, 72066], 'chosen_samples_score': [1.0886745872325743, 1.7911882418136111, 2.0805860452175304, 2.195827420028092, 2.2533608915728816, 2.271105235368424, 2.287672806343203, 2.2711797345113425, 2.336509636074771, 2.2971684020289898], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 17.62798145600027, 'batch_acquisition_elapsed_time': 79.71402885499992})
