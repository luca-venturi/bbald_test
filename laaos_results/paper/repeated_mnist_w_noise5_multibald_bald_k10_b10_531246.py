store = {}
store['args']={'dataset': 'DatasetEnum.repeated_mnist_w_noise5', 'num_inference_samples': 10, 'available_sample_k': 10, 'seed': 531246, 'type': 'AcquisitionFunction.bald', 'acquisition_method': 'AcquisitionMethod.multibald', 'experiment_description': 'RMNIST with noise k10 b5 and b10 (and k100 b10), BALD, BatchBALD and heuristic', 'initial_samples': [38043, 40091, 17418, 2094, 39879, 3133, 5011, 40683, 54379, 24287, 9849, 59305, 39508, 39356, 8758, 52579, 13655, 7636, 21562, 41329], 'batch_size': 64, 'scoring_batch_size': 512, 'test_batch_size': 512, 'validation_set_size': 3072, 'early_stopping_patience': 3, 'epochs': 40, 'epoch_samples': 5056, 'target_accuracy': 1.0, 'target_num_acquired_samples': 300, 'initial_percentage': 100, 'reduce_percentage': 0, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 100, 'log_interval': 20, 'experiment_task_id': 'repeated_mnist_w_noise5_multibald_bald_k10_b10_531246', 'experiments_laaos': 'experiment_configs/rmnist_w_noise_2_5/configs.py', 'no_cuda': False, 'quickquick': False, 'initial_samples_per_class': 2, 'balanced_validation_set': False, 'balanced_test_set': False}
store['cmdline']=['./src/ignite_mnist.py', '--experiment_task_id=repeated_mnist_w_noise5_multibald_bald_k10_b10_531246', '--experiments_laaos=experiment_configs/rmnist_w_noise_2_5/configs.py']
store['iterations']=[]
store['initial_samples']=[38043, 40091, 17418, 2094, 39879, 3133, 5011, 40683, 54379, 24287, 9849, 59305, 39508, 39356, 8758, 52579, 13655, 7636, 21562, 41329]
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6617, 'nll': 1.643267865650549}, 'chosen_targets': [8, 2, 9, 8, 3, 8, 2, 2, 3, 5], 'chosen_samples': [129608, 34558, 55496, 107597, 85538, 65205, 233056, 204722, 24271, 239669], 'chosen_samples_score': [1.4942304850785009, 2.17240184379246, 2.2590753930654666, 2.288217840882094, 2.2982142560081225, 2.3014168177141783, 2.297429760334485, 2.2969674937136046, 2.306640045647091, 2.3046441948798693], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.998010331008118, 'batch_acquisition_elapsed_time': 1262.8398697419907})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.7517, 'nll': 0.889994805225865}, 'chosen_targets': [5, 5, 7, 9, 5, 9, 7, 8, 4, 4], 'chosen_samples': [127354, 95628, 8214, 237670, 227629, 24468, 62673, 175619, 84295, 139922], 'chosen_samples_score': [1.3362508317576238, 2.00670540236025, 2.2110188752554216, 2.2708992186177426, 2.2904798924134466, 2.298386526338729, 2.2982371641837576, 2.296173870878682, 2.304920650439805, 2.2911484917837264], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.247836191003444, 'batch_acquisition_elapsed_time': 1261.9722397320147})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.8128, 'nll': 0.6363868523393788}, 'chosen_targets': [3, 3, 3, 5, 7, 3, 8, 9, 7, 7], 'chosen_samples': [65728, 189516, 167496, 200853, 220701, 137467, 293872, 90712, 210188, 45626], 'chosen_samples_score': [1.3991801766143737, 2.1787756468764408, 2.2718851403941054, 2.2951346669939907, 2.3012974172832172, 2.30228184363043, 2.303672531971234, 2.304280313155073, 2.299924711184729, 2.2962278121854065], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.332810574007453, 'batch_acquisition_elapsed_time': 1261.415973160998})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.8071, 'nll': 0.6712691502069358}, 'chosen_targets': [8, 6, 0, 9, 1, 4, 0, 0, 4, 2], 'chosen_samples': [36299, 95753, 87328, 285800, 207793, 145234, 92693, 147328, 53873, 105094], 'chosen_samples_score': [1.3231287107809022, 1.9605545109009628, 2.1940816223592803, 2.275738025798672, 2.2948276822889335, 2.300099799140021, 2.3046116612783094, 2.3077417496456567, 2.3102924945331016, 2.3059930943078424], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.176118483010214, 'batch_acquisition_elapsed_time': 1261.8517319379898})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8445, 'nll': 0.5413711764429734}, 'chosen_targets': [5, 5, 4, 0, 3, 0, 6, 6, 2, 5], 'chosen_samples': [202180, 136072, 9867, 87082, 54858, 41027, 226580, 229403, 113138, 22180], 'chosen_samples_score': [1.2839275194248672, 2.01993896917335, 2.200632983777616, 2.2693177098609194, 2.291497336365097, 2.2991867770137437, 2.312172144189759, 2.2987161891448413, 2.313950207275901, 2.2985337562316905], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.213598397996975, 'batch_acquisition_elapsed_time': 1263.2857005039987})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8552, 'nll': 0.5035101278092594}, 'chosen_targets': [2, 7, 9, 8, 7, 5, 3, 9, 2, 7], 'chosen_samples': [150951, 114212, 192792, 99266, 96398, 75949, 125683, 238850, 126348, 275391], 'chosen_samples_score': [1.0957148411639492, 1.860543605260888, 2.109971327616436, 2.2234292253191605, 2.2665389573896504, 2.2863066806523387, 2.2960329142990825, 2.3087639675663083, 2.3060818245809047, 2.2945528239586563], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 6.1160006269929, 'batch_acquisition_elapsed_time': 1261.3540345120127})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8681, 'nll': 0.4615145798140586}, 'chosen_targets': [2, 3, 9, 7, 4, 6, 7, 2, 2, 4], 'chosen_samples': [135093, 132345, 162020, 191822, 207473, 95566, 121019, 106163, 255093, 176551], 'chosen_samples_score': [1.1240957123133213, 1.8046767445875682, 2.08488276273299, 2.213470508050397, 2.2663479932929, 2.288459756387393, 2.305588027747868, 2.2858588554454107, 2.3103847971082097, 2.3050819462634964], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 6.201928429014515, 'batch_acquisition_elapsed_time': 1262.2483134130016})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8807, 'nll': 0.4128877888317774}, 'chosen_targets': [4, 2, 8, 6, 9, 9, 3, 4, 2, 9], 'chosen_samples': [17406, 83138, 43008, 115314, 10970, 230698, 162758, 106623, 179390, 31919], 'chosen_samples_score': [1.1860876565028589, 1.9137665679862657, 2.2027944314052417, 2.26987843317608, 2.2886348500862734, 2.2975517613358187, 2.29777887867896, 2.2987201536759096, 2.322303930491527, 2.2946887659732376], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.154143607011065, 'batch_acquisition_elapsed_time': 1262.3399009629793})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.8992, 'nll': 0.3454732467408628}, 'chosen_targets': [0, 3, 4, 2, 0, 4, 3, 7, 3, 2], 'chosen_samples': [182765, 169091, 134769, 120866, 9158, 272049, 134825, 200002, 278549, 87874], 'chosen_samples_score': [1.3989393746447074, 2.0330740010531665, 2.2219263837516414, 2.2836329817794336, 2.2976346995157644, 2.301376068837621, 2.2984411855524556, 2.3096500596236886, 2.3136657753986762, 2.295042925391577], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.336447291978402, 'batch_acquisition_elapsed_time': 1262.7760577489971})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8887, 'nll': 0.38030392020025466}, 'chosen_targets': [3, 5, 8, 5, 7, 3, 2, 5, 5, 5], 'chosen_samples': [112713, 218698, 19253, 199942, 143021, 132656, 53076, 38698, 79942, 289202], 'chosen_samples_score': [1.0181217501559403, 1.7330531017070656, 2.0895618858849865, 2.2088339108569595, 2.2652622951313712, 2.2886501035115154, 2.298154964567802, 2.313879293262903, 2.292033250908508, 2.301034238303152], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.106489682017127, 'batch_acquisition_elapsed_time': 1263.3640274749778})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9082, 'nll': 0.3148410831439792}, 'chosen_targets': [7, 2, 5, 9, 7, 5, 6, 2, 2, 7], 'chosen_samples': [229242, 78003, 267317, 167662, 250014, 259089, 265462, 198003, 87739, 229571], 'chosen_samples_score': [1.2600386945827668, 1.9775906643631285, 2.2082166467789883, 2.2712333198983234, 2.2916793275789824, 2.298767582806484, 2.3036091233289815, 2.3012869439428, 2.3088007321115067, 2.3087582141515535], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.21597705699969, 'batch_acquisition_elapsed_time': 1261.143316938018})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9169, 'nll': 0.3000733391539015}, 'chosen_targets': [0, 3, 9, 5, 0, 2, 3, 2, 8, 0], 'chosen_samples': [248289, 76748, 145624, 107741, 182817, 73374, 106379, 11711, 242942, 279363], 'chosen_samples_score': [1.1579159255467093, 1.8805770516373403, 2.1397519697286724, 2.240635688650652, 2.280977433677205, 2.2944553465085957, 2.3103856871418715, 2.299671211361171, 2.3075631232322786, 2.295424834769993], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.1966777209891, 'batch_acquisition_elapsed_time': 1261.984760951018})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.9103, 'nll': 0.3123989628187121}, 'chosen_targets': [8, 0, 7, 5, 4, 8, 4, 3, 8, 5], 'chosen_samples': [157324, 198042, 11534, 49890, 256572, 45082, 104342, 270077, 84457, 141210], 'chosen_samples_score': [1.283960868270272, 1.9968027589075665, 2.243364954120558, 2.2893696936003884, 2.2986942840511637, 2.3015702398265216, 2.300711034023339, 2.305220606179457, 2.3044555026115434, 2.2974215463703818], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.335168827034067, 'batch_acquisition_elapsed_time': 1261.599199737946})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9216, 'nll': 0.2808513789139428}, 'chosen_targets': [8, 5, 2, 7, 9, 9, 4, 3, 2, 4], 'chosen_samples': [249782, 299747, 200967, 248954, 84620, 193998, 73942, 178048, 232169, 74654], 'chosen_samples_score': [1.1317552513317013, 1.881386445559996, 2.142085582229638, 2.2474479565454972, 2.2861651098189757, 2.2970055555447715, 2.2967953864906896, 2.308947947929776, 2.300442595948068, 2.2995913492975486], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.239291958045214, 'batch_acquisition_elapsed_time': 1261.0646414479706})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9098, 'nll': 0.31177070208151497}, 'chosen_targets': [0, 2, 5, 8, 7, 6, 5, 0, 5, 9], 'chosen_samples': [158102, 68843, 261204, 19025, 198720, 125684, 137747, 168911, 167949, 17501], 'chosen_samples_score': [1.165886770244716, 2.074368236794281, 2.243234133633907, 2.2828382975053563, 2.295518337268318, 2.3000151679978984, 2.2982751556889456, 2.29382135082123, 2.2995241733964202, 2.3045436915696778], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.290472930995747, 'batch_acquisition_elapsed_time': 1260.6569877020083})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9149, 'nll': 0.30256528743645106}, 'chosen_targets': [0, 8, 5, 0, 3, 1, 8, 3, 3, 7], 'chosen_samples': [208102, 177736, 114928, 111464, 230417, 120224, 78674, 155401, 249180, 139642], 'chosen_samples_score': [1.0546582483188303, 1.7867377550247103, 2.0618207105943354, 2.215906573662072, 2.26045197903407, 2.281939436859458, 2.2874590742841967, 2.2946386819799693, 2.3019403838764605, 2.293745159493187], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.174724689044524, 'batch_acquisition_elapsed_time': 1262.4578425690415})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.9271, 'nll': 0.2590837873291416}, 'chosen_targets': [2, 2, 6, 6, 2, 4, 8, 2, 3, 0], 'chosen_samples': [263814, 63980, 25962, 167631, 155974, 185265, 230054, 35138, 113547, 209713], 'chosen_samples_score': [1.258715806516436, 2.0275018016281114, 2.2365240197334697, 2.2875861316361368, 2.2985575415787127, 2.301719934926828, 2.3001311277221155, 2.2945792965964955, 2.3086893163481355, 2.2957218747515964], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.460256531019695, 'batch_acquisition_elapsed_time': 1261.264898127003})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.9368, 'nll': 0.2264870619055996}, 'chosen_targets': [9, 0, 2, 2, 3, 0, 0, 1, 8, 2], 'chosen_samples': [189118, 14746, 271512, 219561, 268536, 230476, 109575, 197505, 67909, 191364], 'chosen_samples_score': [1.3139887091846358, 2.0148870808618695, 2.2213884802286104, 2.2793724831706346, 2.2963493368049495, 2.3007074955358875, 2.3070069653054346, 2.2958096566941224, 2.297773297135197, 2.2998411967714665], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.325479861989152, 'batch_acquisition_elapsed_time': 1261.8230415399885})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.9252, 'nll': 0.2613026233790665}, 'chosen_targets': [2, 5, 5, 6, 8, 9, 2, 3, 5, 9], 'chosen_samples': [268293, 259396, 98378, 114316, 162384, 10070, 22083, 14878, 146150, 13945], 'chosen_samples_score': [1.2224125352616912, 2.0629412923689596, 2.242398023189147, 2.2851217345602475, 2.2981227732782834, 2.301263545959125, 2.2973220395038307, 2.2969693329437675, 2.3067784947390333, 2.288223883451151], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.432497228961438, 'batch_acquisition_elapsed_time': 1260.5993504310027})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.936, 'nll': 0.26245979526467006}, 'chosen_targets': [7, 4, 2, 6, 3, 1, 4, 6, 9, 6], 'chosen_samples': [201700, 82531, 292358, 130956, 64762, 159818, 40334, 107288, 50562, 17382], 'chosen_samples_score': [1.2207553679957952, 1.8986698865844804, 2.1487253033101648, 2.2613655487845, 2.288460318669504, 2.2984533138301897, 2.302535055669421, 2.302118866064559, 2.3105176240569647, 2.295835611016286], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.196119688975159, 'batch_acquisition_elapsed_time': 1260.8388425880112})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9345, 'nll': 0.23320346698948302}, 'chosen_targets': [2, 7, 6, 4, 8, 1, 6, 4, 3, 6], 'chosen_samples': [14295, 4784, 174885, 91717, 145186, 281744, 248417, 13350, 37186, 229624], 'chosen_samples_score': [1.0935697275826337, 1.8560865477090438, 2.1354225522382118, 2.2497583934189445, 2.2840338576468397, 2.296992009016728, 2.3117795491735897, 2.2969291648068895, 2.296538938088633, 2.2951113854648053], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.286065964959562, 'batch_acquisition_elapsed_time': 1261.8333927170024})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9407, 'nll': 0.2137392024453956}, 'chosen_targets': [3, 6, 8, 3, 0, 8, 2, 2, 3, 9], 'chosen_samples': [91664, 255386, 74896, 102477, 272880, 173312, 185129, 106122, 271664, 217048], 'chosen_samples_score': [1.1928831336015127, 1.8382004767299134, 2.133704627933784, 2.2381573551840614, 2.278188640520618, 2.2932468507449526, 2.3095728610350705, 2.299684955903279, 2.3104064451763744, 2.2963900637921197], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.3111013029702, 'batch_acquisition_elapsed_time': 1261.4642091660062})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9369, 'nll': 0.23834368850958232}, 'chosen_targets': [5, 7, 8, 0, 5, 5, 7, 2, 6, 9], 'chosen_samples': [210896, 236274, 49525, 109889, 22272, 170320, 222573, 51986, 1918, 236494], 'chosen_samples_score': [1.1777544333871777, 1.9141870164936263, 2.1658751666325604, 2.2569560859400295, 2.2864094286197916, 2.297892951888972, 2.2883426656729298, 2.2976298180529335, 2.2997150838200557, 2.2874343355163216], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.230857287009712, 'batch_acquisition_elapsed_time': 1260.7068645610125})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.9364, 'nll': 0.2269335713822527}, 'chosen_targets': [2, 8, 7, 3, 8, 4, 4, 8, 3, 4], 'chosen_samples': [172225, 228115, 274378, 8505, 85495, 127965, 297972, 115311, 128663, 30875], 'chosen_samples_score': [1.1057754880054007, 1.8914417559035255, 2.1360815754549556, 2.2398883063429667, 2.2788994875143533, 2.2929937065273496, 2.297175409168784, 2.3024770014122438, 2.3124034075897564, 2.290621193114537], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.33032045402797, 'batch_acquisition_elapsed_time': 1262.0787176900194})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9441, 'nll': 0.24927330778889223}, 'chosen_targets': [5, 1, 3, 8, 4, 5, 4, 3, 6, 3], 'chosen_samples': [16692, 299289, 134367, 30418, 54628, 28455, 32101, 254367, 36527, 140989], 'chosen_samples_score': [1.1218259365421988, 1.821231064549153, 2.130940680246738, 2.2437008297653356, 2.2804025877815914, 2.294143670588099, 2.302974820928248, 2.2956195346012223, 2.312937603028459, 2.2960329090771667], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.183007562998682, 'batch_acquisition_elapsed_time': 1262.7849526929785})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9481, 'nll': 0.22772179103379325}, 'chosen_targets': [6, 2, 2, 6, 9, 6, 0, 9, 6, 4], 'chosen_samples': [159130, 48460, 201042, 73760, 30144, 44382, 290930, 58413, 13760, 162787], 'chosen_samples_score': [1.2292749549535023, 1.8372067596790125, 2.1158325178360418, 2.2225603869823116, 2.2665882172940197, 2.287425752897871, 2.293702935220152, 2.3005051801824417, 2.3045260674799763, 2.3034855780959926], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.253233028983232, 'batch_acquisition_elapsed_time': 1261.5587689130334})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.9543, 'nll': 0.1824416603006307}, 'chosen_targets': [1, 1, 3, 6, 4, 8, 7, 4, 3, 8], 'chosen_samples': [174388, 273383, 248254, 217050, 121568, 109088, 295244, 272776, 298832, 283532], 'chosen_samples_score': [1.1684321277000431, 1.9127836169503178, 2.1694013129014733, 2.257180044587423, 2.288220155673518, 2.2977202456147605, 2.3035693084185396, 2.307065455531084, 2.2995387172817554, 2.302665602120155], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.475641523022205, 'batch_acquisition_elapsed_time': 1260.7725057359785})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9542, 'nll': 0.1924217134846414}, 'chosen_targets': [1, 0, 0, 0, 5, 2, 3, 3, 8, 0], 'chosen_samples': [3070, 128093, 242192, 271321, 32323, 162532, 166269, 209294, 60602, 212298], 'chosen_samples_score': [1.1936467158075101, 1.9466450745221846, 2.2094744560689907, 2.2707031846357904, 2.2907435831773943, 2.298687874354806, 2.303601839985198, 2.295378790791995, 2.303401727227145, 2.301921712934405], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.328391024959274, 'batch_acquisition_elapsed_time': 1260.7069842269993})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9506, 'nll': 0.21017725580759467}, 'chosen_targets': [0, 3, 4, 1, 3, 0, 6, 0, 8, 9], 'chosen_samples': [162703, 228360, 178390, 170632, 5332, 117507, 125600, 169354, 28987, 34090], 'chosen_samples_score': [1.0526434265596905, 1.746830305385762, 2.096787534844509, 2.2276559141172028, 2.2699936706672714, 2.2894245059629714, 2.2953268269676332, 2.308598951719686, 2.292339535019413, 2.296867864576389], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.202472214004956, 'batch_acquisition_elapsed_time': 1266.5889896390145})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.9554, 'nll': 0.19361211784696106}, 'chosen_targets': [3, 8, 4, 5, 6, 2, 0, 6, 4, 8], 'chosen_samples': [13376, 84589, 197949, 95114, 166088, 5762, 87429, 141404, 230916, 131482], 'chosen_samples_score': [1.3106040117673317, 1.9894429836948826, 2.2187485744870608, 2.2790692988998327, 2.2973795064999107, 2.3011630303726296, 2.3049207452158145, 2.294082601990422, 2.301633384685645, 2.2909479727334836], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 14.315430914983153, 'batch_acquisition_elapsed_time': 1263.9292863360024})
