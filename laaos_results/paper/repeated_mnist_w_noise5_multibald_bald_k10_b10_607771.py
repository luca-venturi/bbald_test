store = {}
store['args']={'dataset': 'DatasetEnum.repeated_mnist_w_noise5', 'num_inference_samples': 10, 'available_sample_k': 10, 'seed': 607771, 'type': 'AcquisitionFunction.bald', 'acquisition_method': 'AcquisitionMethod.multibald', 'experiment_description': 'RMNIST with noise k10 b5 and b10 (and k100 b10), BALD, BatchBALD and heuristic', 'initial_samples': [38043, 40091, 17418, 2094, 39879, 3133, 5011, 40683, 54379, 24287, 9849, 59305, 39508, 39356, 8758, 52579, 13655, 7636, 21562, 41329], 'batch_size': 64, 'scoring_batch_size': 512, 'test_batch_size': 512, 'validation_set_size': 3072, 'early_stopping_patience': 3, 'epochs': 40, 'epoch_samples': 5056, 'target_accuracy': 1.0, 'target_num_acquired_samples': 300, 'initial_percentage': 100, 'reduce_percentage': 0, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 100, 'log_interval': 20, 'experiment_task_id': 'repeated_mnist_w_noise5_multibald_bald_k10_b10_607771', 'experiments_laaos': 'experiment_configs/rmnist_w_noise_2_5/configs.py', 'no_cuda': False, 'quickquick': False, 'initial_samples_per_class': 2, 'balanced_validation_set': False, 'balanced_test_set': False}
store['cmdline']=['./src/ignite_mnist.py', '--experiment_task_id=repeated_mnist_w_noise5_multibald_bald_k10_b10_607771', '--experiments_laaos=experiment_configs/rmnist_w_noise_2_5/configs.py']
store['iterations']=[]
store['initial_samples']=[38043, 40091, 17418, 2094, 39879, 3133, 5011, 40683, 54379, 24287, 9849, 59305, 39508, 39356, 8758, 52579, 13655, 7636, 21562, 41329]
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6379, 'nll': 1.5570398428542054}, 'chosen_targets': [2, 8, 4, 5, 2, 3, 2, 4, 2, 5], 'chosen_samples': [33233, 14055, 249348, 64480, 223806, 18812, 253953, 9348, 106250, 37414], 'chosen_samples_score': [1.331056351755242, 1.989458594527191, 2.206093686010319, 2.272707213273219, 2.2929570647102073, 2.2994443876823523, 2.3018691825204636, 2.2992716049111763, 2.3033400077368107, 2.317198713700308], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.898847899021348, 'batch_acquisition_elapsed_time': 1264.1535641500086})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7302, 'nll': 0.9291442226168755}, 'chosen_targets': [4, 0, 7, 0, 7, 2, 0, 7, 0, 3], 'chosen_samples': [90588, 237916, 280216, 83754, 143678, 83642, 218644, 57763, 145518, 260072], 'chosen_samples_score': [1.19724440166658, 1.8723304420067608, 2.111918541600635, 2.2187865614400084, 2.268550020662598, 2.2883191544741983, 2.2945911129332046, 2.304006201437952, 2.292904884506994, 2.3023859765913386], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 5.031226316001266, 'batch_acquisition_elapsed_time': 1263.996212211001})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.7649, 'nll': 1.2192483775703}, 'chosen_targets': [7, 8, 6, 5, 5, 7, 5, 7, 9, 8], 'chosen_samples': [258066, 112115, 35962, 296547, 222428, 39289, 78404, 53129, 281239, 80133], 'chosen_samples_score': [1.5126657896248534, 2.148224209928189, 2.2723886993701834, 2.2957488913483943, 2.3008802359549536, 2.3021163157639806, 2.304821266491487, 2.307984185536551, 2.309735229979556, 2.297619238703955], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 13.97712777601555, 'batch_acquisition_elapsed_time': 1263.5211517720018})
store['iterations'].append({'num_epochs': 23, 'test_metrics': {'accuracy': 0.8003, 'nll': 0.8190938404976644}, 'chosen_targets': [2, 3, 8, 5, 5, 2, 0, 7, 4, 7], 'chosen_samples': [238398, 208299, 40503, 166203, 169202, 298398, 281877, 126481, 11184, 103143], 'chosen_samples_score': [1.528062806778841, 2.270617652668364, 2.298594935867777, 2.3022210132807395, 2.302527583253217, 2.3025799643274425, 2.3086797181274874, 2.303107625501167, 2.3083872351204286, 2.2933883919542186], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 24.112401387974387, 'batch_acquisition_elapsed_time': 1264.071696392988})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.8048, 'nll': 0.7921247529410538}, 'chosen_targets': [2, 6, 2, 2, 3, 6, 5, 4, 8, 6], 'chosen_samples': [177506, 220591, 161234, 12752, 279461, 188733, 93013, 232140, 125248, 168362], 'chosen_samples_score': [1.545719054628395, 2.172507408480915, 2.272853516737203, 2.29634347669682, 2.3012371391827324, 2.302325073641001, 2.292086148713359, 2.309462557411093, 2.3019871217468806, 2.3014833562477497], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.065974427998299, 'batch_acquisition_elapsed_time': 1264.1450622249977})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.8253, 'nll': 0.770848050446825}, 'chosen_targets': [4, 7, 0, 2, 5, 7, 6, 9, 4, 9], 'chosen_samples': [162774, 23782, 226375, 148944, 97380, 198720, 217292, 14726, 258365, 252767], 'chosen_samples_score': [1.3553750192822391, 2.0315167980853595, 2.2326163008163658, 2.282987431346512, 2.297499988254689, 2.301344386692537, 2.306816718976153, 2.3066667056407657, 2.3126780494602617, 2.301737111389169], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.966241746995365, 'batch_acquisition_elapsed_time': 1263.8215359649912})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.869, 'nll': 0.4376138867010445}, 'chosen_targets': [2, 5, 4, 0, 2, 9, 5, 3, 2, 0], 'chosen_samples': [93593, 47146, 212776, 252497, 3791, 130038, 271891, 11777, 160589, 88272], 'chosen_samples_score': [1.319703181983225, 2.0271682347235935, 2.2591835207197426, 2.2896711770580973, 2.2991395540844617, 2.301764351981457, 2.304759000094313, 2.3080094409189784, 2.305880998869533, 2.310239871128031], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 6.962920935009606, 'batch_acquisition_elapsed_time': 1263.9917269040016})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8904, 'nll': 0.4083086025300773}, 'chosen_targets': [2, 5, 8, 8, 8, 3, 8, 6, 5, 2], 'chosen_samples': [188552, 5188, 40398, 186604, 4797, 81325, 293574, 168006, 214829, 14121], 'chosen_samples_score': [1.1995843962442647, 1.9396691106030794, 2.190287374629576, 2.266971176626556, 2.291152862695708, 2.2989527216528463, 2.295785070738563, 2.2996667785717664, 2.304188895833116, 2.2971620180286836], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.013771027995972, 'batch_acquisition_elapsed_time': 1264.4115116120083})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.865, 'nll': 0.4627595326167636}, 'chosen_targets': [9, 8, 0, 8, 9, 9, 3, 7, 0, 8], 'chosen_samples': [20641, 31474, 223045, 222384, 69390, 16170, 283393, 88930, 283045, 35232], 'chosen_samples_score': [1.167724822576698, 1.8456243978813278, 2.1066977611484816, 2.235100231707567, 2.2810275872237638, 2.2928019990123065, 2.302393060750979, 2.2989016327573646, 2.308230764508634, 2.3168919026763506], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 5.986791894014459, 'batch_acquisition_elapsed_time': 1264.9344849059999})
store['iterations'].append({'num_epochs': 15, 'test_metrics': {'accuracy': 0.9004, 'nll': 0.3970351119761674}, 'chosen_targets': [4, 3, 2, 2, 4, 6, 4, 2, 3, 3], 'chosen_samples': [243984, 19302, 112851, 295906, 164283, 104102, 240828, 175906, 49529, 267420], 'chosen_samples_score': [1.416081597472585, 2.106146713712582, 2.2556362226181013, 2.2936665629859885, 2.3007205311671406, 2.302173745369406, 2.2975758016903463, 2.3060706356512766, 2.2978872332629243, 2.30876245366301], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 16.102753979997942, 'batch_acquisition_elapsed_time': 1264.0670398909715})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9186, 'nll': 0.3148760547100317}, 'chosen_targets': [8, 2, 5, 3, 9, 2, 8, 7, 5, 8], 'chosen_samples': [99668, 232801, 128104, 133045, 91919, 172801, 152774, 38688, 148617, 236428], 'chosen_samples_score': [1.2439651092670236, 1.8945778494310355, 2.201618181143241, 2.271460692523215, 2.2933850329409555, 2.2995200822754875, 2.3021855244895906, 2.290590274754592, 2.289704375181552, 2.29027056052767], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.007035683986032, 'batch_acquisition_elapsed_time': 1263.7981788990146})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.9251, 'nll': 0.2868626341762327}, 'chosen_targets': [5, 5, 0, 3, 8, 7, 6, 5, 1, 4], 'chosen_samples': [296014, 171234, 205803, 275401, 89711, 244832, 107260, 169291, 247631, 219943], 'chosen_samples_score': [1.4089347953787934, 2.042099747194277, 2.2227975967748437, 2.2793534392959613, 2.2961294378852095, 2.3009631876584176, 2.3001733968581943, 2.297666124066367, 2.304143777561921, 2.29852695793455], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 15.109038463997422, 'batch_acquisition_elapsed_time': 1264.045780192013})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9182, 'nll': 0.30937320943100777}, 'chosen_targets': [3, 0, 5, 9, 0, 5, 7, 9, 8, 3], 'chosen_samples': [153357, 50808, 38698, 245740, 289200, 99526, 64530, 58422, 212747, 222477], 'chosen_samples_score': [1.2816818431692538, 1.9832520041588517, 2.2140270153089254, 2.279732265738404, 2.2948763734979245, 2.300206834201401, 2.3082718382864194, 2.302692359468092, 2.3128910935249385, 2.2916409067617325], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.024539881007513, 'batch_acquisition_elapsed_time': 1264.340546552994})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.926, 'nll': 0.2876147198008273}, 'chosen_targets': [8, 3, 9, 3, 7, 7, 8, 2, 9, 7], 'chosen_samples': [49487, 1007, 250070, 188447, 50278, 134337, 49525, 80959, 77296, 169571], 'chosen_samples_score': [1.1049755959681984, 1.7929554429373382, 2.094467320967818, 2.214643410294584, 2.268167027645677, 2.290691351475471, 2.2970659761033945, 2.292720899255348, 2.3038284356280574, 2.2934095374183645], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.043214735982474, 'batch_acquisition_elapsed_time': 1263.268820101017})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9292, 'nll': 0.26300374268958254}, 'chosen_targets': [9, 5, 2, 2, 5, 2, 2, 4, 2, 3], 'chosen_samples': [151077, 229928, 51698, 216884, 39700, 103434, 52225, 234878, 144462, 254266], 'chosen_samples_score': [1.201120221234751, 1.923277262617867, 2.2020134782423, 2.2724757451560667, 2.2960125209543047, 2.3004241730545947, 2.310655789678062, 2.303829807714917, 2.3112799474453474, 2.278588777596287], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.962384017009754, 'batch_acquisition_elapsed_time': 1263.1713806139887})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9285, 'nll': 0.26845077702809733}, 'chosen_targets': [5, 1, 5, 1, 8, 2, 6, 8, 2, 2], 'chosen_samples': [211301, 25910, 281602, 180134, 8867, 162538, 137855, 74385, 276450, 133019], 'chosen_samples_score': [1.2172357251757824, 1.9544350874097025, 2.2263264793290682, 2.2800053843743453, 2.2961276492515332, 2.300611931212884, 2.2978217849099227, 2.301182443402964, 2.2956499683152014, 2.30275887264719], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.010669345996575, 'batch_acquisition_elapsed_time': 1264.2626246449945})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.9341, 'nll': 0.2521711099632088}, 'chosen_targets': [3, 0, 2, 6, 0, 2, 9, 4, 8, 8], 'chosen_samples': [216421, 135191, 112358, 84609, 86184, 170898, 58413, 12194, 26061, 204046], 'chosen_samples_score': [1.308936590184587, 2.0108465222825425, 2.2414547995643335, 2.2863648878602376, 2.296939364962168, 2.3010833193125335, 2.3023544568691046, 2.2989359190158525, 2.3010164333314913, 2.3043509697890925], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.038836702995468, 'batch_acquisition_elapsed_time': 1263.265010100993})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9432, 'nll': 0.23355868725989612}, 'chosen_targets': [8, 4, 6, 2, 7, 3, 5, 6, 9, 2], 'chosen_samples': [173872, 198398, 207669, 6846, 242118, 149185, 220654, 98390, 278760, 205332], 'chosen_samples_score': [1.2079875679598469, 1.8910883067387396, 2.137827665041395, 2.258243597212278, 2.2888452271321458, 2.297014979346746, 2.3075475612744967, 2.2890165624479084, 2.2982124509431934, 2.3051657547435944], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.082604387018364, 'batch_acquisition_elapsed_time': 1264.0510244870093})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9371, 'nll': 0.24457233179809404}, 'chosen_targets': [7, 6, 8, 2, 3, 7, 7, 8, 8, 7], 'chosen_samples': [226132, 15870, 93162, 262169, 140903, 2064, 166132, 94946, 182761, 287560], 'chosen_samples_score': [1.045719538461629, 1.770488853203128, 2.1047556972479526, 2.231161441452769, 2.2748848107152146, 2.2910589066493636, 2.3047082196055113, 2.3019867234740214, 2.3076449740603833, 2.3051084132886928], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.087521180015756, 'batch_acquisition_elapsed_time': 1264.1369355469942})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.9398, 'nll': 0.22440404346097412}, 'chosen_targets': [5, 5, 4, 7, 3, 5, 4, 4, 0, 2], 'chosen_samples': [43212, 114880, 99304, 148632, 201947, 234880, 279304, 105069, 123367, 39561], 'chosen_samples_score': [1.1995495148126927, 1.9825662155447785, 2.2189464725367807, 2.283379778074007, 2.2966392270053353, 2.300403846380247, 2.3037363114990805, 2.294185955321505, 2.297050260698741, 2.288027186976806], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 13.05199466898921, 'batch_acquisition_elapsed_time': 1264.0223425290023})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.946, 'nll': 0.20061813161028588}, 'chosen_targets': [8, 9, 8, 3, 6, 9, 0, 3, 5, 7], 'chosen_samples': [102479, 106021, 19877, 115881, 294994, 129481, 161108, 136193, 142686, 34899], 'chosen_samples_score': [1.2897585174143522, 1.9530833778252275, 2.2116350352264487, 2.2759098946908303, 2.2950715354119677, 2.300459481938549, 2.303535765007076, 2.2947755017174187, 2.3009609107334708, 2.3136886262929512], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 14.147366107994458, 'batch_acquisition_elapsed_time': 1263.290424766019})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9401, 'nll': 0.27081530885914795}, 'chosen_targets': [7, 0, 5, 4, 6, 6, 5, 0, 9, 8], 'chosen_samples': [43898, 114832, 84426, 184822, 13084, 181642, 296224, 72184, 171863, 203008], 'chosen_samples_score': [1.0982957284280161, 1.738794600198992, 2.082727035085241, 2.225831689314541, 2.2792189723213063, 2.29347632655568, 2.289903168277547, 2.302455320624679, 2.2998148723123553, 2.289728787237359], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.003321589989355, 'batch_acquisition_elapsed_time': 1263.636646828003})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9443, 'nll': 0.22527472422977957}, 'chosen_targets': [2, 7, 1, 9, 7, 7, 8, 8, 2, 4], 'chosen_samples': [268844, 278598, 279818, 1740, 71534, 115896, 80037, 276008, 28844, 166786], 'chosen_samples_score': [1.1877345641792263, 1.8870071020657861, 2.190393654906261, 2.26873691142577, 2.290254304375317, 2.298637419869189, 2.3023171458044867, 2.3028195366973714, 2.3120389349613735, 2.306388166832299], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.037679898989154, 'batch_acquisition_elapsed_time': 1263.0920848700043})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9483, 'nll': 0.21052168575172855}, 'chosen_targets': [8, 3, 0, 4, 2, 3, 8, 6, 0, 5], 'chosen_samples': [239294, 148734, 225426, 3218, 36072, 257908, 299294, 292674, 219656, 194100], 'chosen_samples_score': [1.1311459729579436, 1.8479918949406136, 2.1338351088385386, 2.236355310389272, 2.2794526245999096, 2.295093459916619, 2.305418190696262, 2.2908070397633162, 2.306400233643874, 2.297590630260415], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.023874230013462, 'batch_acquisition_elapsed_time': 1264.1209306480014})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9501, 'nll': 0.21608093097994538}, 'chosen_targets': [9, 2, 8, 7, 6, 9, 3, 3, 4, 3], 'chosen_samples': [189118, 143138, 194664, 255518, 49517, 249118, 240670, 149791, 140110, 68765], 'chosen_samples_score': [1.1002179364835833, 1.8047343369305593, 2.1163901448518274, 2.2300926304605806, 2.2720740441207647, 2.2917398421215363, 2.2948440291720487, 2.3031776527362755, 2.298995630947534, 2.311610824340203], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.165367909998167, 'batch_acquisition_elapsed_time': 1267.2509400990093})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9497, 'nll': 0.26320885771970054}, 'chosen_targets': [7, 7, 4, 3, 5, 2, 5, 9, 7, 3], 'chosen_samples': [29786, 205300, 22677, 282199, 115886, 223176, 222317, 121674, 271844, 148374], 'chosen_samples_score': [0.9302789660320239, 1.635344651628957, 2.0131412170501153, 2.1775765199009207, 2.2446229848268153, 2.2761821359787913, 2.2847277887568542, 2.287015016838667, 2.293047393947112, 2.3022620439244275], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.038354428019375, 'batch_acquisition_elapsed_time': 1260.8366046859883})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.9563, 'nll': 0.18869706783257179}, 'chosen_targets': [7, 0, 2, 5, 2, 2, 9, 0, 4, 4], 'chosen_samples': [123798, 56454, 267739, 196011, 189472, 169416, 18598, 116454, 280066, 94101], 'chosen_samples_score': [1.1924249903140345, 2.0136940676952984, 2.2072993962417238, 2.2688758700466254, 2.290445869399555, 2.2980347007696853, 2.3059753789490918, 2.305117279546061, 2.303072731692192, 2.3091664542149415], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 14.09442986099748, 'batch_acquisition_elapsed_time': 1259.8618155930017})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9536, 'nll': 0.20896862708855116}, 'chosen_targets': [0, 0, 2, 0, 6, 9, 9, 5, 8, 5], 'chosen_samples': [296397, 111314, 18631, 21372, 104870, 43248, 241518, 295878, 50231, 38624], 'chosen_samples_score': [1.141785583126397, 1.847237205319484, 2.1582715408853805, 2.2516697101251077, 2.282922332979438, 2.2947541256701878, 2.3045344370969776, 2.2940347883070764, 2.311273342989538, 2.301730595477986], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.091294219979318, 'batch_acquisition_elapsed_time': 1259.9067549429892})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.9582, 'nll': 0.18586196046577394}, 'chosen_targets': [9, 1, 3, 2, 3, 0, 5, 3, 1, 3], 'chosen_samples': [259814, 8704, 39397, 243719, 97161, 165455, 2250, 71708, 101959, 101287], 'chosen_samples_score': [1.0925096962526815, 1.799905194135453, 2.1111213445359804, 2.2370327502345333, 2.279818873778418, 2.2938659938218393, 2.2994654100933136, 2.304163436239042, 2.310221394436725, 2.3002476532231126], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 13.117878131015459, 'batch_acquisition_elapsed_time': 1260.5318544440088})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9523, 'nll': 0.22214537574651813}, 'chosen_targets': [8, 2, 5, 5, 3, 3, 8, 2, 4, 2], 'chosen_samples': [251482, 31794, 259173, 295739, 83812, 246428, 71482, 108767, 29744, 207458], 'chosen_samples_score': [1.0839450302932607, 1.704012073713848, 2.033800077870853, 2.172259242288405, 2.2426104824011035, 2.2731521517475013, 2.286956720599801, 2.295343620234246, 2.303469202528638, 2.3162135743914876], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.177694415993756, 'batch_acquisition_elapsed_time': 1261.9809227460064})
