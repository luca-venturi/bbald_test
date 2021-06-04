store = {}
store['args']={'num_inference_samples': 100, 'available_sample_k': 10, 'initial_samples': [38043, 40091, 17418, 2094, 39879, 3133, 5011, 40683, 54379, 24287, 9849, 59305, 39508, 39356, 8758, 52579, 13655, 7636, 21562, 41329], 'seed': 734490, 'type': 'AcquisitionFunction.bald', 'acquisition_method': 'AcquisitionMethod.independent', 'experiment_description': 'Reproduce b1, 5, 10 k10, 100 and default initial samples, no initial samples for all methods with BALD. (No culling!)', 'batch_size': 64, 'scoring_batch_size': 512, 'test_batch_size': 512, 'validation_set_size': 1024, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'target_accuracy': 0.96, 'target_num_acquired_samples': 300, 'log_interval': 20, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 20, 'dataset': 'DatasetEnum.mnist', 'experiment_task_id': 'mnist_independent_bald_k100_b10_734490', 'experiments_laaos': './experiment_configs/big_repro_b10_k100/configs.py', 'no_cuda': False, 'quickquick': False, 'initial_samples_per_class': 2, 'initial_percentage': 100, 'reduce_percentage': 0}
store['cmdline']=['./src/ignite_mnist.py', '--experiment_task_id=mnist_independent_bald_k100_b10_734490', '--experiments_laaos=./experiment_configs/big_repro_b10_k100/configs.py']
store['iterations']=[]
store['initial_samples']=[38043, 40091, 17418, 2094, 39879, 3133, 5011, 40683, 54379, 24287, 9849, 59305, 39508, 39356, 8758, 52579, 13655, 7636, 21562, 41329]
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.655, 'nll': 2.657730563030958}, 'chosen_samples': [26956, 32695, 38009, 21511, 27897, 19815, 23562, 19402, 47650, 21443], 'chosen_samples_score': [1.1411482277475922, 1.1214994688077864, 1.1171262275911107, 1.1150493958128571, 1.1137487799699164, 1.1115191855282895, 1.110015231817354, 1.1015261601314847, 1.1006522154526759, 1.0991114047330777], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.572172365151346, 'batch_acquisition_elapsed_time': 53.77221647510305})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7005, 'nll': 2.0889118400351405}, 'chosen_samples': [16945, 23156, 54397, 22231, 30644, 22918, 12146, 1526, 20586, 43245], 'chosen_samples_score': [1.2308257320543328, 1.2068267720270067, 1.189654933117947, 1.1865222934317525, 1.1788392553119444, 1.1781839724112637, 1.152325364141796, 1.1519434977160796, 1.147718149400713, 1.1352300905480561], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.81064630812034, 'batch_acquisition_elapsed_time': 53.48970452696085})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7465, 'nll': 1.5867504427132904}, 'chosen_samples': [18962, 6520, 49472, 48030, 31396, 56664, 39640, 51440, 10210, 45180], 'chosen_samples_score': [1.1641959695697373, 1.1319705882412872, 1.1216314084082377, 1.0894198088771323, 1.0838675847064199, 1.0836697012928016, 1.066192746545596, 1.0441848563505887, 1.033153132759438, 1.032222158340491], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.821855828166008, 'batch_acquisition_elapsed_time': 53.498715239111334})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7698, 'nll': 1.3426256113693118}, 'chosen_samples': [48752, 35044, 23391, 54481, 41544, 54461, 21457, 21395, 21421, 16745], 'chosen_samples_score': [1.1433958017178876, 1.1012829863210993, 1.0766185283972576, 1.0690381783111609, 1.0665114103071585, 1.0656762790114431, 1.0610284136685966, 1.0484564247849488, 1.0441894532712341, 1.04187107100946], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.799225492402911, 'batch_acquisition_elapsed_time': 53.496927035972476})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7927, 'nll': 1.1446808468886018}, 'chosen_samples': [20840, 36562, 10916, 7786, 40382, 32693, 11269, 33522, 56362, 20641], 'chosen_samples_score': [1.0204711513226266, 1.0096987521802474, 0.9823354662385084, 0.9662337853218899, 0.9625832708842471, 0.9569759626699609, 0.9538076618793178, 0.9426965291377254, 0.9379661690149418, 0.9299587496065148], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.792164872866124, 'batch_acquisition_elapsed_time': 53.54637164901942})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7917, 'nll': 1.0349132704917192}, 'chosen_samples': [21672, 942, 58497, 32065, 28309, 12345, 18597, 19298, 27503, 30508], 'chosen_samples_score': [0.9228269143614658, 0.9127608593020599, 0.9007693236938267, 0.8997575892009564, 0.8789513166898247, 0.8788410995802217, 0.875528167137942, 0.8706961321670591, 0.8690843900221615, 0.8677634773725742], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.816339650191367, 'batch_acquisition_elapsed_time': 53.45732620311901})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7627, 'nll': 1.1652858114311693}, 'chosen_samples': [17494, 47132, 49202, 6898, 25315, 1674, 49955, 37270, 38549, 58476], 'chosen_samples_score': [0.8506765222963232, 0.8476925727270802, 0.8407412317801236, 0.8393926842645263, 0.832897526322582, 0.8243000012056902, 0.8223038615158498, 0.8216931171898075, 0.8204845696854588, 0.8189122876159777], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.889524134807289, 'batch_acquisition_elapsed_time': 53.36239801021293})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7784, 'nll': 1.0740062521579266}, 'chosen_samples': [33312, 56563, 8318, 51761, 48885, 3739, 39812, 28159, 40721, 24596], 'chosen_samples_score': [0.7840311688390169, 0.7767767204434783, 0.7583182137456972, 0.7571851360237927, 0.7498159939299267, 0.7462320862807428, 0.7436480510209817, 0.7432663182666306, 0.7432453604868454, 0.7409432386977006], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.842238612938672, 'batch_acquisition_elapsed_time': 53.11159333400428})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8676, 'nll': 0.8675701436349155}, 'chosen_samples': [8702, 31736, 54880, 33518, 45003, 49889, 39700, 14139, 12157, 14121], 'chosen_samples_score': [1.0764793736702944, 1.033429254215184, 1.0191906288512969, 1.0060929602100221, 1.0057747312545526, 1.0002232809498985, 0.9976085093190433, 0.9906621923627414, 0.9816600647599258, 0.9797405585313493], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.897292202338576, 'batch_acquisition_elapsed_time': 52.71602480439469})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8349, 'nll': 0.8533538154012562}, 'chosen_samples': [45094, 51986, 18739, 22673, 5045, 44267, 47741, 56839, 56726, 11133], 'chosen_samples_score': [0.728363584228084, 0.7105078147655562, 0.7012926501459802, 0.6981370591070613, 0.6882583435313796, 0.6869926467674901, 0.6844408650248972, 0.6801022787069867, 0.6795944328271696, 0.6774505204504416], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 7.826315613929182, 'batch_acquisition_elapsed_time': 53.16270437929779})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8436, 'nll': 0.9114484999289513}, 'chosen_samples': [47068, 2856, 28373, 23187, 30188, 42302, 16692, 11711, 48365, 8996], 'chosen_samples_score': [1.0324319704927265, 0.9359171727587038, 0.9344278533697663, 0.9279001479176564, 0.9139089159919109, 0.9078138194700215, 0.900659565451641, 0.8955974498202849, 0.8914361263635795, 0.8875318921185299], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.815031382720917, 'batch_acquisition_elapsed_time': 53.23433614661917})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8671, 'nll': 0.80141763114053}, 'chosen_samples': [12497, 39668, 3730, 31090, 10265, 59747, 23104, 29320, 28102, 31777], 'chosen_samples_score': [0.9735406202531701, 0.9719632073694724, 0.9605246019367608, 0.9557286492311033, 0.9489488660324109, 0.9470073111849553, 0.946727789553031, 0.9414615083701778, 0.9397327791749028, 0.9317366939944447], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.89201457798481, 'batch_acquisition_elapsed_time': 53.264940951019526})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8734, 'nll': 0.767202547970295}, 'chosen_samples': [7033, 43575, 7621, 38688, 28632, 59615, 3751, 12188, 38669, 26072], 'chosen_samples_score': [0.9263930762010614, 0.9249121540494059, 0.8942169243099408, 0.8891845561714545, 0.8865064733002859, 0.8863164880551015, 0.8758703390266425, 0.8719942295989666, 0.8605803146082116, 0.8543336251651097], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.847596127074212, 'batch_acquisition_elapsed_time': 53.37867783010006})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8889, 'nll': 0.7841643082404136}, 'chosen_samples': [14726, 8594, 15106, 5155, 5175, 29286, 2502, 36282, 14295, 3494], 'chosen_samples_score': [1.04656022742896, 1.0316580530847683, 0.9989937659636211, 0.9504469707834304, 0.9478735280186035, 0.9473685882757431, 0.9462849206629426, 0.9429287430975813, 0.9424149108764572, 0.9421149485897478], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.895700308028609, 'batch_acquisition_elapsed_time': 53.37011136300862})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8882, 'nll': 0.7927638594635726}, 'chosen_samples': [33812, 9633, 2381, 20050, 8047, 59314, 19356, 17756, 4784, 18003], 'chosen_samples_score': [1.1353673968663593, 1.0513506168782925, 1.046246720522263, 1.0273331203607932, 1.0152471599127, 1.0103014650230921, 1.009564619257024, 1.0058046998812813, 1.0034200851329165, 1.0001864640136873], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.907122048083693, 'batch_acquisition_elapsed_time': 53.09884640108794})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8976, 'nll': 0.6448273212362527}, 'chosen_samples': [5129, 41143, 32880, 52697, 59957, 57195, 52753, 49149, 41233, 52771], 'chosen_samples_score': [0.8630678532242455, 0.8468354074858833, 0.8111988947529936, 0.8097623982791379, 0.8053908199043693, 0.788200129147876, 0.785629733494229, 0.7850831073707758, 0.781034252969721, 0.7804411118152352], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 8.865188281051815, 'batch_acquisition_elapsed_time': 53.54445031331852})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.9078, 'nll': 0.6328576196150779}, 'chosen_samples': [50052, 39405, 52358, 42337, 581, 47737, 42121, 13827, 38974, 3980], 'chosen_samples_score': [0.9996472553861064, 0.986630651331358, 0.9684787951959263, 0.9644391361674058, 0.9346843176076645, 0.9325927486704169, 0.9251095514342685, 0.9189341242782357, 0.90919124760046, 0.9061678661334965], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.875412643887103, 'batch_acquisition_elapsed_time': 53.23926763609052})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9187, 'nll': 0.5556979308021665}, 'chosen_samples': [17209, 17213, 34829, 57342, 6474, 24533, 31756, 39526, 47140, 45048], 'chosen_samples_score': [1.0201162321410058, 0.9841119878732373, 0.9751099407056522, 0.9693074205346806, 0.9597872136548967, 0.958512468037516, 0.9511879498801684, 0.9485205946761612, 0.9464382899623022, 0.9452083864176867], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.958294005133212, 'batch_acquisition_elapsed_time': 53.545351123437285})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9203, 'nll': 0.5545686504355667}, 'chosen_samples': [8443, 40208, 34942, 34946, 11534, 49890, 52834, 21174, 10736, 33505], 'chosen_samples_score': [1.0319774142293658, 0.975609544086864, 0.9600728304809774, 0.9331127406784555, 0.9318062604767805, 0.9315241619623111, 0.920912579802465, 0.9172601155832918, 0.9016041724728748, 0.897539764818204], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.932254515122622, 'batch_acquisition_elapsed_time': 53.354565993417054})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.9171, 'nll': 0.5645255165233609}, 'chosen_samples': [57972, 32702, 58560, 44172, 50916, 2803, 4955, 43368, 46610, 55612], 'chosen_samples_score': [0.991372978719957, 0.8621026895224558, 0.8545288425270666, 0.8492051356699529, 0.8475320415115792, 0.8441457369462584, 0.8432751387930533, 0.8418994722419606, 0.8402619937608538, 0.8371479338987818], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 9.914850586093962, 'batch_acquisition_elapsed_time': 53.56776203913614})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9163, 'nll': 0.5115299880441428}, 'chosen_samples': [1075, 4873, 49204, 53872, 55311, 13998, 49242, 6418, 9118, 1239], 'chosen_samples_score': [0.9917754953524837, 0.8980286852432751, 0.8947323268525614, 0.8917374465184364, 0.889278765590936, 0.8808721050989597, 0.8787069411421462, 0.8785519824541806, 0.8780374248449059, 0.8681579499518265], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.867622253019363, 'batch_acquisition_elapsed_time': 53.51315817097202})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9256, 'nll': 0.5548790855804085}, 'chosen_samples': [31124, 57474, 49364, 20035, 59294, 32971, 34304, 22139, 49660, 12513], 'chosen_samples_score': [1.0776677552237532, 1.0759116272820965, 1.0753885885708854, 1.0643347669375476, 1.0603142187838528, 1.06008567064875, 1.055896876120991, 1.0541382517255178, 1.0339810936984826, 1.0152765751602706], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 11.933823533821851, 'batch_acquisition_elapsed_time': 53.24630166217685})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.9372, 'nll': 0.5516479579259753}, 'chosen_samples': [37773, 43176, 37078, 42931, 59834, 18487, 36818, 50905, 23112, 51158], 'chosen_samples_score': [1.1250531476669698, 1.1068817760237293, 1.0985293130915181, 1.0715591150660204, 1.0608878958727828, 1.053891637007652, 1.0502606422872092, 1.0466270182016242, 1.0442784171036092, 1.0421997290923422], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 14.993898816872388, 'batch_acquisition_elapsed_time': 53.61493820976466})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9254, 'nll': 0.5435960230081678}, 'chosen_samples': [13942, 4360, 49187, 10032, 35962, 31954, 15276, 23790, 52544, 32215], 'chosen_samples_score': [1.0488183451611146, 1.027146216268756, 1.0182103053619151, 1.0110265272828824, 0.9877132201340528, 0.9839461168877642, 0.9793744297251377, 0.977885918652627, 0.9755477186017124, 0.9712765115424719], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.923312815837562, 'batch_acquisition_elapsed_time': 53.36795092886314})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9403, 'nll': 0.44177731345152854}, 'chosen_samples': [42078, 28192, 3810, 2450, 57732, 34328, 50459, 36744, 44328, 15855], 'chosen_samples_score': [1.0658965006673453, 1.0291990932944923, 1.0130995739323851, 0.9978337358827261, 0.9791034925831298, 0.9685315100605716, 0.9652821418258524, 0.9623866665229986, 0.9540583137677332, 0.9520403609904895], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 13.059958933852613, 'batch_acquisition_elapsed_time': 53.7121218550019})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9331, 'nll': 0.4951281076288818}, 'chosen_samples': [22531, 27429, 20172, 5265, 26184, 40169, 8459, 51337, 3268, 12305], 'chosen_samples_score': [0.9822288541215946, 0.9695773650148554, 0.9468370613023122, 0.9412772362682728, 0.922439456886634, 0.9201952502555967, 0.9104679071225354, 0.9097265135616603, 0.9063388721819682, 0.9054060433436094], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 13.00777246337384, 'batch_acquisition_elapsed_time': 53.31238414905965})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9332, 'nll': 0.49787802196890124}, 'chosen_samples': [13969, 4153, 32668, 9180, 52086, 6440, 9433, 10044, 1352, 33306], 'chosen_samples_score': [0.8799738815681905, 0.8492510828894547, 0.8319097613547934, 0.8206570878556937, 0.8168997515742774, 0.8134183566241818, 0.7982530000299785, 0.7949349957803932, 0.7946044772409927, 0.7935364181262884], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 10.857537167146802, 'batch_acquisition_elapsed_time': 53.32754321908578})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9447, 'nll': 0.4561089275090694}, 'chosen_samples': [517, 670, 13259, 26208, 9098, 41048, 32519, 509, 14588, 54950], 'chosen_samples_score': [0.9823985314815799, 0.967167241011012, 0.9642946819811372, 0.9636113064802921, 0.9434534903363165, 0.9209787029838612, 0.9188180607418734, 0.9183501672957601, 0.9136577171800564, 0.9074430995595065], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 12.888043161015958, 'batch_acquisition_elapsed_time': 53.20259522134438})
