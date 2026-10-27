# Final validation results — 27 October 2026

{
  "retained": 5858,
  "train": 3104,
  "validation": 2754,
  "clusters_retained": 2222,
  "clusters_train": 1566,
  "clusters_validation": 1004,
  "minimum_clusters_per_subgenre": 25,
  "validation_clusters_crossing_subgenres": 79,
  "clusters_crossing_splits": 348
}

Weights: 1/(n_cg m_g), equal component weight within each subgenre. Components use valid artist/album links and retain mixed-subgenre observations together.
H1 is primary at 0.05; H2 and exploratory H3 use Holm (sorted cumulative maximum). Omnibus alternatives are non-directional; test statistics use the upper tail.
All confidence intervals are pointwise. SESOI values are project-defined analytical benchmarks, not validated perceptual thresholds.

## H1
H0: D_1=...=D_17. Whole-cluster bootstrap covariance of log mean distances; chi-square(16) Wald calibration. No small-sample dispersion multiplier or ridge. Raw-label permutation would not isolate dispersion from centroid differences; PERMANOVA tests location. Bootstrap assumes independent components and adequate effective sample size; training transforms are treated as fixed. Ratio extrema are selected and descriptive.

{
  "statistic": 251.93515403306247,
  "p": 2.0748009281502905e-44,
  "rank": 16,
  "condition": 44.95281516753776,
  "df": 16,
  "status": "inferential (asymptotic)",
  "ratio": 1.3589504727483221,
  "ratio_ci": [
    1.3109497726992447,
    1.468082306529906
  ],
  "B": 1999,
  "attempts": 1999,
  "seed": 20261027,
  "beyond_SESOI": 4,
  "most": "blackmetal",
  "least": "hairmetal",
  "decision": "reject H0",
  "method": "whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald"
}

subgenre,clusters,D_g,D_g_ci_lo,D_g_ci_hi,r_g,r_g_ci_lo,r_g_ci_hi,beyond_SESOI,wald_statistic,df,p_raw,status,p_holm,decision,method,alpha
alternativemetal,34,7.068974487886879,6.520083158074035,7.4040048935768175,0.949881846519854,0.8852342518313677,1.0007514467159728,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
blackmetal,167,8.600039988935643,8.335514520570957,8.811669315944354,1.1556162607225267,1.1271605117213963,1.1974888883953747,True,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
deathmetal,93,8.112870733130903,7.7473998114099905,8.4321590575486,1.0901536914255987,1.050187633778445,1.1425481675088194,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
doommetal,71,7.933636776851087,7.581713254483413,8.201131487092432,1.0660694226760277,1.0278381493283657,1.11185785924526,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
folkmetal,25,8.093200605367905,7.277918111352233,8.57545777638392,1.0875105502864115,0.9908800184450198,1.1575091003909626,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
gothicmetal,47,7.277564864426813,6.874235543970709,7.572084911422807,0.9779108360675659,0.9336795180702016,1.0259302952185638,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
grindcore,25,8.471028951095748,7.659828912895967,9.045430422269986,1.1382806142217803,1.0404364009167557,1.2218435378710228,True,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
hairmetal,43,6.328442545476322,5.984276635918064,6.615191087093635,0.850374081982123,0.8107502707604816,0.8958499699960101,True,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
heavymetal,113,7.122649026568275,6.885195317708901,7.3377653908576335,0.9570942745744678,0.9329817176052334,0.9958127737086523,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
industrialmetal,60,7.3294993869076706,6.91833691385934,7.64996910709432,0.9848894523006244,0.94035244838659,1.0356461263600358,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
melodicdeathmetal,36,7.146498265177066,6.643629081838845,7.479537712337232,0.9602989768755767,0.9050241294893955,1.0125863803270776,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
metalcore,63,6.624682139920125,6.243402061113624,6.94686217216893,0.8901807913520261,0.8485046390461467,0.9400308713918339,True,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
numetal,40,6.71533123836639,6.345497209640447,6.9359266544540334,0.902361615199254,0.8607063236902445,0.9392017972534853,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
postmetal,37,7.646243293320016,6.989907375064688,8.123830772732912,1.0274513949434263,0.9518624101289487,1.101580558173467,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
powermetal,75,7.148116314357783,6.808093115221352,7.414994529623055,0.9605163995789934,0.9234972492987021,1.0040869247427333,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
progressivemetal,91,7.790223797599934,7.4088613235352545,8.115221202624305,1.0467985389319385,1.0048042769692203,1.0995667082069636,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05
thrashmetal,73,7.543998703815386,7.160461852828483,7.853873053805721,1.0137124460136,0.9727551451889621,1.0645321595997006,False,251.93515403306247,16,2.0748009281502905e-44,inferential (asymptotic),,reject H0,whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald,0.05


## H2
H0: beta_1=...=beta_17. WLS subgenre intercepts/slopes plus extractor-version indicator; CR1 factor M/(M-1)*(N-1)/(N-K). Whole-cluster score covariance, chi-square(16) Wald. Pooled means equal-subgenre average, not an imposed common slope. Normal 95% CIs. IID regression ignores repeated recordings; mixed models require a specified random-effect model; Spearman answers a different question. Requires useful log-linear specification and enough independent clusters.

{
  "statistic": 20.869177529901716,
  "p": 0.18359425556220088,
  "rank": 16,
  "condition": 71.80647075836481,
  "df": 16,
  "status": "inferential (asymptotic)",
  "pooled_slope": 0.016369427542354334,
  "pooled_se": 0.025450328392556773,
  "pooled_ci": [
    -0.03351321610705694,
    0.0662520711917656
  ],
  "training_bpm_median": 127.75801086425781,
  "beyond_SESOI": 10,
  "design_rank": 35,
  "parameters": 35,
  "clusters": 1004,
  "CR1_factor": 1.0135140734471833,
  "design_condition": 7.212770380827855,
  "p_holm": 0.18359425556220088,
  "decision": "fail to reject H0",
  "method": "WLS interaction; whole-component CR1; asymptotic chi-square Wald"
}

subgenre,clusters,slope_beta_g,ci_lo,ci_hi,slope_contrast,beyond_SESOI,pooled_slope,pooled_se,pooled_ci_lo,pooled_ci_hi,wald_statistic,df,p_raw,status,p_holm,decision,method,alpha
alternativemetal,34,0.011937158578093751,-0.17713815628025878,0.20101247343644627,-0.0044322689642605825,False,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
blackmetal,167,0.18917106892460275,0.04339824499306408,0.33494389285614146,0.1728016413822484,True,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
deathmetal,93,0.0028368716731267405,-0.18576894119553222,0.1914426845417857,-0.013532555869227594,False,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
doommetal,71,0.18494962543203913,-0.09575351970716689,0.46565277057124516,0.1685801978896848,True,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
folkmetal,25,0.112446268216971,-0.1820495943226858,0.40694213075662783,0.09607684067461666,True,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
gothicmetal,47,0.00396942455739402,-0.2006368683045796,0.20857571741936765,-0.012400002984960315,False,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
grindcore,25,-0.08295856790860656,-0.4343937303920181,0.268476594574805,-0.0993279954509609,True,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
hairmetal,43,-0.08944660834653573,-0.18873649006392035,0.009843273370848893,-0.10581603588889006,True,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
heavymetal,113,-0.025725397263827176,-0.12818093574847644,0.0767301412208221,-0.04209482480618151,False,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
industrialmetal,60,-0.05008243519164935,-0.23050472111086606,0.13033985072756737,-0.06645186273400369,True,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
melodicdeathmetal,36,-0.028063639956149165,-0.136071177555344,0.07994389764304567,-0.044433067498503495,False,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
metalcore,63,-0.0238482559179176,-0.1044661679568461,0.0567696561210109,-0.040217683460271936,False,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
numetal,40,0.04928928921498402,-0.12052609403996353,0.21910467246993157,0.03291986167262968,False,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
postmetal,37,-0.12565657990536194,-0.48274853996313005,0.23143538015240617,-0.14202600744771626,True,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
powermetal,75,0.0695817253467414,-0.02969011498862166,0.16885356568210447,0.05321229780438706,True,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
progressivemetal,91,-0.0904717375616565,-0.2943934304643109,0.11344995534099792,-0.10684116510401083,True,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05
thrashmetal,73,0.17035205832777492,0.00098767881104192,0.3397164378445079,0.15398263078542057,True,0.016369427542354334,0.025450328392556773,-0.03351321610705694,0.0662520711917656,20.869177529901716,16,0.18359425556220088,inferential (asymptotic),0.18359425556220088,fail to reject H0,WLS interaction; whole-component CR1; asymptotic chi-square Wald,0.05


## H3 exploratory
H0: p_1=...=p_17. Saturated weighted linear probability model, whole-cluster CR1 covariance, chi-square(16) Wald. Normal 95% CIs are not clipped. Marginal association, not version-adjusted. Cluster-label permutation is inappropriate for mixed-subgenre components; track-level chi-square ignores dependence. Estimated mode is imperfect. Exploratory because validation mode counts were inspected during design.

{
  "statistic": 30.01255023606004,
  "p": 0.01793722711782703,
  "rank": 16,
  "condition": 45.043444686752586,
  "df": 16,
  "status": "inferential (asymptotic)",
  "overall_minor": 0.6491767012685346,
  "min_share": 0.4178592725104354,
  "max_share": 0.8053333333333336,
  "range_pp": 38.747406082289814,
  "min_group": "hairmetal",
  "max_group": "folkmetal",
  "intervals_outside_unit": 0,
  "beyond_SESOI": 3,
  "design_rank": 17,
  "parameters": 17,
  "clusters": 1004,
  "CR1_factor": 1.0068486538921781,
  "design_condition": 1.0000000000000033,
  "p_holm": 0.03587445423565406,
  "decision": "reject H0",
  "method": "exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald"
}

subgenre,clusters,minor_share,ci_lo,ci_hi,overall_minor,difference_pp,beyond_SESOI,wald_statistic,df,p_raw,status,p_holm,decision,method,alpha
alternativemetal,34,0.4911764705882353,0.32975764606108554,0.652595295115385,0.6491767012685346,-15.800023068029923,True,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
blackmetal,167,0.6920240105868862,0.6364315245960039,0.7476164965777685,0.6491767012685346,4.284730931835168,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
deathmetal,93,0.6826533382984995,0.6072048285798379,0.7581018480171611,0.6491767012685346,3.347663702996495,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
doommetal,71,0.6835558807389792,0.5925504966459761,0.7745612648319823,0.6491767012685346,3.4379179470444665,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
folkmetal,25,0.8053333333333336,0.6856430029890797,0.9250236636775875,0.6491767012685346,15.615663206479901,True,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
gothicmetal,47,0.7349290780141844,0.6271280858840947,0.842730070144274,0.6491767012685346,8.575237674564985,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
grindcore,25,0.6101362365069455,0.46470035648739155,0.7555721165264995,0.6491767012685346,-3.9040464761589067,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
hairmetal,43,0.4178592725104354,0.28782179870482916,0.5478967463160417,0.6491767012685346,-23.131742875809913,True,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
heavymetal,113,0.6728173839755222,0.5986392568475201,0.7469955111035242,0.6491767012685346,2.3640682706987604,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
industrialmetal,60,0.6897222222222222,0.5933564251875748,0.7860880192568697,0.6491767012685346,4.054552095368768,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
melodicdeathmetal,36,0.6099987974987975,0.4694595025141149,0.7505380924834801,0.6491767012685346,-3.9177903769737066,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
metalcore,63,0.7058201058201052,0.6069616303435811,0.8046785812966294,0.6491767012685346,5.664340455157069,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
numetal,40,0.6828030303030306,0.5584479592131314,0.8071581013929298,0.6491767012685346,3.3626329034496028,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
postmetal,37,0.639382239382239,0.5119372807552658,0.7668271980092123,0.6491767012685346,-0.9794461886295514,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
powermetal,75,0.6673540903540902,0.5773339245890641,0.7573742561191164,0.6491767012685346,1.8177389085555662,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
progressivemetal,91,0.6195512820512813,0.5319087433222374,0.7071938207803252,0.6491767012685346,-2.9625419217253257,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05
thrashmetal,73,0.6308871493802999,0.5360884842593457,0.725685814501254,0.6491767012685346,-1.8289551888234667,False,30.01255023606004,16,0.01793722711782703,inferential (asymptotic),0.03587445423565406,reject H0,exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald,0.05


## Limitations
Artist overlap between training and validation remains; no split was changed. Unobserved artist links can leave residual dependence. CR1 and Wald calibration are asymptotic, especially uncertain for subgenres with fewer clusters. Pointwise CIs and counts beyond SESOI do not constitute simultaneous post-hoc testing. No unrun sensitivity analyses are claimed. Bootstrap Monte Carlo precision is finite. Failure to reject is not evidence of equality; dispersion is descriptor variation, not creativity.