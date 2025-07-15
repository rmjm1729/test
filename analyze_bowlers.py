
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

# Test match bowling dataset (source: Cricinfo, see script comments)
data_str = '''Player\tSpan\tMat\tInns\tBalls\tOvers\tMdns\tRuns\tWkts\tBBI\tAve\tEcon\tSR\t4\t5\t10
M Muralidaran (ICC/SL)\t1992-2010\t133\t230\t44039\t7339.5\t1794\t18180\t800\t9/51\t22.72\t2.47\t55.04\t45\t67\t22
SK Warne (AUS)\t1992-2007\t145\t273\t40705\t6784.1\t1761\t17995\t708\t8/71\t25.41\t2.65\t57.49\t48\t37\t10
JM Anderson (ENG)\t2003-2024\t188\t350\t40037\t6672.5\t1730\t18627\t704\t7/42\t26.45\t2.79\t56.87\t32\t32\t3
A Kumble (IND)\t1990-2008\t132\t236\t40850\t6808.2\t1576\t18355\t619\t10/74\t29.65\t2.69\t65.99\t31\t35\t8
SCJ Broad (ENG)\t2007-2023\t167\t309\t33698\t5616.2\t1304\t16719\t604\t8/15\t27.68\t2.97\t55.79\t28\t20\t3
GD McGrath (AUS)\t1993-2007\t124\t243\t29248\t4874.4\t1470\t12186\t563\t8/24\t21.64\t2.49\t51.95\t28\t29\t3
NM Lyon (AUS)\t2011-2025\t139\t259\t34502\t5750.2\t1089\t16942\t562\t8/50\t30.14\t2.94\t61.39\t26\t24\t5
R Ashwin (IND)\t2011-2024\t106\t200\t27246\t4541.0\t907\t12891\t537\t7/59\t24.00\t2.83\t50.73\t25\t37\t8
CA Walsh (WI)\t1984-2001\t132\t242\t30019\t5003.1\t1144\t12688\t519\t7/37\t24.44\t2.53\t57.84\t32\t22\t3
DW Steyn (SA)\t2004-2019\t93\t171\t18608\t3101.2\t660\t10077\t439\t7/51\t22.95\t3.24\t42.38\t27\t26\t5
N Kapil Dev (IND)\t1978-1994\t131\t227\t27740\t4623.2\t1060\t12867\t434\t9/83\t29.64\t2.78\t63.91\t17\t23\t2
HMRKB Herath (SL)\t1999-2018\t93\t170\t25993\t4332.1\t814\t12157\t433\t9/127\t28.07\t2.80\t60.03\t20\t34\t9
Sir RJ Hadlee (NZ)\t1973-1990\t86\t150\t21918\t-\t809\t9611\t431\t9/52\t22.29\t2.63\t50.85\t25\t36\t9
SM Pollock (SA)\t1995-2008\t108\t202\t24353\t4058.5\t1222\t9733\t421\t7/87\t23.11\t2.39\t57.84\t23\t16\t1
Harbhajan Singh (IND)\t1998-2015\t103\t190\t28580\t4763.2\t871\t13537\t417\t8/84\t32.46\t2.84\t68.53\t16\t25\t5
Wasim Akram (PAK)\t1985-2002\t104\t181\t22627\t3771.1\t871\t9779\t414\t7/119\t23.62\t2.59\t54.65\t20\t25\t5
CEL Ambrose (WI)\t1988-2000\t98\t179\t22103\t3683.5\t1001\t8501\t405\t8/45\t20.99\t2.30\t54.57\t21\t22\t3
MA Starc (AUS)\t2011-2025\t100\t192\t19094\t3182.2\t606\t10863\t402\t6/9\t27.02\t3.41\t47.49\t20\t16\t2
TG Southee (NZ)\t2008-2024\t107\t203\t23490\t3915.0\t889\t11832\t391\t7/64\t30.26\t3.02\t60.07\t19\t15\t1
M Ntini (SA)\t1998-2009\t101\t190\t20834\t3472.2\t759\t11242\t390\t7/37\t28.82\t3.23\t53.42\t19\t18\t4
IT Botham (ENG)\t1977-1992\t102\t168\t21815\t-\t788\t10878\t383\t8/34\t28.40\t2.99\t56.95\t17\t27\t4
MD Marshall (WI)\t1978-1991\t81\t151\t17584\t2930.4\t614\t7876\t376\t7/22\t20.94\t2.68\t46.76\t19\t22\t4
Waqar Younis (PAK)\t1989-2003\t87\t154\t16224\t2704.0\t516\t8788\t373\t7/76\t23.56\t3.25\t43.49\t28\t22\t5
Imran Khan (PAK)\t1971-1992\t88\t142\t19458\t-\t727\t8258\t362\t8/58\t22.81\t2.54\t53.75\t17\t23\t6
DL Vettori (ICC/NZ)\t1997-2014\t113\t187\t28814\t4802.2\t1197\t12441\t362\t7/87\t34.36\t2.59\t79.59\t19\t20\t3
DK Lillee (AUS)\t1971-1984\t70\t132\t18467\t-\t652\t8493\t355\t7/83\t23.92\t2.75\t52.01\t23\t23\t7
WPUJC Vaas (SL)\t1994-2009\t111\t194\t23438\t3906.2\t895\t10501\t355\t7/71\t29.58\t2.68\t66.02\t20\t12\t2
K Rabada (SA)\t2015-2025\t71\t130\t13100\t2183.2\t433\t7306\t336\t7/112\t21.74\t3.34\t38.98\t15\t17\t4
AA Donald (SA)\t1992-2002\t72\t129\t15519\t2586.3\t661\t7344\t330\t8/71\t22.25\t2.83\t47.02\t11\t20\t3
RA Jadeja (IND)\t2012-2025\t83\t156\t19067\t3177.5\t742\t8129\t326\t7/42\t24.93\t2.55\t58.48\t13\t15\t3
RGD Willis (ENG)\t1971-1984\t90\t165\t17357\t-\t554\t8190\t325\t8/43\t25.20\t2.83\t53.40\t12\t16\t-\nTA Boult (NZ)\t2011-2022\t78\t149\t17417\t2902.5\t656\t8717\t317\t6/30\t27.49\t3.00\t54.94\t18\t10\t1
MG Johnson (AUS)\t2007-2015\t73\t140\t16001\t2666.5\t514\t8891\t313\t8/61\t28.40\t3.33\t51.12\t16\t12\t3
I Sharma (IND)\t2007-2021\t105\t188\t19160\t3193.2\t640\t10078\t311\t7/74\t32.40\t3.15\t61.60\t10\t11\t1
Z Khan (IND)\t2000-2014\t92\t165\t18785\t3130.5\t624\t10247\t311\t7/87\t32.94\t3.27\t60.40\t15\t11\t1
B Lee (AUS)\t1999-2008\t76\t150\t16531\t2755.1\t547\t9554\t310\t5/30\t30.81\t3.46\t53.32\t17\t10\t-\nPJ Cummins (AUS)\t2011-2025\t71\t132\t14187\t2364.3\t514\t6829\t309\t6/23\t22.10\t2.88\t45.91\t17\t14\t2
M Morkel (SA)\t2006-2018\t86\t160\t16498\t2749.4\t605\t8550\t309\t6/23\t27.66\t3.10\t53.39\t18\t8\t-\nLR Gibbs (WI)\t1958-1976\t79\t148\t27115\t-\t1313\t8989\t309\t8/38\t29.09\t1.98\t87.75\t11\t18\t2
FS Trueman (ENG)\t1952-1965\t67\t127\t15178\t-\t522\t6625\t307\t8/31\t21.57\t2.61\t49.43\t19\t17\t3
DL Underwood (ENG)\t1966-1982\t86\t151\t21862\t-\t1239\t7674\t297\t8/51\t25.83\t2.10\t73.60\t13\t17\t6
JR Hazlewood (AUS)\t2014-2025\t76\t143\t15423\t2570.3\t657\t7144\t295\t6/67\t24.21\t2.77\t52.28\t11\t13\t-\nJH Kallis (ICC/SA)\t1995-2013\t166\t272\t20232\t3372.0\t848\t9535\t292\t6/54\t32.65\t2.82\t69.28\t7\t5\t-\nCJ McDermott (AUS)\t1984-1996\t71\t124\t16586\t2764.2\t579\t8332\t291\t8/97\t28.63\t3.01\t56.99\t17\t14\t2
KAJ Roach (WI)\t2009-2025\t85\t154\t14944\t2490.4\t540\t7729\t284\t6/48\t27.21\t3.10\t52.61\t14\t11\t1
BS Bedi (IND)\t1966-1979\t67\t118\t21364\t-\t1096\t7637\t266\t7/98\t28.71\t2.14\t80.31\t13\t14\t1
Danish Kaneria (PAK)\t2000-2010\t61\t112\t17697\t2949.3\t517\t9082\t261\t7/77\t34.79\t3.07\t67.80\t8\t15\t2
N Wagner (NZ)\t2012-2024\t64\t122\t13725\t2287.3\t473\t7169\t260\t7/39\t27.57\t3.13\t52.78\t13\t9\t-\nJ Garner (WI)\t1977-1987\t58\t111\t13169\t2194.5\t576\t5433\t259\t6/56\t20.97\t2.47\t50.84\t18\t7\t-\nJN Gillespie (AUS)\t1996-2006\t71\t137\t14234\t2372.2\t630\t6770\t259\t7/37\t26.13\t2.85\t54.95\t8\t8\t-\nGP Swann (ENG)\t2008-2013\t60\t109\t15349\t2558.1\t493\t7642\t255\t6/65\t29.96\t2.98\t60.19\t14\t17\t3
JB Statham (ENG)\t1951-1965\t70\t129\t16056\t-\t595\t6261\t252\t7/39\t24.84\t2.33\t63.71\t9\t9\t1
MA Holding (WI)\t1975-1987\t60\t113\t12680\t-\t459\t5898\t249\t8/92\t23.68\t2.79\t50.92\t11\t13\t2
R Benaud (AUS)\t1952-1964\t63\t116\t19108\t-\t805\t6704\t248\t7/72\t27.03\t2.10\t77.04\t16\t16\t1
MJ Hoggard (ENG)\t2000-2008\t67\t122\t13909\t2318.1\t493\t7564\t248\t7/61\t30.50\t3.26\t56.08\t13\t7\t1
GD McKenzie (AUS)\t1961-1971\t60\t113\t17681\t-\t547\t7328\t246\t8/71\t29.78\t2.48\t71.87\t7\t16\t3
Shakib Al Hasan (BAN)\t2007-2024\t71\t121\t15675\t2612.3\t486\t7804\t246\t7/36\t31.72\t2.98\t63.71\t11\t19\t2
Yasir Shah (PAK)\t2014-2022\t48\t89\t14255\t2375.5\t359\t7657\t244\t8/41\t31.38\t3.22\t58.42\t15\t16\t3
BS Chandrasekhar (IND)\t1964-1979\t58\t97\t15963\t-\t584\t7199\t242\t8/79\t29.74\t2.70\t65.96\t12\t16\t2
Taijul Islam (BAN)\t2014-2025\t55\t98\t14585\t2430.5\t425\t7423\t237\t8/39\t31.32\t3.05\t61.54\t12\t17\t2
AV Bedser (ENG)\t1946-1955\t51\t92\t15918\t-\t574\t5876\t236\t7/44\t24.89\t2.21\t67.44\t11\t15\t5
J Srinath (IND)\t1991-2002\t67\t121\t15104\t2517.2\t599\t7196\t236\t8/86\t30.49\t2.85\t64.00\t8\t10\t1
Abdul Qadir (PAK)\t1977-1990\t67\t111\t17126\t-\t608\t7742\t236\t9/56\t32.80\t2.71\t72.56\t12\t15\t5
GS Sobers (WI)\t1954-1974\t93\t159\t21599\t-\t974\t7999\t235\t6/73\t34.03\t2.22\t91.91\t8\t6\t-\nAR Caddick (ENG)\t1993-2003\t62\t105\t13558\t2259.4\t501\t6999\t234\t7/46\t29.91\t3.09\t57.94\t9\t13\t1
CS Martin (NZ)\t2000-2013\t71\t126\t14026\t2337.4\t486\t7878\t233\t6/26\t33.81\t3.37\t60.19\t10\t10\t1
Mohammed Shami (IND)\t2013-2023\t64\t122\t11515\t1919.1\t364\t6346\t229\t6/56\t27.71\t3.30\t50.28\t12\t6\t-\nD Gough (ENG)\t1994-2003\t58\t95\t11821\t1970.1\t369\t6503\t229\t6/42\t28.39\t3.30\t51.62\t14\t9\t-\nRR Lindwall (AUS)\t1946-1960\t61\t113\t13650\t-\t419\t5251\t228\t7/38\t23.03\t2.30\t59.86\t8\t12\t-\nSJ Harmison (ENG/ICC)\t2002-2009\t63\t115\t13375\t2229.1\t431\t7192\t226\t7/12\t31.82\t3.22\t59.18\t11\t8\t1
A Flintoff (ENG/ICC)\t1998-2009\t79\t137\t14951\t2491.5\t507\t7410\t226\t5/58\t32.78\t2.97\t66.15\t11\t3\t-\nVD Philander (SA)\t2011-2020\t64\t119\t11391\t1898.3\t507\t5000\t224\t6/21\t22.32\t2.63\t50.85\t8\t13\t2
BA Stokes (ENG)\t2013-2025\t114\t168\t12950\t2158.2\t374\t7173\t224\t6/22\t32.02\t3.32\t57.81\t9\t4\t-\nPM Siddle (AUS)\t2008-2019\t67\t126\t13907\t2317.5\t615\t6777\t221\t6/54\t30.66\t2.92\t62.92\t8\t8\t-\nCL Cairns (NZ)\t1989-2004\t62\t104\t11698\t1949.4\t414\t6410\t218\t7/27\t29.40\t3.28\t53.66\t11\t13\t1
JJ Bumrah (IND)\t2018-2025\t47\t90\t9150\t1525.0\t361\t4229\t217\t6/27\t19.48\t2.77\t42.16\t7\t15\t-\nCV Grimmett (AUS)\t1925-1936\t37\t67\t14513\t-\t736\t5231\t216\t7/40\t24.21\t2.16\t67.18\t7\t21\t7
HH Streak (ZIM)\t1993-2005\t65\t102\t13559\t2259.5\t595\t6079\t216\t6/73\t28.14\t2.69\t62.77\t16\t7\t-\nMG Hughes (AUS)\t1985-1994\t53\t97\t12285\t2047.3\t499\t6017\t212\t8/87\t28.38\t2.93\t57.94\t14\t7\t1
SCG MacGill (AUS)\t1998-2008\t44\t85\t11237\t1872.5\t365\t6038\t208\t8/108\t29.02\t3.22\t54.02\t9\t12\t2
Saqlain Mushtaq (PAK)\t1995-2004\t49\t86\t14070\t2345.0\t541\t6206\t208\t8/164\t29.83\t2.64\t67.64\t12\t13\t3
Mehidy Hasan Miraz (BAN)\t2016-2025\t54\t94\t12696\t2116.0\t328\t6635\t205\t7/58\t32.36\t3.13\t61.93\t9\t13\t3
MM Ali (ENG)\t2014-2023\t68\t119\t12610\t2101.4\t293\t7612\t204\t6/53\t37.31\t3.62\t61.81\t13\t5\t1
KA Maharaj (SA)\t2016-2025\t59\t100\t11666\t1944.2\t367\t6055\t203\t9/129\t29.82\t3.11\t57.46\t6\t11\t1
AME Roberts (WI)\t1974-1983\t47\t90\t11135\t-\t382\t5174\t202\t7/54\t25.61\t2.78\t55.12\t8\t11\t2
JA Snow (ENG)\t1965-1976\t49\t93\t12021\t-\t415\t5387\t202\t7/40\t26.66\t2.68\t59.50\t12\t8\t1
JR Thomson (AUS)\t1972-1985\t51\t90\t10535\t-\t301\t5601\t200\t6/46\t28.00\t3.18\t52.67\t16\t8\t-'''

# This dataset was taken from Cricinfo (espncricinfo.com), manually compiled for analysis purposes.
data = pd.read_csv(StringIO(data_str), sep='\t')



# --- ALGORITHM EXPLANATION ---
# This script ranks Test match bowlers using a composite score based on multiple metrics, normalized for era and career length.
#
# 1. Feature Engineering:
#    - Wickets per match: Measures strike power, not just longevity.
#    - Career span (years): Rewards bowlers with long, sustained careers.
#
# 2. Era Normalization:
#    - Bowling average and economy rate are adjusted by dividing by the typical average/economy for the bowler's era (based on career midpoint).
#    - This allows fair comparison between bowlers from different periods, accounting for changes in scoring rates and conditions.
#
# 3. Parameter Selection:
#    - Metrics used: Adjusted bowling average, strike rate, adjusted economy, wickets per match, 5-fors, 10-fors, career span.
#    - Lower is better for average, strike rate, economy; higher is better for the rest.
#
# 4. Normalization:
#    - All metrics are standardized (z-score) so they are on the same scale.
#    - For metrics where lower is better, the z-score is inverted so that higher is always better.
#
# 5. Ideal Bowler Definition:
#    - The 'ideal' bowler is defined as the mean of the top 10% for each metric (i.e., a composite of the best performers on each axis).
#    - This avoids biasing toward outliers and gives a realistic benchmark.
#
# 6. Scoring:
#    - Each bowler's metrics are compared to the ideal using Euclidean distance in the normalized space.
#    - Lower distance means closer to the ideal, i.e., a better all-round record.
#
# 7. Output:
#    - The top 10 bowlers by this composite score are printed, along with their metrics.

# --- Feature Engineering ---
# Calculate wickets per match (strike power)
data['WktsPerMatch'] = data['Wkts'] / data['Mat']

# Calculate career span in years (longevity)
def get_span_years(span):
    try:
        start, end = span.split('-')
        return int(end) - int(start) + 1
    except:
        return np.nan
data['SpanYears'] = data['Span'].apply(get_span_years)

# --- Era Normalization ---
# Assign each bowler to an era based on career midpoint, then adjust average/economy by era mean
era_bins = [0, 1919, 1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2100]
era_labels = [
    'pre1920', '1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'
]
era_avg = {
    'pre1920': {'Avg': 25, 'Econ': 2.63},
    '1920s':   {'Avg': 32, 'Econ': 2.63},
    '1930s':   {'Avg': 35, 'Econ': 2.63},
    '1940s':   {'Avg': 32, 'Econ': 2.28},
    '1950s':   {'Avg': 28.5, 'Econ': 2.28},
    '1960s':   {'Avg': 32, 'Econ': 2.49},
    '1970s':   {'Avg': 32, 'Econ': 2.66},
    '1980s':   {'Avg': 32, 'Econ': 2.85},
    '1990s':   {'Avg': 31.5, 'Econ': 2.83},
    '2000s':   {'Avg': 34.1, 'Econ': 3.08},
    '2010s':   {'Avg': 33.85, 'Econ': 3.15},
    '2020s':   {'Avg': 33.85, 'Econ': 3.22},
}
def get_era(span):
    try:
        start, end = span.split('-')
        mid = (int(start) + int(end)) // 2
        for i in range(len(era_bins)-1):
            if era_bins[i] <= mid < era_bins[i+1]:
                return era_labels[i]
        return '2010s'
    except:
        return '2010s'
data['Era'] = data['Span'].apply(get_era)
data['AdjAvg'] = data.apply(lambda x: x['Avg'] / era_avg[x['Era']]['Avg'] if pd.notnull(x['Avg']) else np.nan, axis=1)
data['AdjEcon'] = data.apply(lambda x: x['Econ'] / era_avg[x['Era']]['Econ'] if pd.notnull(x['Econ']) else np.nan, axis=1)

# --- Parameters for Scoring ---
# These are the metrics used for ranking
parameters = [
    'AdjAvg',        # Adjusted bowling average (lower better)
    'SR',            # Strike rate (lower better)
    'AdjEcon',       # Adjusted economy rate (lower better)
    'WktsPerMatch',  # Wickets per match (higher better)
    '5w',            # 5-wicket hauls (higher better)
    '10w',           # 10-wicket hauls (higher better)
    'SpanYears'      # Career span in years (higher better)
]

# --- Normalization ---
# Standardize all metrics (z-score), then invert lower-is-better so higher is always better
lower_better = ['AdjAvg', 'SR', 'AdjEcon']
higher_better = ['WktsPerMatch', '5w', '10w', 'SpanYears']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_z = pd.DataFrame(scaler.fit_transform(data[parameters]), columns=parameters)
for col in lower_better:
    data_z[col] = -data_z[col]

# --- Define Ideal Bowler (center) as mean of top 10% for each metric ---
# This creates a composite 'ideal' bowler for comparison
def top10_mean(col):
    n = max(1, int(0.1 * len(data_z)))
    return data_z[col].sort_values(ascending=False).head(n).mean()
center = np.array([top10_mean(col) for col in data_z.columns])

# --- Euclidean Distance from Ideal ---
# The lower the distance, the closer the bowler is to the ideal
def euclidean(row):
    return np.linalg.norm(row.values - center)
data['score'] = data_z.apply(euclidean, axis=1)

# Lower score = closer to ideal bowler
best_bowlers = data.sort_values('score')


# --- K-MEANS CLUSTERING OF BOWLERS ---
# We use k-means to find natural groupings of bowlers based on their normalized metrics.
# Number of clusters = int(sqrt(N)), where N is the number of bowlers (minimum 2).
from sklearn.cluster import KMeans
N = len(data_z)
n_clusters = max(2, int(np.sqrt(N)))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(data_z)

# Print cluster assignments for each bowler (top 10 by score)
print("\nTop 10 bowlers and their cluster assignments:")
print(best_bowlers[['Player', 'score', 'cluster'] + parameters].head(10))

# Print cluster centers (in normalized metric space)
print(f"\nK-means cluster centers (z-score space, columns: {parameters}):")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}: {center}")
