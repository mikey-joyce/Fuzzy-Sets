import numpy
import matplotlib.pyplot as plot
import mpjyky_fuzzy as mpjyky

bid = 222.75
ask = 221.95
normalized_spread = abs((ask - bid)/ask)
tsla_beta = 2
ma_50 = 278.18
ma_200 = 285.47
moving_average = ma_50 - ma_200
rsi_14 = 36.40

beta = numpy.linspace(0, 2, 150) #beta values are 0-2
spread = numpy.arange(0, 1, 0.001) #spread ranges from 0-1
beta_spread_fusion = numpy.arange(0, 101, 1) #fusion between beta and spread
crossovers = numpy.linspace(-100, 100, 150) #crossover values for moving averages
rsi = numpy.arange(0, 101, 1) #rsi values range from 0-100
crossover_rsi_fusion = numpy.arange(0, 101, 1) # fusion between crossover and spread
flag = numpy.arange(0, 101, 1) #buy/sell decision, neutral peaks at 5; also acts as fusion between the two fusions

beta_lo = mpjyky.gaussian(beta, 0.4, 0) #0 is peak
beta_hi = mpjyky.gaussian(beta, 0.4, 2) #2 is peak

spread_lo = mpjyky.trapezoid(spread, [0, 0, 0.015, 0.05])
spread_hi = mpjyky.trapezoid(spread, [0.015, 0.05, 1, 1])

bsf_lo = mpjyky.triangle(beta_spread_fusion, [0, 0, 50])
bsf_mid = mpjyky.triangle(beta_spread_fusion, [0, 50, 100])
bsf_hi = mpjyky.triangle(beta_spread_fusion, [50, 100, 100])

crossover_lo = mpjyky.gaussian(crossovers, 40, -100)
crossover_mid = mpjyky.gaussian(crossovers, 5, 0)
crossover_hi = mpjyky.gaussian(crossovers, 40, 100)

rsi_lo = mpjyky.triangle(rsi, [0, 25, 50])
rsi_mid = mpjyky.triangle(rsi, [40, 50, 60])
rsi_hi = mpjyky.triangle(rsi, [50, 75, 100])

crf_lo = mpjyky.triangle(crossover_rsi_fusion, [0, 0, 50])
crf_mid = mpjyky.triangle(crossover_rsi_fusion, [20, 50, 80])
crf_hi = mpjyky.triangle(crossover_rsi_fusion, [50, 100, 100])

flag_lo = mpjyky.triangle(flag, [0, 0, 50])
flag_mid = mpjyky.triangle(flag, [30, 50, 70])
flag_hi = mpjyky.triangle(flag, [50, 100, 100])

fig, (flag_plot) = plot.subplots(nrows=1, figsize=(7, 7))

#interpret membership for the volatility of tesla for the given day
beta_level_lo = mpjyky.getMembership(beta_lo, beta, tsla_beta)
beta_level_hi = mpjyky.getMembership(beta_hi, beta, tsla_beta)

# interpret membership for the normalized spread of tesla for the given day
spread_level_lo = mpjyky.getMembership(spread_lo, spread, normalized_spread)
spread_level_hi = mpjyky.getMembership(spread_hi, spread, normalized_spread)

#if beta has high volatility and the spread is bad then return neutral
bsf_rule1 = numpy.fmin(beta_level_hi, spread_level_hi)
bsf_activation_mid1 = numpy.fmin(bsf_rule1, bsf_mid)# put the data in the neutral section of the fusion

# if beta has high volatility and the spread is good then return good
bsf_rule2 = numpy.fmin(beta_level_hi, spread_level_lo)
bsf_activation_hi = numpy.fmin(bsf_rule2, bsf_hi)

#if beta has low volaility and the spread is bad then return bad
bsf_rule3 = numpy.fmin(beta_level_lo, spread_level_hi)
bsf_activation_lo = numpy.fmin(bsf_rule3, bsf_lo)

# if beta has low volatility and the spread is good then return neutral
bsf_rule4 = numpy.fmin(beta_level_lo, spread_level_lo)
bsf_activation_mid2 = numpy.fmin(bsf_rule4, bsf_mid)

bsf0 = numpy.zeros_like(beta_spread_fusion)

bsf_aggregated = numpy.fmax(bsf_activation_lo, 
             numpy.fmax(bsf_activation_mid1, 
             numpy.fmax(bsf_activation_mid2, bsf_activation_hi)))


bsf_final = mpjyky.defuzzify(beta_spread_fusion, bsf_aggregated)

#interpret membership for the double crossover indicator of tesla for the given day
crossover_level_lo = mpjyky.getMembership(crossover_lo, crossovers,  moving_average)
crossover_level_mid = mpjyky.getMembership(crossover_mid, crossovers, moving_average)
crossover_level_hi = mpjyky.getMembership(crossover_hi, crossovers, moving_average)

#interpret membership for the rsi indicator for tesla for the given day
rsi_level_lo = mpjyky.getMembership(rsi_lo, rsi, rsi_14)
rsi_level_mid = mpjyky.getMembership(rsi_mid, rsi, rsi_14)
rsi_level_hi = mpjyky.getMembership(rsi_hi, rsi, rsi_14)

#if crossover is buy and rsi is upward trend/overbought then buy
crf_rule1 = numpy.fmin(crossover_level_hi, rsi_level_hi)
crf_activation_hi = numpy.fmin(crf_rule1, crf_hi)# put the data in the buy section of the fusion

# if crossover is sell and rsi is upward trend/overbought then neutral
crf_rule2 = numpy.fmin(crossover_level_lo, rsi_level_hi)
crf_activation_mid1 = numpy.fmin(crf_rule2, crf_mid)

#if crossover is buy and rsi is downward trend/oversold then neutral
crf_rule3 = numpy.fmin(crossover_level_hi, rsi_level_lo)
crf_activation_mid2 = numpy.fmin(crf_rule3, crf_mid)

#if crossover is sell and rsi is downward trend/oversold then sell
crf_rule4 = numpy.fmin(crossover_level_lo, rsi_level_lo)
crf_activation_lo = numpy.fmin(crf_rule4, crf_lo)

#if crossover is mid then neutral
crf_activation_mid3 = numpy.fmin(crossover_level_mid, crf_mid) 

#if rsi is mid then neutral
crf_activation_mid4 = numpy.fmin(rsi_level_mid, crf_mid) 

crf0 = numpy.zeros_like(crossover_rsi_fusion)

crf_aggregated = numpy.fmax(crf_activation_lo, 
             numpy.fmax(crf_activation_mid1, 
             numpy.fmax(crf_activation_mid2, 
             numpy.fmax(crf_activation_mid3,
             numpy.fmax(crf_activation_mid4, crf_activation_hi)))))

crf_final = mpjyky.defuzzify(crossover_rsi_fusion, crf_aggregated)

#interpret membership for the beta and spread fusion
bsf_level_lo = mpjyky.getMembership(bsf_lo, beta_spread_fusion, bsf_final)
bsf_level_mid = mpjyky.getMembership(bsf_mid, beta_spread_fusion, bsf_final)
bsf_level_hi = mpjyky.getMembership(bsf_hi, beta_spread_fusion, bsf_final)

#interpret membership for the crossover and rsi fusion
crf_level_lo = mpjyky.getMembership(crf_lo, crossover_rsi_fusion, crf_final)
crf_level_mid = mpjyky.getMembership(crf_mid, crossover_rsi_fusion, crf_final)
crf_level_hi = mpjyky.getMembership(crf_hi, crossover_rsi_fusion, crf_final)

#if bsf is neutral and crf is good then long
flag_rule1 = numpy.fmin(bsf_level_mid, crf_level_hi)
flag_activation_hi1 = numpy.fmin(flag_rule1, flag_hi)

#if bsf is neutral and crf is neutral then no trade
flag_rule2 = numpy.fmin(bsf_level_mid, crf_level_mid)
flag_activation_mid1 = numpy.fmin(flag_rule2, flag_mid)

#if bsf is neutral and crf is bad then short
flag_rule3 = numpy.fmin(bsf_level_mid, crf_level_lo)
flag_activation_lo1 = numpy.fmin(flag_rule3, flag_lo)

#if bsf is best and crf is good then long
flag_rule4 = numpy.fmin(bsf_level_hi, crf_level_hi)
flag_activation_hi2 = numpy.fmin(flag_rule4, flag_hi)

#if bsf is best and crf is neutral then no trade
flag_rule5 = numpy.fmin(bsf_level_mid, crf_level_hi)
flag_activation_mid2 = numpy.fmin(flag_rule5, flag_mid)

#if bsf is best and crf is bad then short
flag_rule6 = numpy.fmin(bsf_level_hi, crf_level_lo)
flag_activation_lo2 = numpy.fmin(flag_rule6, flag_lo)

#if bsf is worst then no trade
flag_activation_mid3 = numpy.fmin(bsf_level_lo, flag_mid)

flag0 = numpy.zeros_like(flag)

flag_aggregated = numpy.fmax(flag_activation_lo1, 
             numpy.fmax(flag_activation_lo2, 
             numpy.fmax(flag_activation_mid1, 
             numpy.fmax(flag_activation_mid2,
             numpy.fmax(flag_activation_mid3,
             numpy.fmax(flag_activation_hi1, flag_activation_hi2))))))

final = mpjyky.defuzzify(flag, flag_aggregated)
print("Indicator value:", final)
activation = mpjyky.getMembership(crf_aggregated, flag, crf_final)  # for plot

flag_plot.plot(flag, flag_lo, 'b', linewidth=0.5, linestyle='--', label="Short")
flag_plot.plot(flag, flag_mid, 'g', linewidth=0.5, linestyle='--', label="No Trade")
flag_plot.plot(flag, flag_hi, 'r', linewidth=0.5, linestyle='--', label="Long")
flag_plot.fill_between(flag, flag0, flag_aggregated, facecolor='Orange', alpha=0.7)
flag_plot.plot([final, final], [0, activation], 'k', linewidth=1.5, alpha=0.9)
flag_plot.set_title('Decision')
flag_plot.legend()

classifier = 1 #0 = short, 1 = no-trade, 2 = long

if(final < 35):
    print("Open short position")
    classifier = 0
elif(final > 65):
    print("Open long position")
    classifier = 2
else:
    print("No-trade")

print("Classifier:",classifier)
fig.tight_layout()
plot.show()