import numpy
import matplotlib.pyplot as plot
import mpjyky_fuzzy as mpjyky

'''
Mamdani style trading system
'''
#Below is the data used for testing
#This data was taken from TSLA (Tesla) on October 17th, 2022
bid = 383.75#222.75
ask = 383.9#221.95
normalized_spread = abs((ask - bid)/ask)
tsla_beta = 2.13
ma_50 = 334.38#278.18
ma_200 = 287.72#285.47
moving_average = ma_50 - ma_200
rsi_14 = 73.21#36.40

if(moving_average > 100):
    moving_average == 100
elif(moving_average < -100):
    moving_average == -100

if(tsla_beta > 2):
    tsla_beta = 2

beta = numpy.linspace(0, 2, 150) #beta values are 0-2
spread = numpy.arange(0, 1, 0.001) #spread ranges from 0-1
crossovers = numpy.linspace(-100, 100, 150) #crossover values for moving averages
rsi = numpy.linspace(0, 100, 150) #rsi values range from 0-100
beta_spread_fusion = numpy.arange(0, 101, 1) #fusion between beta and spread
crossover_rsi_fusion = numpy.arange(0, 101, 1) # fusion between crossover and spread
flag = numpy.arange(0, 101, 1) #buy/sell decision, neutral peaks at 5; also acts as fusion between the two fusions

#criteria: assigning membership
beta_lo = mpjyky.gaussian(beta, 0, 0.4) #0 is peak, 0.5 is sigma
beta_hi = mpjyky.gaussian(beta, 2, 0.4) #2 is peak, 0.5 is sigma

spread_lo = mpjyky.trapezoid(spread, [0, 0, 0.015, 0.05])
spread_hi = mpjyky.trapezoid(spread, [0.015, 0.05, 1, 1])

crossover_lo = mpjyky.gaussian(crossovers, -100, 40)
crossover_mid = mpjyky.gaussian(crossovers, 0, 32.5)
crossover_hi = mpjyky.gaussian(crossovers, 100, 40)

rsi_lo = mpjyky.gaussian(rsi, 0, 20)
rsi_mid = mpjyky.gaussian(rsi, 50, 16)
rsi_hi = mpjyky.gaussian(rsi, 100, 20)

bsf_lo = mpjyky.triangle(beta_spread_fusion, [0, 0, 50])
bsf_mid = mpjyky.triangle(beta_spread_fusion, [0, 50, 100])
bsf_hi = mpjyky.triangle(beta_spread_fusion, [50, 100, 100])

crf_lo = mpjyky.triangle(crossover_rsi_fusion, [0, 0, 50])
crf_mid = mpjyky.triangle(crossover_rsi_fusion, [0, 50, 100])
crf_hi = mpjyky.triangle(crossover_rsi_fusion, [50, 100, 100])

flag_lo = mpjyky.triangle(flag, [0, 0, 50])
flag_mid = mpjyky.triangle(flag, [25, 50, 75])
flag_hi = mpjyky.triangle(flag, [50, 100, 100])

#fig, (beta_plot, spread_plot, crossover_plot, rsi_plot, bsf_plot, crf_plot, flag_plot) = plot.subplots(nrows=7, figsize=(7, 10))
#fig, (bsf_plot) = plot.subplots(nrows=1, figsize=(7, 10))
#fig, (crf_plot) = plot.subplots(nrows=1, figsize=(7, 10))
fig, (flag_plot) = plot.subplots(nrows=1, figsize=(7, 10))

#interpret membership for the volatility of tesla for the given day
beta_level_lo = mpjyky.getMembership(beta, beta_lo, tsla_beta)
beta_level_hi = mpjyky.getMembership(beta, beta_hi, tsla_beta)

'''beta_lo_degree = numpy.minimum(beta_lo, beta_level_lo)
beta_mid_degree = numpy.minimum(beta_mid,beta_level_mid)
beta_hi_degree = numpy.minimum(beta_hi, beta_level_hi)'''

# interpret membership for the normalized spread of tesla for the given day
spread_level_lo = mpjyky.getMembership(spread, spread_lo, normalized_spread)
spread_level_hi = mpjyky.getMembership(spread, spread_hi, normalized_spread)

'''spread_lo_degree = numpy.minimum(spread_lo, spread_level_lo)
spread_mid_degree = numpy.minimum(spread_mid, spread_level_mid)
spread_hi_degree = numpy.minimum(spread_hi, spread_level_hi)

print('Volatility membership values')
print('Low: ', beta_level_lo)
print('Average: ', beta_level_mid)
print('High: ', beta_level_hi)
print()
print('Bid-Ask Spread membership values')
print('Liquid: ', spread_level_lo)
print('Average: ', spread_level_mid)
print('Illiqid: ', spread_level_hi)'''

#if beta has high volatility and the spread is illiquid then return neutral
bsf_rule1 = numpy.fmin(beta_level_hi, spread_level_hi)
bsf_activation_mid1 = numpy.fmin(bsf_rule1, bsf_mid)# put the data in the neutral section of the fusion

# if beta has high volatility and the spread is liquid then return good
bsf_rule2 = numpy.fmin(beta_level_hi, spread_level_lo)
bsf_activation_hi = numpy.fmin(bsf_rule2, bsf_hi)

#if beta has low volaility and the spread is illiquid then return bad
bsf_rule3 = numpy.fmin(beta_level_lo, spread_level_hi)
bsf_activation_lo = numpy.fmin(bsf_rule3, bsf_lo)

# if beta has low volatility and the spread is liquid then return neutral
bsf_rule4 = numpy.fmin(beta_level_lo, spread_level_lo)
bsf_activation_mid2 = numpy.fmin(bsf_rule4, bsf_mid)

bsf0 = numpy.zeros_like(beta_spread_fusion)

bsf_aggregated = numpy.fmax(bsf_activation_lo, 
             numpy.fmax(bsf_activation_mid1, 
             numpy.fmax(bsf_activation_mid2, bsf_activation_hi)))

bsf_final = mpjyky.defuzzify(beta_spread_fusion, bsf_aggregated)
#bsf_activation = mpjyky.getMembership(beta_spread_fusion, bsf_aggregated, bsf_final)  # for plot

#below is the plot for the first aggregation
'''bsf_plot.plot(beta_spread_fusion, bsf_lo, 'b', linewidth=0.5, linestyle='--', )
bsf_plot.plot(beta_spread_fusion, bsf_mid, 'g', linewidth=0.5, linestyle='--')
bsf_plot.plot(beta_spread_fusion, bsf_hi, 'r', linewidth=0.5, linestyle='--')
bsf_plot.fill_between(beta_spread_fusion, bsf0, bsf_aggregated, facecolor='Orange', alpha=0.7)
bsf_plot.plot([final, final], [0, bsf_activation], 'k', linewidth=1.5, alpha=0.9)
bsf_plot.set_title('Aggregated membership and result (line)')'''

#interpret membership for the double crossover indicator of tesla for the given day
crossover_level_lo = mpjyky.getMembership(crossovers, crossover_lo, moving_average)
crossover_level_mid = mpjyky.getMembership(crossovers, crossover_mid, moving_average)
crossover_level_hi = mpjyky.getMembership(crossovers, crossover_hi, moving_average)

'''crossover_lo_degree = numpy.minimum(crossover_lo, crossover_level_lo)
crossover_mid_degree = numpy.minimum(crossover_mid, crossover_level_mid)
crossover_hi_degree = numpy.minimum(crossover_hi, crossover_level_hi)'''

#interpret membership for the rsi indicator for tesla for the given day
rsi_level_lo = mpjyky.getMembership(rsi, rsi_lo, rsi_14)
rsi_level_mid = mpjyky.getMembership(rsi, rsi_mid, rsi_14)
rsi_level_hi = mpjyky.getMembership(rsi, rsi_hi, rsi_14)

'''rsi_lo_degree = numpy.minimum(rsi_lo, rsi_level_lo)
rsi_mid_degree = numpy.minimum(rsi_mid, rsi_level_mid)
rsi_hi_degree = numpy.minimum(rsi_hi, rsi_level_hi)'''

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

crf0 = numpy.zeros_like(crossover_rsi_fusion)

crf_aggregated = numpy.fmax(crf_activation_lo, 
             numpy.fmax(crf_activation_mid1, 
             numpy.fmax(crf_activation_mid2, crf_activation_hi)))

crf_final = mpjyky.defuzzify(crossover_rsi_fusion, crf_aggregated)
crf_activation = mpjyky.getMembership(crossover_rsi_fusion, crf_aggregated, crf_final)  # for plot

'''crf_plot.plot(crossover_rsi_fusion, crf_lo, 'b', linewidth=0.5, linestyle='--', )
crf_plot.plot(crossover_rsi_fusion, crf_mid, 'g', linewidth=0.5, linestyle='--')
crf_plot.plot(crossover_rsi_fusion, crf_hi, 'r', linewidth=0.5, linestyle='--')
crf_plot.fill_between(crossover_rsi_fusion, crf0, crf_aggregated, facecolor='Orange', alpha=0.7)
crf_plot.plot([crf_final, crf_final], [0, crf_activation], 'k', linewidth=1.5, alpha=0.9)
crf_plot.set_title('Aggregated membership and result (line)')'''

#interpret membership for the beta and spread fusion
bsf_level_lo = mpjyky.getMembership(beta_spread_fusion, bsf_lo, bsf_final)
bsf_level_mid = mpjyky.getMembership(beta_spread_fusion, bsf_mid, bsf_final)
bsf_level_hi = mpjyky.getMembership(beta_spread_fusion, bsf_hi, bsf_final)

#interpret membership for the crossover and rsi fusion
crf_level_lo = mpjyky.getMembership(crossover_rsi_fusion, crf_lo, crf_final)
crf_level_mid = mpjyky.getMembership(crossover_rsi_fusion, crf_mid, crf_final)
crf_level_hi = mpjyky.getMembership(crossover_rsi_fusion, crf_hi, crf_final)

#if bsf is neutral and crf is buy then buy
flag_rule1 = numpy.fmin(bsf_level_mid, crf_level_hi)
flag_activation_hi1 = numpy.fmin(flag_rule1, flag_hi)

#if bsf is neutral and crf is neutral then no trade
flag_rule2 = numpy.fmin(bsf_level_mid, crf_level_mid)
flag_activation_mid1 = numpy.fmin(flag_rule2, flag_mid)

#if bsf is neutral and crf is sell then sell
flag_rule3 = numpy.fmin(bsf_level_mid, crf_level_lo)
flag_activation_lo1 = numpy.fmin(flag_rule3, flag_lo)

#if bsf is good and crf is buy then buy
flag_rule4 = numpy.fmin(bsf_level_hi, crf_level_hi)
flag_activation_hi2 = numpy.fmin(flag_rule4, flag_hi)

#if bsf is good and crf is neutral then no trade
flag_rule5 = numpy.fmin(bsf_level_mid, crf_level_hi)
flag_activation_mid2 = numpy.fmin(flag_rule5, flag_mid)

#if bsf is good and crf is sell then sell
flag_rule6 = numpy.fmin(bsf_level_hi, crf_level_lo)
flag_activation_lo2 = numpy.fmin(flag_rule6, flag_lo)

#if bsf is bad and crf is buy then no trade
flag_rule7 = numpy.fmin(bsf_level_lo, crf_level_hi)
flag_activation_mid3 = numpy.fmin(flag_rule7, flag_mid)

#if bsf is bad and crf is neutral then no trade
flag_rule8 = numpy.fmin(bsf_level_lo, crf_level_mid)
flag_activation_mid4 = numpy.fmin(flag_rule8, flag_mid)

#if bsf is bad and crf is sell then no trade
flag_rule9 = numpy.fmin(bsf_level_lo, crf_level_lo)
flag_activation_mid5 = numpy.fmin(flag_rule9, flag_mid)

flag0 = numpy.zeros_like(flag)

flag_aggregated = numpy.fmax(flag_activation_lo1, 
             numpy.fmax(flag_activation_lo2, 
             numpy.fmax(flag_activation_mid1, 
             numpy.fmax(flag_activation_mid2,
             numpy.fmax(flag_activation_mid3,
             numpy.fmax(flag_activation_mid4,
             numpy.fmax(flag_activation_mid5,
             numpy.fmax(flag_activation_hi1, flag_activation_hi2))))))))

final = mpjyky.defuzzify(flag, flag_aggregated)
print("Indicator value:", final)
activation = mpjyky.getMembership(flag, crf_aggregated, crf_final)  # for plot

flag_plot.plot(flag, flag_lo, 'b', linewidth=0.5, linestyle='--', label="Short")
flag_plot.plot(flag, flag_mid, 'g', linewidth=0.5, linestyle='--', label="No Trade")
flag_plot.plot(flag, flag_hi, 'r', linewidth=0.5, linestyle='--', label="Long")
flag_plot.fill_between(flag, flag0, flag_aggregated, facecolor='Orange', alpha=0.7)
flag_plot.plot([final, final], [0, activation], 'k', linewidth=1.5, alpha=0.9)
flag_plot.set_title('Aggregated membership and result (line)')
flag_plot.legend()

#Below is visualization for identifying membership values
'''beta_plot.plot(beta, beta_lo, 'b', linewidth=1.5, label='Low')
beta_plot.fill_between(beta, beta_lo_degree,color="blue",alpha=0.4)

beta_plot.plot(beta, beta_mid, 'g', linewidth=1.5, label='Average')
beta_plot.fill_between(beta, beta_mid_degree,color="green",alpha=0.4)

beta_plot.plot(beta, beta_hi, 'r', linewidth=1.5, label='High')
beta_plot.fill_between(beta, beta_hi_degree,color="red",alpha=0.4)

beta_plot.set_title('Volatility')
beta_plot.legend()'''

#below is visualization of the membership values
'''
#Plot volatility
beta_plot.plot(beta, beta_lo, 'b', linewidth=1.5, label='Low')
beta_plot.plot(beta, beta_mid, 'g', linewidth=1.5, label='Average')
beta_plot.plot(beta, beta_hi, 'r', linewidth=1.5, label='High')
beta_plot.set_title('Volatility')
beta_plot.legend()

#plot bid-ask spread
spread_plot.plot(spread, spread_lo, 'b', linewidth=1.5, label='Liquid')
spread_plot.plot(spread, spread_mid, 'g', linewidth=1.5, label='Average')
spread_plot.plot(spread, spread_hi, 'r', linewidth=1.5, label='Illiquid')
spread_plot.set_title('Bid-Ask Spread')
spread_plot.legend()

#plot double crossover strategy
crossover_plot.plot(crossovers, crossover_lo, 'b', linewidth=1.5, label='Sell')
crossover_plot.plot(crossovers, crossover_mid, 'g', linewidth=1.5, label='Neutral')
crossover_plot.plot(crossovers, crossover_hi, 'r', linewidth=1.5, label='Buy')
crossover_plot.set_title('Double Crossover')
crossover_plot.legend()

#plot rsi indicator
rsi_plot.plot(rsi, rsi_lo, 'b', linewidth=1.5, label='Oversold')
rsi_plot.plot(rsi, rsi_mid, 'g', linewidth=1.5, label='Neutral')
rsi_plot.plot(rsi, rsi_hi, 'r', linewidth=1.5, label='Overbought')
rsi_plot.set_title('RSI')
rsi_plot.legend()

#plot beta spread fusion
bsf_plot.plot(beta_spread_fusion, bsf_lo, 'b', linewidth=1.5, label='Low')
bsf_plot.plot(beta_spread_fusion, bsf_mid, 'g', linewidth=1.5, label='Average')
bsf_plot.plot(beta_spread_fusion, bsf_hi, 'r', linewidth=1.5, label='High')
bsf_plot.set_title('Beta Spread Fusion')
bsf_plot.legend()

#plot crossover and rsi fusion
crf_plot.plot(crossover_rsi_fusion, crf_lo, 'b', linewidth=1.5, label='Low')
crf_plot.plot(crossover_rsi_fusion, crf_mid, 'g', linewidth=1.5, label='Average')
crf_plot.plot(crossover_rsi_fusion, crf_hi, 'r', linewidth=1.5, label='High')
crf_plot.set_title('Crossover RSI Fusion')
crf_plot.legend()

#plot the decision
flag_plot.plot(flag, flag_lo, 'b', linewidth=1.5, label='Sell')
flag_plot.plot(flag, flag_mid, 'g', linewidth=1.5, label='Neutral')
flag_plot.plot(flag, flag_hi, 'r', linewidth=1.5, label='Buy')
flag_plot.set_title('Decision')
flag_plot.legend()'''

fig.tight_layout()
plot.show()