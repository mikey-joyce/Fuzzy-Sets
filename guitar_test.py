import numpy
import matplotlib.pyplot as plot
import mpjyky_fuzzy as mpjyky

'''
Mamdani style system that assigns the fair value price of Gibson Les Pauls based on the following metrics:
1. Age
2. Model
3. Last Price
'''

gibson_age = 65
gibson_model = 45 #standard = 0, pro = 45, hi = 70
gibson_lprice = 3000

age = numpy.arange(0, 71, 1)
model = numpy.arange(0, 71, 1)
last_price = numpy.arange(0, 300000, 1)
value = numpy.arange(0, 300000, 1)

age_lo = mpjyky.trapezoid(age, [0, 0, 15, 25]) #young
age_mid = mpjyky.trapezoid(age, [17, 22, 40, 55]) #middle aged
age_hi = mpjyky.trapezoid(age, [47, 58, 70, 70]) #old

model_lo = mpjyky.triangle(model, [0, 0, 30]) #standard
model_mid = mpjyky.triangle(model, [30, 45, 60]) #pro
model_hi = mpjyky.triangle(model, [60, 70, 70]) #custom

lprice_lo = mpjyky.trapezoid(last_price, [0, 2000, 3000, 4000]) #cheap
lprice_mid = mpjyky.trapezoid(last_price, [3500, 4000, 8000, 9000]) #average
lprice_hi = mpjyky.trapezoid(last_price, [7500, 10000, 300000, 300000]) #expensive

value_lo = mpjyky.trapezoid(value, [0, 2000, 3000, 4000]) #cheap
value_mid = mpjyky.trapezoid(value, [3500, 4000, 8000, 9000]) #average
value_hi = mpjyky.trapezoid(value, [7500, 10000, 50000, 100000]) #expensive
value_rare = mpjyky.trapezoid(value, [50000, 80000, 300000, 300000]) #rare

#fig, (age_plot, model_plot, lprice_plot, value_plot) = plot.subplots(nrows=4, figsize=(7, 10))
fig, (value_plot) = plot.subplots(nrows=1, figsize=(7, 10))

'''age_plot.plot(age, age_lo, 'b', linewidth=1.5, label='Young')
age_plot.plot(age, age_mid, 'g', linewidth=1.5, label='Middle Aged')
age_plot.plot(age, age_hi, 'r', linewidth=1.5, label='Old')
age_plot.set_title('Age')
age_plot.legend()

model_plot.plot(model, model_lo, 'b', linewidth=1.5, label='Standard')
model_plot.plot(model, model_mid, 'g', linewidth=1.5, label='Pro')
model_plot.plot(model, model_hi, 'r', linewidth=1.5, label='Custom')
model_plot.set_title('Model')
model_plot.legend()

lprice_plot.plot(last_price, lprice_lo, 'b', linewidth=1.5, label='Cheap')
lprice_plot.plot(last_price, lprice_mid, 'g', linewidth=1.5, label='Average')
lprice_plot.plot(last_price, lprice_hi, 'r', linewidth=1.5, label='Expensive')
lprice_plot.set_title('Model')
lprice_plot.legend()

value_plot.plot(value, value_lo, linewidth=1.5, label='Cheap')
value_plot.plot(value, value_mid, linewidth=1.5, label='Average')
value_plot.plot(value, value_hi, linewidth=1.5, label='Expensive')
value_plot.plot(value, value_rare, linewidth=1.5, label='Rare')
value_plot.set_title('Model')
value_plot.legend()'''

age_level_lo = mpjyky.getMembership(age, age_lo, gibson_age)
age_level_mid = mpjyky.getMembership(age, age_mid, gibson_age)
age_level_hi = mpjyky.getMembership(age, age_hi, gibson_age)

model_level_lo = mpjyky.getMembership(model, model_lo, gibson_model)
model_level_mid = mpjyky.getMembership(model, model_mid, gibson_model)
model_level_hi = mpjyky.getMembership(model, model_hi, gibson_model)

lprice_level_lo = mpjyky.getMembership(last_price, lprice_lo, gibson_lprice)
lprice_level_mid = mpjyky.getMembership(last_price, lprice_mid, gibson_lprice)
lprice_level_hi = mpjyky.getMembership(last_price, lprice_hi, gibson_lprice)

'''
Below is the rule set for this non-classification problem
'''
''' The following 9 rules are for old guitars '''
#if age is old and the model is standard and last_price is expensive then return value = rare
value_rule1 = numpy.fmin(age_level_hi, numpy.fmin(model_level_lo, lprice_level_hi))
value_activation_rare1 = numpy.fmin(value_rule1, value_rare)# put the data in the rare section of value

#if age is old and the model is pro and last_price is expensive then return value = rare
value_rule2 = numpy.fmin(age_level_hi, numpy.fmin(model_level_mid, lprice_level_hi))
value_activation_rare2 = numpy.fmin(value_rule2, value_rare)# put the data in the rare section of value

#if age is old and the model is custom and last_price is expensive then return value = rare
value_rule3 = numpy.fmin(age_level_hi, numpy.fmin(model_level_hi, lprice_level_hi))
value_activation_rare3 = numpy.fmin(value_rule3, value_rare)# put the data in the rare section of value

#if age is old aged and the model is standard and last_price is average then return value = rare
value_rule4 = numpy.fmin(age_level_hi, numpy.fmin(model_level_lo, lprice_level_mid))
value_activation_rare4 = numpy.fmin(value_rule4, value_rare)# put the data in the rare section of value

#if age is old and the model is pro and last_price is average then return value = rare
value_rule5 = numpy.fmin(age_level_hi, numpy.fmin(model_level_mid, lprice_level_mid))
value_activation_rare5 = numpy.fmin(value_rule5, value_rare)# put the data in the rare section of value

#if age is old and the model is custom and last_price is average then return value = rare
value_rule6 = numpy.fmin(age_level_hi, numpy.fmin(model_level_hi, lprice_level_mid))
value_activation_rare6 = numpy.fmin(value_rule6, value_rare)# put the data in the rare section of value

#if age is old aged and the model is standard and last_price is cheap then return value = expensive
value_rule7 = numpy.fmin(age_level_hi, numpy.fmin(model_level_lo, lprice_level_lo))
value_activation_hi1 = numpy.fmin(value_rule7, value_hi)# put the data in the expensive section of value

#if age is old and the model is pro and last_price is cheap then return value = expensive
value_rule8 = numpy.fmin(age_level_hi, numpy.fmin(model_level_mid, lprice_level_lo))
value_activation_hi2 = numpy.fmin(value_rule8, value_hi)# put the data in the expensive section of value

#if age is old and the model is custom and last_price is cheap then return value = expensive
value_rule9 = numpy.fmin(age_level_hi, numpy.fmin(model_level_hi, lprice_level_lo))
value_activation_hi3 = numpy.fmin(value_rule9, value_hi)# put the data in the expensive section of value

''' The following 9 rules are for middle aged guitars '''
#if age is middle aged and the model is standard and last_price is expensive then return value = expensive
value_rule10 = numpy.fmin(age_level_mid, numpy.fmin(model_level_lo, lprice_level_hi))
value_activation_hi4 = numpy.fmin(value_rule10, value_hi)# put the data in the expensive section of value

#if age is middle aged and the model is pro and last_price is expensive then return value = expensive
value_rule11 = numpy.fmin(age_level_mid, numpy.fmin(model_level_mid, lprice_level_hi))
value_activation_hi5 = numpy.fmin(value_rule11, value_hi)# put the data in the expensive section of value

#if age is middle aged and the model is custom and last_price is expensive then return value = expensive
value_rule12 = numpy.fmin(age_level_mid, numpy.fmin(model_level_hi, lprice_level_hi))
value_activation_hi6 = numpy.fmin(value_rule3, value_hi)# put the data in the expensive section of value

#if age is middle aged and the model is standard and last_price is average then return value = average
value_rule13 = numpy.fmin(age_level_mid, numpy.fmin(model_level_lo, lprice_level_mid))
value_activation_mid1 = numpy.fmin(value_rule13, value_mid)# put the data in the average section of value

#if age is middle aged and the model is pro and last_price is average then return value = average
value_rule14 = numpy.fmin(age_level_mid, numpy.fmin(model_level_mid, lprice_level_mid))
value_activation_mid2 = numpy.fmin(value_rule14, value_mid)# put the data in the average section of value

#if age is middle aged and the model is custom and last_price is average then return value = expensive
value_rule15 = numpy.fmin(age_level_hi, numpy.fmin(model_level_hi, lprice_level_mid))
value_activation_hi7 = numpy.fmin(value_rule15, value_hi)# put the data in the expensive section of value

#if age is middle aged and the model is standard and last_price is cheap then return value = average
value_rule16 = numpy.fmin(age_level_mid, numpy.fmin(model_level_lo, lprice_level_lo))
value_activation_mid3 = numpy.fmin(value_rule16, value_mid)# put the data in the average section of value

#if age is middle aged and the model is pro and last_price is cheap then return value = average
value_rule17 = numpy.fmin(age_level_mid, numpy.fmin(model_level_mid, lprice_level_lo))
value_activation_mid4 = numpy.fmin(value_rule17, value_mid)# put the data in the average section of value

#if age is middle aged and the model is custom and last_price is cheap then return value = average
value_rule18 = numpy.fmin(age_level_mid, numpy.fmin(model_level_hi, lprice_level_lo))
value_activation_mid5 = numpy.fmin(value_rule18, value_mid)# put the data in the average section of value

''' The following 9 rules are for young aged guitars '''
#if age is young and the model is standard and last_price is expensive then return value = average
value_rule19 = numpy.fmin(age_level_lo, numpy.fmin(model_level_lo, lprice_level_hi))
value_activation_mid6 = numpy.fmin(value_rule19, value_mid)# put the data in the average section of value

#if age is young and the model is pro and last_price is expensive then return value = average
value_rule20 = numpy.fmin(age_level_lo, numpy.fmin(model_level_mid, lprice_level_hi))
value_activation_mid7 = numpy.fmin(value_rule20, value_mid)# put the data in the average section of value

#if age is young and the model is custom and last_price is expensive then return value = expensive
value_rule21 = numpy.fmin(age_level_lo, numpy.fmin(model_level_hi, lprice_level_hi))
value_activation_hi8 = numpy.fmin(value_rule21, value_hi)# put the data in the expensive section of value

#if age is young and the model is standard and last_price is average then return value = cheap
value_rule22 = numpy.fmin(age_level_lo, numpy.fmin(model_level_lo, lprice_level_mid))
value_activation_lo1 = numpy.fmin(value_rule22, value_lo)# put the data in the cheap section of value

#if age is young and the model is pro and last_price is average then return value = cheap
value_rule23 = numpy.fmin(age_level_lo, numpy.fmin(model_level_mid, lprice_level_mid))
value_activation_lo2 = numpy.fmin(value_rule23, value_lo)# put the data in the cheap section of value

#if age is young and the model is custom and last_price is average then return value = average
value_rule24 = numpy.fmin(age_level_lo, numpy.fmin(model_level_hi, lprice_level_mid))
value_activation_mid8 = numpy.fmin(value_rule24, value_mid)# put the data in the average section of value

#if age is young and the model is standard and last_price is cheap then return value = cheap
value_rule25 = numpy.fmin(age_level_lo, numpy.fmin(model_level_lo, lprice_level_lo))
value_activation_lo3 = numpy.fmin(value_rule25, value_lo)# put the data in the cheap section of value

#if age is young and the model is pro and last_price is cheap then return value = cheap
value_rule26 = numpy.fmin(age_level_lo, numpy.fmin(model_level_mid, lprice_level_lo))
value_activation_lo4 = numpy.fmin(value_rule26, value_lo)# put the data in the cheap section of value

#if age is young and the model is custom and last_price is cheap then return value = cheap
value_rule27 = numpy.fmin(age_level_lo, numpy.fmin(model_level_hi, lprice_level_lo))
value_activation_lo5 = numpy.fmin(value_rule27, value_lo)# put the data in the cheap section of value

value0 = numpy.zeros_like(value)

#lo 1-5, mid 1-8, expensive 1-8, rare 1-6
values_aggregated = numpy.fmax(value_activation_lo1,
                    numpy.fmax(value_activation_lo2,
                    numpy.fmax(value_activation_lo3,
                    numpy.fmax(value_activation_lo4,
                    numpy.fmax(value_activation_lo5,
                    numpy.fmax(value_activation_mid1,
                    numpy.fmax(value_activation_mid2,
                    numpy.fmax(value_activation_mid3,
                    numpy.fmax(value_activation_mid4,
                    numpy.fmax(value_activation_mid5,
                    numpy.fmax(value_activation_mid6,
                    numpy.fmax(value_activation_mid7,
                    numpy.fmax(value_activation_mid8,
                    numpy.fmax(value_activation_hi1,
                    numpy.fmax(value_activation_hi2,
                    numpy.fmax(value_activation_hi3,
                    numpy.fmax(value_activation_hi4,
                    numpy.fmax(value_activation_hi5,
                    numpy.fmax(value_activation_hi6,
                    numpy.fmax(value_activation_hi7,
                    numpy.fmax(value_activation_hi8,
                    numpy.fmax(value_activation_rare1,
                    numpy.fmax(value_activation_rare2,
                    numpy.fmax(value_activation_rare3,
                    numpy.fmax(value_activation_rare4,
                    numpy.fmax(value_activation_rare5, value_activation_hi6))))))))))))))))))))))))))

value_final = mpjyky.defuzzify(value, values_aggregated)
value_activation = mpjyky.getMembership(value, values_aggregated, value_final)  # for the plot

#below is the plot for the first aggregation
value_plot.plot(value, value_lo, linewidth=0.5, linestyle='--', )
value_plot.plot(value, value_mid, linewidth=0.5, linestyle='--')
value_plot.plot(value, value_hi, linewidth=0.5, linestyle='--')
value_plot.plot(value, value_rare, linewidth=0.5, linestyle='--')
value_plot.fill_between(value, value0, values_aggregated, facecolor='Orange', alpha=0.7)
value_plot.plot([value_final, value_final], [0, value_activation], 'k', linewidth=1.5, alpha=0.9)
value_plot.set_title('Aggregated membership and result (line)')

print("Fair value price: $", value_final)

fig.tight_layout()
plot.show()