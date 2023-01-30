import numpy
import matplotlib.pyplot as plot
import mpjyky_fuzzy as mpjyky

'''
Mamdani style system that assigns the fair value price of Gibson Les Pauls based on the following metrics:
1. Age
2. Model
3. Last Price
'''

gibson_age = 59
gibson_model = 0 #standard = 0, pro = 45, custom = 70
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

lprice_lo = mpjyky.trapezoid(last_price, [0, 0, 3000, 4000]) #cheap
lprice_mid = mpjyky.trapezoid(last_price, [3500, 4000, 8000, 9000]) #average
lprice_hi = mpjyky.trapezoid(last_price, [7500, 10000, 300000, 300000]) #expensive

value_lo = mpjyky.trapezoid(value, [0, 0, 3000, 4000]) #cheap
value_mid = mpjyky.trapezoid(value, [3500, 4000, 8000, 9000]) #average
value_hi = mpjyky.trapezoid(value, [7500, 10000, 50000, 100000]) #expensive
value_rare = mpjyky.trapezoid(value, [50000, 80000, 300000, 300000]) #rare

fig, (value_plot) = plot.subplots(nrows=1, figsize=(7, 10))

age_level_lo = mpjyky.getMembership(age_lo, age, gibson_age)
age_level_mid = mpjyky.getMembership(age_mid, age, gibson_age)
age_level_hi = mpjyky.getMembership(age_hi, age, gibson_age)

model_level_lo = mpjyky.getMembership(model_lo, model, gibson_model)
model_level_mid = mpjyky.getMembership(model_mid, model, gibson_model)
model_level_hi = mpjyky.getMembership(model_hi, model, gibson_model)

lprice_level_lo = mpjyky.getMembership(lprice_lo, last_price, gibson_lprice)
lprice_level_mid = mpjyky.getMembership(lprice_mid, last_price, gibson_lprice)
lprice_level_hi = mpjyky.getMembership(lprice_hi, last_price, gibson_lprice)

'''
Below is the rule set for this non-classification problem
'''
''' The following 4 rules are for old guitars '''
#if age is old and the model is standard and last_price is expensive OR average then return value = rare
value_rule1 = numpy.fmin(age_level_hi, numpy.fmin(model_level_lo, numpy.fmax(lprice_level_hi, lprice_level_mid)))
value_activation_rare1 = numpy.fmin(value_rule1, value_rare)# put the data in the rare section of value

#if age is old and the model is pro and last_price is expensive OR average then return value = rare
value_rule2 = numpy.fmin(age_level_hi, numpy.fmin(model_level_mid, numpy.fmax(lprice_level_hi, lprice_level_mid)))
value_activation_rare2 = numpy.fmin(value_rule2, value_rare)# put the data in the rare section of value

#if age is old and the model is custom and last_price is expensive OR average then return value = rare
value_rule3 = numpy.fmin(age_level_hi, numpy.fmin(model_level_hi, numpy.fmax(lprice_level_hi, lprice_level_mid)))
value_activation_rare3 = numpy.fmin(value_rule3, value_rare)# put the data in the rare section of value

#if age is old aged last_price is cheap then return value = expensive
value_rule4 = numpy.fmin(age_level_hi, lprice_level_lo)
value_activation_hi1 = numpy.fmin(value_rule4, value_hi)# put the data in the expensive section of value

''' The following 6 rules are for middle aged guitars '''
#if age is middle aged and the model is standard and last_price is expensive then return value = expensive
value_rule5 = numpy.fmin(age_level_mid, numpy.fmin(model_level_lo, lprice_level_hi))
value_activation_hi2 = numpy.fmin(value_rule5, value_hi)# put the data in the expensive section of value

#if age is middle aged and the model is pro and last_price is expensive then return value = expensive
value_rule6 = numpy.fmin(age_level_mid, numpy.fmin(model_level_mid, lprice_level_hi))
value_activation_hi3 = numpy.fmin(value_rule6, value_hi)# put the data in the expensive section of value

#if age is middle aged and the model is custom and last_price is expensive OR average then return value = expensive
value_rule7 = numpy.fmin(age_level_mid, numpy.fmin(model_level_hi, numpy.fmax(lprice_level_hi, lprice_level_mid)))
value_activation_hi4 = numpy.fmin(value_rule7, value_hi)# put the data in the expensive section of value

#if age is middle aged and the model is standard and last_price is average OR cheap then return value = average
value_rule8 = numpy.fmin(age_level_mid, numpy.fmin(model_level_lo, numpy.fmax(lprice_level_lo, lprice_level_mid)))
value_activation_mid1 = numpy.fmin(value_rule8, value_mid)# put the data in the average section of value

#if age is middle aged and the model is pro and last_price is average OR cheap then return value = average
value_rule9 = numpy.fmin(age_level_mid, numpy.fmin(model_level_mid, numpy.fmax(lprice_level_mid, lprice_level_lo)))
value_activation_mid2 = numpy.fmin(value_rule9, value_mid)# put the data in the average section of v

#if age is middle aged and the model is custom and last_price is cheap then return value = average
value_rule10 = numpy.fmin(age_level_mid, numpy.fmin(model_level_hi, lprice_level_lo))
value_activation_mid3 = numpy.fmin(value_rule10, value_mid)# put the data in the average section of value

''' The following 6 rules are for young aged guitars '''
#if age is young and the model is standard and last_price is expensive then return value = average
value_rule11 = numpy.fmin(age_level_lo, numpy.fmin(model_level_lo, lprice_level_hi))
value_activation_mid4 = numpy.fmin(value_rule11, value_mid)# put the data in the average section of value

#if age is young and the model is pro and last_price is expensive then return value = average
value_rule12 = numpy.fmin(age_level_lo, numpy.fmin(model_level_mid, lprice_level_hi))
value_activation_mid5 = numpy.fmin(value_rule12, value_mid)# put the data in the average section of value

#if age is young and the model is custom and last_price is expensive then return value = expensive
value_rule13 = numpy.fmin(age_level_lo, numpy.fmin(model_level_hi, lprice_level_hi))
value_activation_hi5 = numpy.fmin(value_rule13, value_hi)# put the data in the expensive section of value

#if age is young and the model is standard and last_price is average OR cheap then return value = cheap
value_rule14 = numpy.fmin(age_level_lo, numpy.fmin(model_level_lo, numpy.fmax(lprice_level_mid, lprice_level_lo)))
value_activation_lo1 = numpy.fmin(value_rule14, value_lo)# put the data in the cheap section of value

#if age is young and the model is pro and last_price is average OR cheap then return value = cheap
value_rule15 = numpy.fmin(age_level_lo, numpy.fmin(model_level_mid, numpy.fmax(lprice_level_mid, lprice_level_lo)))
value_activation_lo2 = numpy.fmin(value_rule15, value_lo)# put the data in the cheap section of value

#if age is young and the model is custom and last_price is average or cheap then return value = average
value_rule16 = numpy.fmin(age_level_lo, numpy.fmin(model_level_hi, numpy.fmax(lprice_level_mid, lprice_level_lo)))
value_activation_mid6 = numpy.fmin(value_rule16, value_mid)# put the data in the average section of value

value0 = numpy.zeros_like(value)

#lo 1-2, mid 1-6, expensive 1-5, rare 1-3
values_aggregated = numpy.fmax(value_activation_lo1,
                    numpy.fmax(value_activation_lo2,
                    numpy.fmax(value_activation_mid1,
                    numpy.fmax(value_activation_mid2,
                    numpy.fmax(value_activation_mid3,
                    numpy.fmax(value_activation_mid4,
                    numpy.fmax(value_activation_mid5,
                    numpy.fmax(value_activation_mid6,
                    numpy.fmax(value_activation_hi1,
                    numpy.fmax(value_activation_hi2,
                    numpy.fmax(value_activation_hi3,
                    numpy.fmax(value_activation_hi4,
                    numpy.fmax(value_activation_hi5,
                    numpy.fmax(value_activation_rare1,
                    numpy.fmax(value_activation_rare2, value_activation_rare3)))))))))))))))

value_final = mpjyky.defuzzify(value, values_aggregated)
value_activation = mpjyky.getMembership(values_aggregated, value, value_final)

#below is the plot for the first aggregation
value_plot.plot(value, value_lo, linewidth=0.5, linestyle='--', )
value_plot.plot(value, value_mid, linewidth=0.5, linestyle='--')
value_plot.plot(value, value_hi, linewidth=0.5, linestyle='--')
value_plot.plot(value, value_rare, linewidth=0.5, linestyle='--')
value_plot.fill_between(value, value0, values_aggregated, facecolor='Orange', alpha=0.7)
value_plot.plot([value_final, value_final], [0, value_activation], 'k', linewidth=1.5, alpha=0.9)
value_plot.set_title('Fair Market Value')

print("Fair value price: $", value_final)

fig.tight_layout()
plot.show()