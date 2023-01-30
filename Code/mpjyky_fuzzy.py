import numpy
import matplotlib.pyplot as plot

'''
Membership functions:
1. Gaussian
2. Triangle
3. Trapezoidal
'''
failCondition = "FAIL"

'''
gaussian membership function
'''
def gaussian(domain, deviation, median):
    return numpy.exp(-((domain - median) ** 2.) / (2*(float(deviation) ** 2.)))

'''
triangle membership function
'''
def triangle(domain, vectors):
    if len(vectors) != 3:
        return failCondition
    
    first = vectors[0] 
    second = vectors[1] 
    third = vectors[2]
    
    if first > second or second > third:
        return failCondition
    
    result = numpy.zeros(len(domain))

    if first != second:
        temp = numpy.nonzero(numpy.logical_and(first < domain, domain < second))[0]
        result[temp] = (domain[temp] - first)/float(second - first)
    
    if second != third:
        temp = numpy.nonzero(numpy.logical_and(second < domain, domain < third))[0]
        result[temp] = (third - domain[temp])/float(third - second)
    
    temp = numpy.nonzero(domain == second)
    result[temp] = 1
    return result

'''
trapezoid membership function
'''
def trapezoid(domain, vectors):
    if len(vectors) != 4:
        return failCondition

    first = vectors[0]
    second = vectors[1] 
    third = vectors[2] 
    fourth = vectors[3]

    if first > second or second > third or third > fourth:
        return failCondition
    
    result = numpy.ones(len(domain))

    temp = numpy.nonzero(domain <= second)[0]
    result[temp] = triangle(domain[temp], numpy.r_[first, second, second])
    if type(result[temp]) is str:
        return failCondition
    
    temp = numpy.nonzero(domain >= third)[0]
    result[temp] = triangle(domain[temp], numpy.r_[third, third, fourth])
    if type(result[temp]) is str:
        return failCondition
    
    temp = numpy.nonzero(domain < first)[0]
    result[temp] = numpy.zeros(len(temp))

    temp = numpy.nonzero(domain > fourth)[0]
    result[temp] = numpy.zeros(len(temp))

    return result

"""
Centroid Defuzzify Function
"""
def defuzzify(input, membership):
    return sum(membership * input) / sum(membership)

'''
Gives us the value of the membership within our domain
'''
def getMembership(value, membership, data):
    first_bound = membership[membership >= data][0] #first element
    last_bound = membership[membership <= data][-1] #last element

    #gives us an index for the last bound in the membership function.
    i = numpy.nonzero(membership == first_bound)[0][0]
    #gives us an index for the last bound in the membership function
    j = numpy.nonzero(membership == last_bound)[0][0]

    first_value = value[i]
    last_value = value[j]

    if first_bound == last_bound: return value[j]
    
    return (float(first_value - last_value)/float(first_bound - last_bound)) * (data - last_bound) + last_value