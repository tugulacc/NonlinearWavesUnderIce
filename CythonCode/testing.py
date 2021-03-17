import timeit
from runCode import main

main()

#cy=timeit.timeit(mainFunc,setup = 'import main_cy', number = 1)
#py=timeit.timeit('main.test()',setup = 'import main', number = 0)

#print(cy)
#print("Cython is {}x faster than Python".format(py/cy))