#include "stdio.h"
#include "math.h"

static thread_local double tlvar;
//following line is needed so get_value() is not inlined by compiler
double get_value() __attribute__ ((noinline));
double get_value()
{
    return tlvar;
}

int test()
{
    int i;
    double f=0.0;
    tlvar = 1.0;
    for(i=0; i<1000000000; i++)
    {
        f += sqrt(get_value());
    }
    printf("f = %f\n", f);
    return 1;
}