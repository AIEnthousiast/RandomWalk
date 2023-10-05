#include <stdio.h>
#include <stdlib.h>

float calculateProbabilyByRecurrence(int n, int k)
{
    if (n < abs(k) || (n == 1 && abs(k) > 1 ))
    {
        return 0;
    }
    else if ((n == 1 && abs(k) == 1))
    {
        return 0.5;
    }
    else
    {
        return 0.5 * calculateProbabilyByRecurrence(n-1,k-1) + 0.5 * calculateProbabilyByRecurrence(n-1,k+1);
    }

}


int main(int argc, char ** argv)
{

    printf("%d\n",calculateProbabilyByRecurrence(100,2));
    return EXIT_SUCCESS;
}