#ifndef __CLASSIAS_LBFGS_H__
#define __CLASSIAS_LBFGS_H__

namespace classias
{

/**
 * A stub class for using L-BFGS library from a template class.
 */
class lbfgs_solver
{
public:
    /**
     * Constructs.
     */
    lbfgs_solver()
    {
    }

    int lbfgs_solve(
        const int n, double *x, double *ptr_fx, double epsilon, double c1
        );

    virtual double lbfgs_evaluate(
        const double *x, double *g, const int n, const double step
        ) = 0;

    virtual int lbfgs_progress(
        const double *x, const double *g, const double fx,
        const double xnorm, const double gnorm,
        const double step,
        int n, int k, int ls
        ) = 0;
};

};

#endif/*__CLASSIAS_LBFGS_H__*/
