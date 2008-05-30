#include <classias/base.h>
#include <classias/lbfgs.h>
#include <lbfgs.h>

namespace classias
{

static double
__lbfgs_evaluate(
    void *inst, const double *x, double *g, const int n, const double step)
{
    lbfgs_solver* pt = reinterpret_cast<lbfgs_solver*>(inst);
    return pt->lbfgs_evaluate(x, g, n, step);
}

static int
__lbfgs_progress(
    void *inst,
    const double *x, const double *g, const double fx,
    const double xnorm, const double gnorm,
    const double step,
    int n, int k, int ls
    )
{
    lbfgs_solver* pt = reinterpret_cast<lbfgs_solver*>(inst);
    int ret = pt->lbfgs_progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    if (ret != 0) {
        return LBFGSERR_MAXIMUMITERATION;
    }
    return 0;
}

int lbfgs_solver::lbfgs_solve(const int n, double *x, double *ptr_fx, double epsilon, double c1)
{
    // Set L-BFGS parameters.
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.epsilon = epsilon;
    param.orthantwise_c = c1;

    // Call L-BFGS routine.
    return lbfgs(
        n,
        x,
        ptr_fx,
        __lbfgs_evaluate,
        __lbfgs_progress,
        this,
        &param
        );
}

};
