#ifndef __OPTION_H__
#define __OPTION_H__

#include <vector>
#include <string>

class option
{
public:
    typedef std::vector<std::string> files_type;

    enum {
        TYPE_NONE = 0,          /// Default type.
        TYPE_MULTICLASS,        /// Classification.
        TYPE_SELECTOR,          /// Selection.
        TYPE_RANKER,            /// Ranker.
        TYPE_BICLASS,           /// Logistic regression.
    };

    enum {
        MODE_NONE = 0,          /// No mode.
        MODE_TRAIN,             /// Training mode.
        MODE_TAG,               /// Tagging mode.
        MODE_HELP,              /// Usage mode.
    };

    std::istream&   is;
    std::ostream&   os;
    std::ostream&   es;

    int         type;
    bool        unify;
    int         mode;
    int         split;
    int         holdout;
    bool        cross_validation;

    files_type  files;
    std::string model;

    std::string algorithm;
    double      sigma;
    int         maxiter;

    option(
        std::istream& _is = std::cin,
        std::ostream& _os = std::cout,
        std::ostream& _es = std::cerr
        ) : is(_is), os(_os), es(_es),
        type(TYPE_NONE), unify(false), mode(MODE_NONE),
        split(0), holdout(-1), cross_validation(false),
        model(""),
        algorithm("L2"), sigma(1.),
        maxiter(1000)
    {
    }
};

#endif/*__OPTION_H__*/
