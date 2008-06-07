#ifndef __OPTION_H__
#define __OPTION_H__

#include <vector>
#include <string>

class option
{
public:
    typedef std::vector<std::string> files_type;
    typedef std::vector<std::string> params_type;

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

    files_type  files;

    int         mode;

    int         type;
    std::string model;

    std::string algorithm;
    params_type params;
    int         split;
    int         holdout;
    bool        cross_validation;

    option(
        std::istream& _is = std::cin,
        std::ostream& _os = std::cout,
        std::ostream& _es = std::cerr
        ) : is(_is), os(_os), es(_es),
        mode(MODE_NONE), type(TYPE_NONE), model(""),
        algorithm("MaxEnt"),        
        split(0), holdout(-1), cross_validation(false)
    {
    }
};

#endif/*__OPTION_H__*/
