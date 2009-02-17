/*
 *		Data I/O for binary-class classification.
 *
 * Copyright (c) 2008, Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Northwestern University, University of Tokyo,
 *       nor the names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* $Id$ */

#ifdef  HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <iostream>
#include <string>

#include <classias/base.h>
#include <classias/logress.h>

#include "option.h"
#include "tokenize.h"
#include "train.h"

/*
<line>          ::= <comment> | <instance> | <br>
<comment>       ::= "#" <string> <br>
<instance>      ::= <class> ("\t" <feature>)+ <br>
<class>         ::= "-1" | "1" | "+1"
<feature>       ::= <name> [ ":" <weight> ]
<name>          ::= <string>
<weight>        ::= <numeric>
<br>            ::= "\n"
*/

template <
    class instance_type,
    class features_quark_type
>
static void
read_line(
    const std::string& line,
    instance_type& instance,
    features_quark_type& features,
    const std::string& comment,
    const option& opt,
    int lines = 0
    )
{
    double value;
    std::string name;

    // Split the line with tab characters.
    tokenizer values(line, '\t');
    tokenizer::iterator itv = values.begin();
    if (itv == values.end()) {
        throw invalid_data("no field found in the line", lines);
    }

    // Make sure that the first token (class) is not empty.
    if (itv->empty()) {
        throw invalid_data("an empty label found", lines);
    }

    // Parse the instance label.
    get_name_value(*itv, name, value);

    // Set the class label of this instance.
    if (name == "+1" || name == "1") {
        instance.set_truth(true);
    } else if (name == "-1") {
        instance.set_truth(false);
    } else {
        throw invalid_data("a class label must be either '+1' or '-1'", lines);
    }

    // Set the instance weight.
    instance.set_weight(value);

    // Set the comment of this instance.
    if (!comment.empty()) {
        instance.set_comment(comment);
    }

    // Set featuress for the instance.
    for (++itv;itv != values.end();++itv) {
        if (!itv->empty()) {
            get_name_value(*itv, name, value);
            instance.append(features(name), value);
        }
    }
}

template <
    class data_type
>
static void
read_stream(
    std::istream& is,
    data_type& data,
    const option& opt,
    int group = 0
    )
{
    int lines = 0;
    std::string comment;
    typedef typename data_type::instance_type instance_type;
    typedef typename data_type::feature_type feature_type;
    typedef typename data_type::iterator iterator;

    for (;;) {
        // Read a line.
        std::string line;
        std::getline(is, line);
        if (is.eof()) {
            break;
        }
        ++lines;

        // Skip an empty line.
        if (line.empty()) {
            continue;
        }

        // Skip a comment line.
        if (line.compare(0, 1, "#") == 0) {
            comment += line;
            continue;
        }

        // A new candidate.
        read_line(line, data.new_element(), data.features, comment, opt, lines);
        comment.clear();
    }

    // Set the end index of the user features.
    data.set_user_feature_end(data.features.size());

    // Generate a bias feature if necessary.
    if (opt.generate_bias) {
        // Allocate a bias feature.
        feature_type bf = data.features("@bias");

        // Insert the bias feature to each instance.
        for (iterator it = data.begin();it != data.end();++it) {
            it->append(bf, 1.0);
        }
    }
}

int binary_train(option& opt)
{
    // Branches for training algorithms.
    if (opt.algorithm == "logress") {
        return train<
            classias::bdata,
            classias::trainer_logress<classias::bdata, double>
        >(opt);
    } else {
        throw invalid_algorithm(opt.algorithm);
    }
}

bool binary_usage(option& opt)
{
    if (opt.algorithm == "logress") {
        classias::trainer_logress<classias::bdata, double> tr;
        tr.params().help(opt.os);
        return true;
    }
    return true;
}
