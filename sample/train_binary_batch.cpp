/*
 *		A sample program for training a binary classifier.
 *
 * Copyright (c) 2008,2009 Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
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

#include <iostream>
#include <string>
#include <vector>

#include <classias/classias.h>
#include <classias/classify/linear/binary.h>
#include <classias/train/pegasos.h>
#include <classias/train/online_scheduler.h>

#include "strsplit.h"   // necessary for strsplit() and get_id_value().

typedef classias::train::online_scheduler_binary<
    classias::bdata,
    classias::train::pegasos_binary<
        classias::classify::linear_binary_hinge<classias::weight_vector>
        >
    >
    trainer_type;


int main(int argc, char *argv[])
{
    int max_fid = -1;
    classias::bdata data;
    std::istream& is = std::cin;
    std::ostream& os = std::cout;
    std::ostream& es = std::cerr;

    // Read a data set from STDIN.
    for (;;) {
        // Read a line.
        std::string line;
        std::getline(is, line);
        if (is.eof()) {
            break;
        }

        // Split the line into fields with space characters.
        std::vector<std::string> fields;
        strsplit(fields, line);

        // The line must have at least a label and a feature.
        if (fields.size() > 2) {
            // Create a new instance in the data set.
            classias::binstance& inst = data.new_element();
            // Set the boolean label for the instance.
            inst.set_label(fields[0] != "-1");

            // Loop over the rest of fields.
            for (size_t i = 1;i < fields.size();++i) {
                // Split the field into a feature identifier and value.
                int fid;
                double value;
                get_id_value(fields[i], fid, value, ':');

                // Store the maximum number of feature identifiers.
                if (max_fid < fid) {
                    max_fid = fid;
                }

                // Append the feature to the instance.
                inst.append(fid, value);
            }
        }
    }

    // Do not forget to set the number of features in the data set.
    data.set_num_features(max_fid+1);

    // Create an instance of the training algorithm.
    trainer_type tr;

    // Set some parameters for the training algorithm.
    tr.params().set("c", 1.0);
    tr.params().set("max_iterations", 10);

    // Start training; progress report will be shown in es (STDERR).
    tr.train(data, es);

    // Output the model.
    const classias::weight_vector& w = tr.model();
    for (int i = 0;i <= max_fid;++i) {
        // Feature ID and its weight.
        os << i << '\t' << w[i] << std::endl;
    }

    return 0;
}
