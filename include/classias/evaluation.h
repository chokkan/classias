/*
 *		Utilities for evaluation.
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

#ifndef __CLASSIAS_EVALUATION_H__
#define __CLASSIAS_EVALUATION_H__

#include <iomanip>
#include <vector>

namespace classias
{

class accuracy
{
protected:
    int c;
    int n;

public:
    accuracy() : n(0), c(0)
    {
    }

    virtual ~accuracy()
    {
    }

    inline void set(bool b)
    {
        c += static_cast<int>(b);
        ++n;
    }

    void output(std::ostream& os) const
    {
        double acc = (0 < n ? c / (double)n : 0);
        os << "Accuracy: " <<
            std::fixed << std::setprecision(4) << acc <<
            std::setprecision(6) << 
            " (" << c << "/" << n << ")" << std::endl;
        os.unsetf(std::ios::fixed);
    }
};

class confusion_matrix
{
protected:
    int n;
    int *matrix;

public:
    confusion_matrix(int N) : n(N)
    {
        matrix = new int[N * N];
        clear();
    }

    virtual ~confusion_matrix()
    {
        delete[] matrix;
    }

    inline int& operator() (int x, int y)
    {
        return matrix[x + y * n];
    }

    inline const int& operator() (int x, int y) const
    {
        return matrix[x + y * n];
    }

    inline int xsum(int x) const
    {
        int sum = 0;
        for (int y = 0;y < n;++y) {
            sum += this->operator()(x, y);
        }
        return sum;
    }

    inline int ysum(int y) const
    {
        int sum = 0;
        for (int x = 0;x < n;++x) {
            sum += this->operator()(x, y);
        }
        return sum;
    }

    void clear()
    {
        for (int x = 0;x < n;++x) {
            for (int y = 0;y < n;++y) {
                this->operator()(x, y) = 0;
            }
        }
    }

    inline int match(int l) const
    {
        return this->operator()(l, l);
    }

    inline int correct() const
    {
        int sum = 0;
        for (int l = 0;l < n;++l) {
            sum += this->match(l);
        }
        return sum;
    }

    inline int reference(int x) const
    {
        int sum = 0;
        for (int y = 0;y < n;++y) {
            sum += this->operator()(x, y);
        }
        return sum;
    }

    inline int prediction(int y) const
    {
        int sum = 0;
        for (int x = 0;x < n;++x) {
            sum += this->operator()(x, y);
        }
        return sum;
    }

    inline int total() const
    {
        int sum = 0;
        for (int x = 0;x < n;++x) {
            for (int y = 0;y < n;++y) {
                sum += this->operator()(x, y);
            }
        }
        return sum;
    }

    template <typename value_type>
    static inline double divide(value_type a, value_type b)
    {
        return (b != 0) ? (a / (double)b) : 0.;
    }

    template <class positive_iterator_type>
    void compute_micro(
        positive_iterator_type pb,
        positive_iterator_type pe,
        int& num_match,
        int& num_reference,
        int& num_prediction
        ) const
    {
        num_match = 0;
        num_reference = 0;
        num_prediction = 0;

        for (positive_iterator_type it = pb;it != pe;++it) {
            num_match += this->match(*it);
            num_reference += this->xsum(*it);
            num_prediction += this->ysum(*it);
        }
    }

    template <class positive_iterator_type>
    void output_micro(
        std::ostream& os,
        positive_iterator_type pb,
        positive_iterator_type pe
        ) const
    {
        int num_match = 0;
        int num_reference = 0;
        int num_prediction = 0;
        compute_micro<positive_iterator_type>(
            pb, pe, num_match, num_reference, num_prediction);

        double precision = divide(num_match, num_prediction);
        double recall = divide(num_match, num_reference);
        double f1score = divide(2 * precision * recall, precision + recall);

        os << "Micro P, R, F1: " <<
            std::fixed << std::setprecision(4) << precision <<
            std::setprecision(6) << 
            " (" << num_match << "/" << num_prediction << ")" << ", " <<
            std::fixed << std::setprecision(4) << recall <<
            std::setprecision(6) <<
            " (" << num_match << "/" << num_reference << ")" << ", " <<
            std::fixed << std::setprecision(4) << f1score << std::endl;
        os << std::setprecision(6);
        os.unsetf(std::ios::fixed);
    }

    template <class positive_iterator_type>
    void compute_macro(
        positive_iterator_type pb,
        positive_iterator_type pe,
        double& precision,
        double& recall,
        double& f1
        ) const
    {
        int n = 0;
        int num_match = 0;
        int num_reference = 0;
        int num_prediction = 0;
        precision = recall = f1 = 0.;

        for (positive_iterator_type it = pb;it != pe;++it) {
            num_match = this->match(*it);
            num_reference = this->xsum(*it);
            num_prediction = this->ysum(*it);
            double p = divide(num_match, num_prediction);
            double r = divide(num_match, num_reference);
            double f = divide(2 * p * r, p + r);
            precision += p;
            recall += r;
            f1 += f;
            ++n;
        }

        precision /= n;
        recall /= n;
        f1 /= n;
    }

    template <class positive_iterator_type>
    void output_macro(
        std::ostream& os,
        positive_iterator_type pb,
        positive_iterator_type pe
        ) const
    {
        double precision = 0., recall = 0., f1 = 0.;
        compute_macro(pb, pe, precision, recall, f1);
        os << "Macro P, R, F1: " << 
            std::fixed << std::setprecision(4) << precision << ", " <<
            std::fixed << std::setprecision(4) << recall << ", " <<
            std::fixed << std::setprecision(4) << f1 << std::endl;
        os << std::setprecision(6);
        os.unsetf(std::ios::fixed);
    }
};

};

#endif/*__CLASSIAS_EVALUATION_H__*/
