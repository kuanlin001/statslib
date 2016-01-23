using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace statslib
{
    public interface RankedTestDataList
    {
        string GetTestGroup();
        double GetRankingValue();
        Dictionary<string, int> GetDuplicateCnts();
    }

    public interface RankedTestVisualControl
    {
        void LoadData(RankedTestDataList datalist);
    }

    public static class Stats
    {
        public static double CorrelationCoefficient(double[] X, double[] Y)
        {
            if (X.Length != Y.Length)
                throw new Exception("X and Y must have equal sample sizes");
            else if (X.Length < 3)
                return double.NaN;
            else
            {
                var xbar = Mean(X);
                var ybar = Mean(Y);
                double result = 0.0;
                for (int i = 0; i < X.Length; i++)
                    result = result + (X[i] - xbar) * (Y[i] - ybar);

                return (result / ((X.Length - 1) * Stdev(X) * Stdev(Y)));
            }
        }

        public static void LeastSquareFitLinearRegression(double[] X, double[] Y, out double slope, out double intercept)
        {
            if (X.Length != Y.Length)
                throw new Exception("X and Y must have equal sample sizes");
            else if (X.Length < 2)
            {
                slope = 0; intercept = 0;
            }
            else
            {
                double sumXY = 0, sumXsqr = 0;
                for (int i = 0; i < X.Length; i++)
                {
                    sumXY = sumXY + X[i] * Y[i];
                    sumXsqr = sumXsqr + Math.Pow(X[i], 2);
                }

                slope = (X.Length * sumXY - Sum(X) * Sum(Y)) / (X.Length * sumXsqr - Math.Pow(Sum(X), 2));
                intercept = (Sum(Y) - slope * (Sum(X))) / X.Length;
            }
        }

        public static double CorrelationProbValue(double r, double samplesize)
        {
            var df = samplesize - 2;
            var t_denom = Math.Sqrt((1 - (r * r)) / df);
            var t = r / t_denom;
            t = Math.Abs(t);
            var rt = t / Math.Sqrt(df);
            var fk = Math.Atan(rt);

            if(df == 1)
                return 1 - fk / (Math.PI /2);

            var ek = Math.Sin(fk);
            var dk = Math.Cos(fk);

            double z;
            if((df%2)==1)
                z= 1-(fk+ek*dk*zip(dk*dk,2,df-3,-1))/(Math.PI /2);
            else
                z= 1 - ek * zip(dk * dk, 1, df - 3, -1);

            if (z >= 0)
                z = z + 0.0000005;
            else
                z = z - 0.0000005;

            return z;
        }

        private static double zip(double q,double i,double j,double b)
        {
	        double zz=1;
	        var z=zz;
	        var k=i;
            while(k<=j) { zz=zz*q*k/(k-b); z=z+zz; k=k+2; }
            return z;
        }

        public static double Max(double[] input)
        {
            if (!(input.Length > 0))
                throw new Exception("No input number");

            double max = input[0];

            foreach (double num in input)
            {
                if (num > max)
                    max = num;
            }
            return max;
        }

        public static double MaxInRange(double[] input, double lowerBound, double upperBound)
        {

            Array.Sort(input);

            if (input[0] > upperBound || input[input.Length - 1] < lowerBound)
                throw new Exception("Data not in range!" + Environment.NewLine + "LowerBound: " + lowerBound.ToString() + Environment.NewLine + "UpperBound: " + upperBound.ToString()
                    + Environment.NewLine + "First Item: " + input[0] + Environment.NewLine + "Last Item " + input[input.Length - 1].ToString());

            if (input.Length == 1)
                return input[0];

            if (input[input.Length - 1] <= upperBound)
                return input[input.Length - 1];
            else
            {
                int i = 0;
                while (input[i] < upperBound)
                    i++;

                if (input[i] > upperBound && i > 0)
                    i -= 1;

                return input[i];
            }
        }

        public static double MinInRange(double[] input, double lowerBound, double upperBound)
        {
            Array.Sort(input);

            if (input[0] > upperBound || input[input.Length - 1] < lowerBound)
                throw new Exception("Data not in range!" + Environment.NewLine + "LowerBound: " + lowerBound.ToString() + Environment.NewLine + "UpperBound: " + upperBound.ToString()
                    + Environment.NewLine + "First Item: " + input[0] + Environment.NewLine + "Last Item " + input[input.Length - 1].ToString());

            if (input.Length == 1)
                return input[0];

            if (input[0] >= lowerBound)
                return input[0];
            else
            {
                int i = 0;
                while (input[i] < lowerBound)
                    i++;

                return input[i];
            }
        }

        public static double Min(double[] input)
        {
            if (!(input.Length > 0))
                throw new Exception("No input number");

            double min = input[0];

            foreach (double num in input)
            {
                if (num < min)
                    min = num;
            }
            return min;
        }

        public static double Sum(double[] input)
        {
            if (!(input.Length > 0))
                throw new Exception("No input number");

            double sum = 0;

            foreach (double num in input)
            {
                sum += num;
            }

            return sum;
        }

        public static double Mean(double[] input)
        {
            if (!(input.Length > 0))
                throw new Exception("No input number");

            return Sum(input) / input.Length;

        }

        public static double Stdev(double[] input)
        {
            if (!(input.Length > 0))
                throw new Exception("No input number");
            else if (input.Length == 1)
                return 0;

            double output = 0;
            double mean = Mean(input);

            foreach (double num in input)
            {
                output += Math.Pow(num - mean, 2.0);
            }

            return Math.Sqrt(output / (input.Length - 1));
        }

        public static double SumOfSqrs(double[] input)
        {
            double mean = Mean(input);
            double result = 0;

            foreach (double num in input)
                result += Math.Pow((mean - num), 2);

            return result;
        }

        public static double Percentile(double[] data, double percentile)
        {
            if (data.Length == 1)
                return data[0];

            double[] sortedData = new double[data.Length];
            data.CopyTo(sortedData, 0);
            Array.Sort(sortedData);

            if (percentile >= 100.0d) return sortedData[sortedData.Length - 1];

            double position = (double)(sortedData.Length + 1) * percentile / 100.0;
            double leftNumber = 0.0d, rightNumber = 0.0d;

            double n = percentile / 100.0d * (sortedData.Length - 1) + 1.0d;

            if (position >= 1)
            {
                leftNumber = sortedData[(int)System.Math.Floor(n) - 1];
                rightNumber = sortedData[(int)System.Math.Floor(n)];
            }
            else
            {
                leftNumber = sortedData[0]; // first data
                rightNumber = sortedData[1]; // first data
            }

            if (leftNumber == rightNumber)
                return leftNumber;
            else
            {
                double part = n - System.Math.Floor(n);
                return leftNumber + part * (rightNumber - leftNumber);
            }
        }

        public static double KSampleFtest(out double Fratio, double[][] inputs)
        {
            if (inputs.Length < 2)
            {
                Fratio = double.NaN;
                return double.NaN;
            }

            double[] groupMeans = new double[inputs.Length];

            var uniqueMeans = new List<double>();
            for (int i = 0; i < inputs.Length; i++)
            {
                groupMeans[i] = Mean(inputs[i]);
                if(!uniqueMeans.Contains(groupMeans[i]))
                    uniqueMeans.Add(groupMeans[i]);
            }

            if (uniqueMeans.Count < 2)
            {
                Fratio = 0;
                return 1;
            }

            uniqueMeans = null;

            var nums = new List<double>();
            foreach (double[] input in inputs)
            {
                foreach (double num in input)
                {
                    nums.Add(num);
                }
            }

            double KSampleMean = Mean(nums.ToArray());

            //Between-group sum of squares:
            double Sb = 0;
            for (int i = 0; i < inputs.Length; i++)
                Sb += inputs[i].Length * Math.Pow((groupMeans[i] - KSampleMean), 2);

            //DoF of Treatments (Between-group DoF):
            double Fb = (inputs.Length - 1);

            double avgGroupObs = 0;
            foreach (double[] input in inputs)
                avgGroupObs += input.Length;
            avgGroupObs = avgGroupObs / inputs.Length;
            //DoF of Error (Within-group DoF):
            double Fw = inputs.Length * (avgGroupObs - 1);

            //Between-group MeanSqr (Treatments)
            double Mb = Sb / Fb;

            //Within-group sum of squares (Error):
            double Sw = 0;
            foreach (double[] input in inputs)
                Sw += SumOfSqrs(input);
            double Ms = Sw / Fw;

            //Calculate F-ratio and ProbF:
            double F, P;

            if (Ms != 0)
            {
                F = Mb / Ms;
                P = 1 - Math.Round(probf(F, (int)Fb, (int)Fw), 6);
                //P = 1 - Math.Round(probf(9.3, 2, 15), 6);
            }
            else
            {
                F = Single.MaxValue;
                if (Mb < 0)
                    F *= -1;
                P = 0;
            }

            Fratio = F;
            return P;
        }

        public static double probf(double fin, int v1, int v2)
        {
            //System.Windows.Forms.MessageBox.Show(v1.ToString() + " " + v2.ToString());
            //{*translation of probf.rpl**********************}
            //{ reference:
            //  handbook of mathematical functions, edited by milton abramowitz and
            //  irene stegun 1965. page 946, sections 26.6.4, 26.6.12, and 26.6.8
            //
            //   probf handles the following cases...
            //          a: at least one df is even and less than dfmax.
            //          b: both df's odd, and their sum is less than dfmax.
            //          c: neither of the above, but one df is 1.
            //          d: everything else.
            //   known accuracy:
            //          cases a - c: at least 1e-10
            //          case  d: about 1e-04 }
            //
            // { the next 3 could be different on different computers but these
            //   are fine on the vax which has the fewest bits for exponents, so
            //   they will work (no overflow or underflow) elsewhere too. }

            const double eps = 1E-25;
            //{ smallest non-zero value to return  }
            //{ 1-eps is largest value to return   }
            const double big = 1E+25;
            //{ used to predict imminent overflows }
            const double maxexp = 80;
            //{ log of a very large number         }
            const int dfmax = 500;
            //{ cutoff for using normal approx.    }

            double f = 0;
            double x = 0;
            double x1 = 0;
            double q = 0;
            double xx = 0;
            double th = 0;
            double c2th = 0;
            double s2th = 0;
            double sth = 0;
            double cth = 0;
            double scth = 0;
            double p = 0;
            double a = 0;
            double t = 0;
            double b = 0;
            double f1 = 0;
            double f2 = 0;
            double cbrf = 0;
            int df1 = 0;
            int df2 = 0;
            int df12 = 0;
            int i = 0;
            int j = 0;
            int half_df2 = 0;
            bool even1 = false;
            bool even2 = false;
            bool flip = false;
            bool uselog = false;

            if ((fin <= 0))
            {
                p = 0;

            }
            else
            {
                f = fin;
                df1 = v1;
                df2 = v2;
                df12 = df1 + df2 - 2;

                //{  note that probf(df1,df2,f) = 1-probf(df2,df1,1/f):
                //   we sometimes want to use the latter form (i.e. 'flip' it
                //   for reasons of efficiency or stability:
                //   (i)  if only one df is even, make that df1
                //   (ii) otherwise, make df1 the smaller of the 2 df's }

                even1 = ((df1 % 2) == 0);
                even2 = ((df2 % 2) == 0);
                if ((even1 && !(even2)))
                {
                    flip = false;
                }
                else if ((even2 && !(even1)))
                {
                    flip = true;
                    even1 = true;
                    //// even2 := false;
                }
                else
                {
                    flip = (df2 < df1);
                }

                if ((flip))
                {
                    df1 = df2;
                    df2 = v1;
                    f = 1 / f;
                }

                if ((even1 && (df1 <= dfmax)))
                {
                    //{  a: even-even, even-odd or odd-even (flipped to even-odd)
                    //      use eqn. 26.6.4 if df1 is small enough.
                    //      in the loop below, if q is getting large, the loop is
                    //      prematurely exited (via the doexit) in favor of the
                    //      succeeding loop which propogates ln(q) instead of q. }

                    x = df2 / (df2 + df1 * f);
                    x1 = 1.0 - x;
                    q = 0.0;
                    uselog = false;
                    j = 0;
                    xx = df2 / 2.0 * Math.Log(x);

                    i = df1 - 2;
                    while ((i >= 1))
                    {
                        df12 = df12 - 2;
                        q = x1 * df12 / i * (1 + q);
                        if ((q > big))
                        {
                            q = Math.Log(q);
                            uselog = true;
                            j = i - 2;
                            i = 0;
                        }
                        i = i - 2;
                    }

                    if (uselog)
                    {
                        i = j;
                        while ((i >= 1))
                        {
                            df12 = df12 - 2;
                            q = q + Math.Log(x1 * df12 / i);
                            i = i - 2;
                        }
                        q = q + xx;
                        if ((Math.Abs(q) < maxexp))
                            q = Math.Exp(q);
                    }
                    else
                    {
                        if ((Math.Abs(xx) < maxexp))
                            xx = Math.Exp(xx);
                        q = xx * (1 + q);
                    }

                    p = 1 - q;

                }
                else if (((df1 + df2) <= dfmax))
                {
                    //{ b: odd-odd... use eqn. 26.6.8 if df1+df2 is small enough }

                    th = Math.Atan(Math.Sqrt(df1 * f / df2));
                    c2th = df2 / (df2 + f * df1);
                    s2th = f * df1 / (df2 + f * df1);
                    sth = Math.Sqrt(s2th);
                    cth = Math.Sqrt(c2th);
                    scth = sth * cth;
                    half_df2 = df2 / 2;

                    if ((df2 < 100))
                    {
                        //    {  computation of a: for larger df2, the call
                        //       to PROBT approximates the more exact
                        //       expression to at least 8 significant digits
                        //       (and 11 decimal places). }
                        a = 0;
                        if ((df2 > 1))
                        {
                            i = df2 - 2;
                            while ((i >= 2))
                            {
                                a = c2th * (i - 1) / i * (1 + a);
                                i = i - 2;
                            }
                            a = scth * (1 + a);
                        }
                        a = (a + th) / (Math.PI / 2);
                    }
                    else
                    {
                        t = Math.Sqrt(f * df1);
                        a = probt(t, df2) * 2 - 1;
                    }

                    //{   computation of b: loop 1 does the terms
                    //    involving powers of sin(theta). loop 2
                    //    does the factorials and the power of
                    //    cos(theta).  start in loop 1a. if b gets
                    //    too big, leave loop 1a (via doexit) and
                    //    let loop 1b propogate ln(b). in this
                    //    case, loop 2a propogates ln(b) until it
                    //    is back down to reasonable size and loop
                    //    2b can finish up. }

                    uselog = false;
                    j = 0;
                    b = 0;

                    if ((df1 > 1))
                    {
                        i = df1 - 2;
                        //{ begin loop 1a }
                        while ((i >= 2))
                        {
                            df12 = df12 - 2;
                            b = s2th * df12 / i * (1 + b);
                            if ((b > big))
                            {
                                b = Math.Log(b);
                                uselog = true;
                                j = i - 2;
                                break; // TODO: might not be correct. Was : Exit Do
                                //{ exit 1a if log }
                            }
                            i = i - 2;
                        }
                        //{ end 1a normal }

                        if (uselog)
                        {
                            i = j;
                            //{ begin loop 1b }
                            while ((i >= 2))
                            {
                                df12 = df12 - 2;
                                b = Math.Log(s2th * df12 / i) + b;
                                i = i - 2;
                            }
                            //{ end of loop 1b }
                            b = Math.Log(scth) + b;
                        }
                        else
                        {
                            b = scth * (b + 1.0);
                        }

                        if ((df2 > 1))
                        {
                            j = 1;
                            if (uselog)
                            {
                                j = df2;
                                i = 1;
                                //{loop 2a}
                                while ((i <= half_df2))
                                {
                                    b = b + Math.Log(i / (i - 0.5) * c2th);
                                    if ((Math.Abs(b) <= maxexp))
                                    {
                                        j = i + 1;
                                        //// uselog := false;
                                        break; // TODO: might not be correct. Was : Exit Do
                                        //{ exit loop 2a }
                                    }
                                    i = i + 1;
                                }
                                b = Math.Exp(b);
                            }
                        }

                        //{ begin loop 2b }
                        for (i = j; i <= half_df2; i++)
                        {
                            b = i / (i - 0.5) * c2th * b;
                        }

                    }
                    b = b / (Math.PI / 2);
                    p = a - b;

                }
                else if ((df1 == 1))
                {
                    //{ c: large df, but df1 or df2 is 1. use equation 26.6.10 }
                    t = Math.Sqrt(f);
                    p = probt(t, df2) * 2.0 - 1.0;

                }
                else if ((df2 == 1))
                {
                    t = Math.Sqrt(1.0 / f);
                    flip = !(flip);
                    p = probt(t, df1) * 2.0 - 1.0;

                }
                else
                {
                    //{ d: all else, use the normal approximation, eq. 26.6.15 }
                    f1 = 2.0 / (9.0 * df1);
                    f2 = 2.0 / (9.0 * df2);
                    cbrf = Math.Exp(Math.Log(f) * (1.0 / 3.0));
                    x = (cbrf * (1.0 - f2) - (1.0 - f1));
                    x = x / Math.Sqrt(f1 + cbrf * cbrf * f2);
                    p = probnorm(x, false);
                }

                if (flip)
                    p = 1.0 - p;
                if ((p < eps))
                    p = 0.0;
                if ((p > (1.0 - eps)))
                    p = 1.0;
            }

            return p;
        }

        public static double probnorm(double xin, bool approx)
        {
            //{*translation of probnorm.rpl*******************}
            //   {  return integral from minus infinity to xin of n(0,1) distribution.
            //          if approx is true, return approximation (accuracy 7.5e-08)
            //          if approx is false, use exact evaluation (accuracy at least 1e-10)
            //          references:
            //          1. approximation: handbook of mathematical functions, edited by
            //          milton abramowitz and irene stegun, 1965. eqn 26.2.17.
            //          2. exact expression: - cacm algorithm 304 (hill and joyce) cacm 10:6
            //          (1967), pp. 374-375. this is an implementation of eqn 26.2.11 and
            //          26.2.14 of abromowitz and stegun. (translated from algol to rpl). }

            double r2pi = 0;
            double n = 0;
            double x = 0;
            double y = 0;
            double x2 = 0;
            double t1 = 0;
            double t2 = 0;
            double t3 = 0;
            double t4 = 0;
            double t5 = 0;
            double cutoff = 0;
            double m = 0;
            double t = 0;
            double q1 = 0;
            double q2 = 0;
            double p1 = 0;
            double p2 = 0;
            double s = 0;
            bool over = false;
            bool flag = false;

            if ((xin == 0.0))
                return 0.5;

            r2pi = Math.Sqrt(2 * Math.PI);
            x = Math.Abs(xin);
            x2 = x * x;
            y = Math.Exp(-0.5 * x2) / r2pi;
            over = (xin > 0);
            n = y / x;

            if ((!(over) && ((1 - n) == 1)))
                return 0;
            //{ close to zero }

            if ((over && (n == 0)))
                return 1;
            //{ close to one }

            //{ use approximation }
            if (approx)
            {
                t1 = 1 / (1 + 0.2316419 * x);
                t2 = t1 * t1;
                t3 = t2 * t1;
                t4 = t2 * t2;
                t5 = t3 * t2;
                t = y * (0.31938153 * t1 - 0.356563782 * t2 + 1.781477937 * t3 - 1.821255978 * t4 + 1.330274429 * t5);
                if (over)
                    t = 1 - t;
                return t;
            }

            //{ use cacm 304 }

            if (over)
                cutoff = 3.5;
            else
                cutoff = 2.32;

            //{ 26.2.14 }
            if ((x > cutoff))
            {
                q1 = x;
                p2 = y * x;
                p1 = y;
                q2 = x2 + 1;
                m = n;
                t = p2 / q2;
                if (!(over))
                {
                    m = 1 - m;
                    t = 1 - t;
                }
                n = 1;
                flag = true;
                while (flag)
                {
                    n = n + 1;
                    s = x * p2 + n * p1;
                    p1 = p2;
                    p2 = s;
                    s = x * q2 + n * q1;
                    q1 = q2;
                    q2 = s;
                    s = m;
                    m = t;
                    if (over)
                        t = 1 - p2 / q2;
                    else
                        t = p2 / q2;
                    flag = !((m == t) || (s == t));
                    if (((q2 >= 1E+20) && flag))
                    {
                        q2 = q2 / p1;
                        q1 = q1 / p1;
                        p2 = p2 / p1;
                        p1 = 1;
                    }
                }

                return t;
            }

            //{ 26.2.11 }
            x = y * x;
            s = x;
            t = 0;
            n = 1;
            while ((s != t))
            {
                n = n + 2;
                t = s;
                x = x * x2 / n;
                s = s + x;
            }
            if (over)
                return 0.5 + s;
            else
                return 0.5 - s;
        }

        public static double probt(double tstat, double df)
        {
            //{*translation of probt.rpl*****************}
            //{ returns a (lower-tail) significance level for t with df degrees of freedom.
            //         reference: hill, "Algorithm 395 Student's t-Distribution",
            //         cacm 13,10, october, 1970, pp 617-619.
            //         this is a direct translation of hill's algol routine into rpl.
            //         accuracy:
            //         integer df's: 11 decimal places and 8 significant digits.
            //         non-integer : over 6 places for df > 4.3.
            //                       2 places for df approx. 1 }

            double n = 0;
            double t = 0;
            double y = 0;
            double a = 0;
            double b = 0;
            double z = 0;
            double student = 0;
            double j = 0;
            bool even = false;

            if ((df < 1))
                throw new Exception("Zero or negative degrees of freedom in ProbT");

            if ((tstat == 0))
                return 0.5;

            n = df;
            even = ((n % 2) == 0);
            t = tstat * tstat;
            y = t / n;
            b = 1 + y;
            z = 1;

            if (((n > Math.Truncate(n)) || ((n >= 20.0) && (t < n)) || (n > 200.0)))
            {
                //{ asymptotic series for large or noninteger n }
                if ((y > 1E-06))
                    y = Math.Log(b);
                a = n - 0.5;
                b = 48 * a * a;
                y = a * y;
                y = (((((-0.4 * y - 3.3) * y - 24) * y - 85.5) / (0.8 * y * y + 100 + b) + y + 3) / b + 1) * Math.Sqrt(y);

                student = probnorm(-y, false);

            }
            else
            {
                if (((n >= 20) || (t >= 4)))
                {
                    //{ Tail series expansion for large t-values }
                    a = Math.Sqrt(b);
                    y = a * n;
                    j = 0.0;
                    while ((a != z))
                    {
                        j = j + 2;
                        z = a;
                        y = y * (j - 1) / (b * j);
                        a = a + y / (n + j);
                    }
                    n = n + 2;
                    z = 0;
                    y = 0;
                    a = -a;
                }
                else
                {
                    //{ nested summation of "cosine series" }
                    y = Math.Sqrt(y);
                    a = y;
                    if ((n == 1))
                        a = 0;
                }

                j = n - 2;
                while ((j >= 2))
                {
                    a = (j - 1) / (b * j) * a + y;
                    j = j - 2;
                }

                if (even)
                    a = a / Math.Sqrt(b);
                else
                    a = (Math.Atan(y) + a / b) / (Math.PI / 2);
                student = (z - a) / 2;

            }

            if ((tstat > 0))
                student = 1 - student;
            return student;
        }
    }
}
