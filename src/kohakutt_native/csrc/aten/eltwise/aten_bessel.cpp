// aten Bessel-type special functions for tt: bessel_j0 / j1 / y0 / y1, modified_bessel
// i0 / i1 / k0 / k1, scaled_modified_bessel_k0 / k1, spherical_bessel_j0, airy_ai
// (.out forms). PyTorch's (Cephes) rational and Chebyshev approximations, in fp32;
// each branch evaluated on inputs clamped into its range, then selected.
#include <torch/library.h>

#include <limits>
#include <vector>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        using Coeffs = std::vector<double>;

        constexpr double kInf = std::numeric_limits<double>::infinity();
        constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
        constexpr double kTwoOverPi = 0.636619772367581343075535053490057448;
        constexpr double kSqrtTwoOverPi = 0.797884560802865355879892119868763737;
        constexpr double kQuarterPi = 0.785398163397448309615660845819875721;
        constexpr double kThreeQuarterPi = 2.356194490192344928846982537459627163;

        // c[0] z^(n-1) + ... + c[n-1] (Horner).
        at::Tensor horner(const at::Tensor& z, const Coeffs& c) {
            auto acc = at::full_like(z, c[0]);
            for (size_t i = 1; i < c.size(); ++i) {
                acc = acc * z + c[i];
            }
            return acc;
        }

        // Cephes chbevl: Chebyshev series in t over the first `len` coefficients.
        at::Tensor chbevl(const at::Tensor& t, const Coeffs& c, size_t len) {
            auto a = at::full_like(t, c[0]);
            auto p = at::zeros_like(t), q = at::zeros_like(t);
            for (size_t i = 1; i < len; ++i) {
                p = q;
                q = a;
                a = t * q - p + c[i];
            }
            return (a - p) * 0.5;
        }

        // `x` where `keep`, else `fill`.
        at::Tensor clamp_to(const at::Tensor& x, const at::Tensor& keep, double fill) {
            return at::where(keep, x, fill);
        }

        // ---- J0 / J1 / Y0 / Y1 ----

        const Coeffs kPP0 = {
            7.96936729297347051624e-04,
            8.28352392107440799803e-02,
            1.23953371646414299388e+00,
            5.44725003058768775090e+00,
            8.74716500199817011941e+00,
            5.30324038235394892183e+00,
            9.99999999999999997821e-01
        };
        const Coeffs kPQ0 = {
            9.24408810558863637013e-04,
            8.56288474354474431428e-02,
            1.25352743901058953537e+00,
            5.47097740330417105182e+00,
            8.76190883237069594232e+00,
            5.30605288235394617618e+00,
            1.00000000000000000218e+00
        };
        const Coeffs kQP0 = {
            -1.13663838898469149931e-02,
            -1.28252718670509318512e+00,
            -1.95539544257735972385e+01,
            -9.32060152123768231369e+01,
            -1.77681167980488050595e+02,
            -1.47077505154951170175e+02,
            -5.14105326766599330220e+01,
            -6.05014350600728481186e+00
        };
        const Coeffs kQQ0 = {
            6.43178256118178023184e+01,
            8.56430025976980587198e+02,
            3.88240183605401609683e+03,
            7.24046774195652478189e+03,
            5.93072701187316984827e+03,
            2.06209331660327847417e+03,
            2.42005740240291393179e+02
        };
        const Coeffs kPP1 = {
            7.62125616208173112003e-04,
            7.31397056940917570436e-02,
            1.12719608129684925192e+00,
            5.11207951146807644818e+00,
            8.42404590141772420927e+00,
            5.21451598682361504063e+00,
            1.00000000000000000254e+00
        };
        const Coeffs kPQ1 = {
            5.71323128072548699714e-04,
            6.88455908754495404082e-02,
            1.10514232634061696926e+00,
            5.07386386128601488557e+00,
            8.39985554327604159757e+00,
            5.20982848682361821619e+00,
            9.99999999999999997461e-01
        };
        const Coeffs kQP1 = {
            5.10862594750176621635e-02,
            4.98213872951233449420e+00,
            7.58238284132545283818e+01,
            3.66779609360150777800e+02,
            7.10856304998926107277e+02,
            5.97489612400613639965e+02,
            2.11688757100572135698e+02,
            2.52070205858023719784e+01
        };
        const Coeffs kQQ1 = {
            7.42373277035675149943e+01,
            1.05644886038262816351e+03,
            4.98641058337653607651e+03,
            9.56231892404756170795e+03,
            7.99704160447350683650e+03,
            2.82619278517639096600e+03,
            3.36093607810698293419e+02
        };

        // Large-argument form (x > 5) of J0 / Y0 (order 0) or J1 / Y1 (order 1):
        // c / sqrt(x) (P cos(x - phase) - 5/x Q sin(x - phase)) for J, sin / cos
        // swapped (and +) for Y.
        at::Tensor bessel_far(const at::Tensor& x, int order, bool y) {
            const auto& pp = order == 0 ? kPP0 : kPP1;
            const auto& pq = order == 0 ? kPQ0 : kPQ1;
            const auto& qp = order == 0 ? kQP0 : kQP1;
            const auto& qq = order == 0 ? kQQ0 : kQQ1;
            auto w = 25.0 / (x * x);
            auto p = horner(w, pp) / horner(w, pq);
            auto q = horner(w, qp) / horner(w, qq) * (5.0 / x);
            auto angle = x - (order == 0 ? kQuarterPi : kThreeQuarterPi);
            auto s = at::sin(angle), c = at::cos(angle);
            auto sum = y ? p * s + q * c : p * c - q * s;
            return sum * kSqrtTwoOverPi / at::sqrt(x);
        }

        // J0(x), x >= 0.
        at::Tensor j0_nonneg(const at::Tensor& x) {
            static const Coeffs rp = {
                -4.79443220978201773821e+09,
                1.95617491946556577543e+12,
                -2.49248344360967716204e+14,
                9.70862251047306323952e+15
            };
            static const Coeffs rq = {
                4.99563147152651017219e+02,
                1.73785401676374683123e+05,
                4.84409658339962045305e+07,
                1.11855537045356834862e+10,
                2.11277520115489217587e+12,
                3.10518229857422583814e+14,
                3.18121955943204943306e+16,
                1.71086294081043136091e+18
            };
            auto near_mask = at::le(x, 5.0);
            auto xn = clamp_to(x, near_mask, 1.0);
            auto z = xn * xn;
            auto near =
                ((z - 5.78318596294678452118e+00) * (z - 3.04712623436620863991e+01) *
                 horner(z, rp) / horner(z, rq));
            near = at::where(at::lt(xn, 1e-5), 1.0 - z * 0.25, near);
            auto far = (bessel_far(
                clamp_to(x, at::logical_not(near_mask), 6.0), 0, false
            ));
            return at::where(near_mask, near, far);
        }

        // J1(x), x >= 0.
        at::Tensor j1_nonneg(const at::Tensor& x) {
            static const Coeffs rp = {
                -8.99971225705559398224e+08,
                4.52228297998194034323e+11,
                -7.27494245221818276015e+13,
                3.68295732863852883286e+15
            };
            static const Coeffs rq = {
                6.20836478118054335476e+02,
                2.56987256757748830383e+05,
                8.35146791431949253037e+07,
                2.21511595479792499675e+10,
                4.74914122079991414898e+12,
                7.84369607876235854894e+14,
                8.95222336184627338078e+16,
                5.32278620332680085395e+18
            };
            auto near_mask = at::le(x, 5.0);
            auto xn = clamp_to(x, near_mask, 1.0);
            auto z = xn * xn;
            auto near =
                (horner(z, rp) / horner(z, rq) * xn * (z - 1.46819706421238932572e+01) *
                 (z - 4.92184563216946036703e+01));
            auto far = (bessel_far(
                clamp_to(x, at::logical_not(near_mask), 6.0), 1, false
            ));
            return at::where(near_mask, near, far);
        }

        at::Tensor bessel_j0(const at::Tensor& x) {
            return j0_nonneg(at::abs(x));
        }

        at::Tensor bessel_j1(const at::Tensor& x) {
            return at::sign(x) * j1_nonneg(at::abs(x));
        }

        // Y0 / Y1: -inf at 0, NaN below.
        at::Tensor y_edges(const at::Tensor& x, at::Tensor value) {
            value = at::where(at::eq(x, 0.0), -kInf, value);
            return at::where(at::lt(x, 0.0), kNan, value);
        }

        at::Tensor bessel_y0(const at::Tensor& x) {
            static const Coeffs yp = {
                1.55924367855235737965e+04,
                -1.46639295903971606143e+07,
                5.43526477051876500413e+09,
                -9.82136065717911466409e+11,
                8.75906394395366999549e+13,
                -3.46628303384729719441e+15,
                4.42733268572569800351e+16,
                -1.84950800436986690637e+16
            };
            static const Coeffs yq = {
                1.04128353664259848412e+03,
                6.26107330137134956842e+05,
                2.68919633393814121987e+08,
                8.64002487103935000337e+10,
                2.02979612750105546709e+13,
                3.17157752842975028269e+15,
                2.50596256172653059228e+17
            };
            auto near_mask = at::le(x, 5.0);
            auto xn = clamp_to(x, at::logical_and(near_mask, at::gt(x, 0.0)), 1.0);
            auto z = xn * xn;
            auto near =
                (horner(z, yp) / horner(z, yq) +
                 kTwoOverPi * at::log(xn) * j0_nonneg(xn));
            auto far = (bessel_far(
                clamp_to(x, at::logical_not(near_mask), 6.0), 0, true
            ));
            return y_edges(x, at::where(near_mask, near, far));
        }

        at::Tensor bessel_y1(const at::Tensor& x) {
            static const Coeffs yp = {
                1.26320474790178026440e+09,
                -6.47355876379160291031e+11,
                1.14509511541823727583e+14,
                -8.12770255501325109621e+15,
                2.02439475713594898196e+17,
                -7.78877196265950026825e+17
            };
            static const Coeffs yq = {
                5.94301592346128195359e+02,
                2.35564092943068577943e+05,
                7.34811944459721705660e+07,
                1.87601316108706159478e+10,
                3.88231277496238566008e+12,
                6.20557727146953693363e+14,
                6.87141087355300489866e+16,
                3.97270608116560655612e+18
            };
            auto near_mask = at::le(x, 5.0);
            auto xn = clamp_to(x, at::logical_and(near_mask, at::gt(x, 0.0)), 1.0);
            auto z = xn * xn;
            auto near =
                (xn * (horner(z, yp) / horner(z, yq)) +
                 kTwoOverPi * (j1_nonneg(xn) * at::log(xn) - 1.0 / xn));
            auto far = (bessel_far(
                clamp_to(x, at::logical_not(near_mask), 6.0), 1, true
            ));
            return y_edges(x, at::where(near_mask, near, far));
        }

        // ---- I0 / I1 / K0 / K1 ----

        const Coeffs kI0A = {-4.41534164647933937950e-18, 3.33079451882223809783e-17,
                             -2.43127984654795469359e-16, 1.71539128555513303061e-15,
                             -1.16853328779934516808e-14, 7.67618549860493561688e-14,
                             -4.85644678311192946090e-13, 2.95505266312963983461e-12,
                             -1.72682629144155570723e-11, 9.67580903537323691224e-11,
                             -5.18979560163526290666e-10, 2.65982372468238665035e-09,
                             -1.30002500998624804212e-08, 6.04699502254191894932e-08,
                             -2.67079385394061173391e-07, 1.11738753912010371815e-06,
                             -4.41673835845875056359e-06, 1.64484480707288970893e-05,
                             -5.75419501008210370398e-05, 1.88502885095841655729e-04,
                             -5.76375574538582365885e-04, 1.63947561694133579842e-03,
                             -4.32430999505057594430e-03, 1.05464603945949983183e-02,
                             -2.37374148058994688156e-02, 4.93052842396707084878e-02,
                             -9.49010970480476444210e-02, 1.71620901522208775349e-01,
                             -3.04682672343198398683e-01, 6.76795274409476084995e-01};
        const Coeffs kI0B = {-7.23318048787475395456e-18, -4.83050448594418207126e-18,
                             4.46562142029675999901e-17,  3.46122286769746109310e-17,
                             -2.82762398051658348494e-16, -3.42548561967721913462e-16,
                             1.77256013305652638360e-15,  3.81168066935262242075e-15,
                             -9.55484669882830764870e-15, -4.15056934728722208663e-14,
                             1.54008621752140982691e-14,  3.85277838274214270114e-13,
                             7.18012445138366623367e-13,  -1.79417853150680611778e-12,
                             -1.32158118404477131188e-11, -3.14991652796324136454e-11,
                             1.18891471078464383424e-11,  4.94060238822496958910e-10,
                             3.39623202570838634515e-09,  2.26666899049817806459e-08,
                             2.04891858946906374183e-07,  2.89137052083475648297e-06,
                             6.88975834691682398426e-05,  3.36911647825569408990e-03,
                             8.04490411014108831608e-01};
        const Coeffs kI1A = {2.77791411276104639959e-18, -2.11142121435816608115e-17,
                             1.55363195773620046921e-16, -1.10559694773538630805e-15,
                             7.60068429473540693410e-15, -5.04218550472791168711e-14,
                             3.22379336594557470981e-13, -1.98397439776494371520e-12,
                             1.17361862988909016308e-11, -6.66348972350202774223e-11,
                             3.62559028155211703701e-10, -1.88724975172282928790e-09,
                             9.38153738649577178388e-09, -4.44505912879632808065e-08,
                             2.00329475355213526229e-07, -8.56872026469545474066e-07,
                             3.47025130813767847674e-06, -1.32731636560394358279e-05,
                             4.78156510755005422638e-05, -1.61760815825896745588e-04,
                             5.12285956168575772895e-04, -1.51357245063125314899e-03,
                             4.15642294431288815669e-03, -1.05640848946261981558e-02,
                             2.47264490306265168283e-02, -5.29459812080949914269e-02,
                             1.02643658689847095384e-01, -1.76416518357834055153e-01,
                             2.52587186443633654823e-01};
        const Coeffs kI1B = {7.51729631084210481353e-18,  4.41434832307170791151e-18,
                             -4.65030536848935832153e-17, -3.20952592199342395980e-17,
                             2.96262899764595013876e-16,  3.30820231092092828324e-16,
                             -1.88035477551078244854e-15, -3.81440307243700780478e-15,
                             1.04202769841288027642e-14,  4.27244001671195135429e-14,
                             -2.10154184277266431302e-14, -4.08355111109219731823e-13,
                             -7.19855177624590851209e-13, 2.03562854414708950722e-12,
                             1.41258074366137813316e-11,  3.25260358301548823856e-11,
                             -1.89749581235054123450e-11, -5.58974346219658380687e-10,
                             -3.83538038596423702205e-09, -2.63146884688951950684e-08,
                             -2.51223623787020892529e-07, -3.88256480887769039346e-06,
                             -1.10588938762623716291e-04, -9.76109749136146840777e-03,
                             7.78576235018280120474e-01};
        const Coeffs kK0A = {
            1.37446543561352307156e-16,
            4.25981614279661018399e-14,
            1.03496952576338420167e-11,
            1.90451637722020886025e-09,
            2.53479107902614945675e-07,
            2.28621210311945178607e-05,
            1.26461541144692592338e-03,
            3.59799365153615016266e-02,
            3.44289899924628486886e-01,
            -5.35327393233902768720e-01
        };
        const Coeffs kK0B = {5.30043377268626276149e-18, -1.64758043015242134646e-17,
                             5.21039150503902756861e-17, -1.67823109680541210385e-16,
                             5.51205597852431940784e-16, -1.84859337734377901440e-15,
                             6.34007647740507060557e-15, -2.22751332699166985548e-14,
                             8.03289077536357521100e-14, -2.98009692317273043925e-13,
                             1.14034058820847496303e-12, -4.51459788337394416547e-12,
                             1.85594911495471785253e-11, -7.95748924447710747776e-11,
                             3.57739728140030116597e-10, -1.69753450938905987466e-09,
                             8.57403401741422608519e-09, -4.66048989768794782956e-08,
                             2.76681363944501510342e-07, -1.83175552271911948767e-06,
                             1.39498137188764993662e-05, -1.28495495816278026384e-04,
                             1.56988388573005337491e-03, -3.14481013119645005427e-02,
                             2.44030308206595545468e+00};
        const Coeffs kK1A = {
            -7.02386347938628759343e-18,
            -2.42744985051936593393e-15,
            -6.66690169419932900609e-13,
            -1.41148839263352776110e-10,
            -2.21338763073472585583e-08,
            -2.43340614156596823496e-06,
            -1.73028895751305206302e-04,
            -6.97572385963986435018e-03,
            -1.22611180822657148235e-01,
            -3.53155960776544875667e-01,
            1.52530022733894777053e+00
        };
        const Coeffs kK1B = {-5.75674448366501715755e-18, 1.79405087314755922667e-17,
                             -5.68946255844285935196e-17, 1.83809354436663880070e-16,
                             -6.05704724837331885336e-16, 2.03870316562433424052e-15,
                             -7.01983709041831346144e-15, 2.47715442448130437068e-14,
                             -8.97670518232499435011e-14, 3.34841966607842919884e-13,
                             -1.28917396095102890680e-12, 5.13963967348173025100e-12,
                             -2.12996783842756842877e-11, 9.21831518760500529508e-11,
                             -4.19035475934189648750e-10, 2.01504975519703286596e-09,
                             -1.03457624656780970260e-08, 5.74108412545004946722e-08,
                             -3.50196060308781257119e-07, 2.40648494783721712015e-06,
                             -1.93619797416608296024e-05, 1.95215518471351631108e-04,
                             -2.85781685962277938680e-03, 1.03923736576817238437e-01,
                             2.72062619048444266945e+00};

        // exp(-|x|) I_order(|x|) (order 0 or 1): Chebyshev in |x|/2 - 2 up to 8, in
        // 32/|x| - 2 above (divided by sqrt|x|); order 1 times |x| near 0.
        at::Tensor scaled_i(const at::Tensor& x, int order) {
            auto ax = at::abs(x);
            auto near_mask = at::le(ax, 8.0);
            auto xn = clamp_to(ax, near_mask, 1.0);
            auto xf = clamp_to(ax, at::logical_not(near_mask), 9.0);
            const auto& a = order == 0 ? kI0A : kI1A;
            const auto& b = order == 0 ? kI0B : kI1B;
            auto near = chbevl(xn * 0.5 - 2.0, a, a.size());
            if (order == 1) {
                near = near * xn;
            }
            auto far = chbevl(32.0 / xf - 2.0, b, b.size()) / at::sqrt(xf);
            return at::where(near_mask, near, far);
        }

        at::Tensor modified_i0(const at::Tensor& x) {
            return at::exp(at::abs(x)) * scaled_i(x, 0);
        }

        at::Tensor modified_i1(const at::Tensor& x) {
            return at::sign(x) * at::exp(at::abs(x)) * scaled_i(x, 1);
        }

        // K_order(x) (or exp(x) K_order(x) when `scaled`): Chebyshev in x^2 - 2 with
        // the log(x/2) I term up to 2, in 8/x - 2 above; +inf at 0, NaN below.
        at::Tensor k(const at::Tensor& x, int order, bool scaled) {
            auto near_mask = at::le(x, 2.0);
            auto xn = clamp_to(x, at::logical_and(near_mask, at::gt(x, 0.0)), 1.0);
            auto xf = clamp_to(x, at::logical_not(near_mask), 3.0);
            const auto& a = order == 0 ? kK0A : kK1A;
            const auto& b = order == 0 ? kK0B : kK1B;
            auto series = chbevl(xn * xn - 2.0, a, a.size());
            auto near =
                (order == 0 ? series - at::log(xn * 0.5) * modified_i0(xn)
                            : at::log(xn * 0.5) * modified_i1(xn) + series / xn);
            if (scaled) {
                near = near * at::exp(xn);
            }
            auto far = chbevl(8.0 / xf - 2.0, b, b.size()) / at::sqrt(xf);
            if (!scaled) {
                far = far * at::exp(-xf);
            }
            auto value = at::where(near_mask, near, far);
            value = at::where(at::eq(x, 0.0), kInf, value);
            return at::where(at::lt(x, 0.0), kNan, value);
        }

        // sin(x) / x: the series to x^12 for |x| < 0.5, 0 at +-inf.
        at::Tensor spherical_j0(const at::Tensor& x) {
            auto small = at::lt(at::abs(x), 0.5);
            auto xs = clamp_to(x, small, 0.0);
            auto z = xs * xs;
            auto series = horner(
                z,
                {1.0 / 6227020800.0,
                 -1.0 / 39916800.0,
                 1.0 / 362880.0,
                 -1.0 / 5040.0,
                 1.0 / 120.0,
                 -1.0 / 6.0,
                 1.0}
            );
            auto xl = clamp_to(x, at::logical_not(small), 1.0);
            auto value = at::where(small, series, at::sin(xl) / xl);
            return at::where(at::isinf(x), 0.0, value);
        }

        // Ai(x): asymptotic oscillating form below -2.09, decaying form from 2.09, the
        // Maclaurin series f(x) c1 - g(x) c2 between; NaN at +-inf, 0 above 103.892.
        at::Tensor airy_ai(const at::Tensor& x) {
            static const Coeffs an = {
                3.46538101525629032477e-01,
                1.20075952739645805542e+01,
                7.62796053615234516538e+01,
                1.68089224934630576269e+02,
                1.59756391350164413639e+02,
                7.05360906840444183113e+01,
                1.40264691163389668864e+01,
                9.99999999999999995305e-01
            };
            static const Coeffs ad = {
                5.67594532638770212846e-01,
                1.47562562584847203173e+01,
                8.45138970141474626562e+01,
                1.77318088145400459522e+02,
                1.64234692871529701831e+02,
                7.14778400825575695274e+01,
                1.40959135607834029598e+01,
                1.00000000000000000470e+00
            };
            static const Coeffs afn = {
                -1.31696323418331795333e-01,
                -6.26456544431912369773e-01,
                -6.93158036036933542233e-01,
                -2.79779981545119124951e-01,
                -4.91900132609500318020e-02,
                -4.06265923594885404393e-03,
                -1.59276496239262096340e-04,
                -2.77649108155232920844e-06,
                -1.67787698489114633780e-08
            };
            static const Coeffs afd = {
                1.33560420706553243746e+01,
                3.26825032795224613948e+01,
                2.67367040941499554804e+01,
                9.18707402907259625840e+00,
                1.47529146771666414581e+00,
                1.15687173795188044134e-01,
                4.40291641615211203805e-03,
                7.54720348287414296618e-05,
                4.51850092970580378464e-07
            };
            static const Coeffs agn = {
                1.97339932091685679179e-02,
                3.91103029615688277255e-01,
                1.06579897599595591108e+00,
                9.39169229816650230044e-01,
                3.51465656105547619242e-01,
                6.33888919628925490927e-02,
                5.85804113048388458567e-03,
                2.82851600836737019778e-04,
                6.98793669997260967291e-06,
                8.11789239554389293311e-08,
                3.41551784765923618484e-10
            };
            static const Coeffs agd = {
                9.30892908077441974853e+00,
                1.98352928718312140417e+01,
                1.55646628932864612953e+01,
                5.47686069422975497931e+00,
                9.54293611618961883998e-01,
                8.64580826352392193095e-02,
                4.12656523824222607191e-03,
                1.01259085116509135510e-04,
                1.17166733214413521882e-06,
                4.91834570062930015649e-09
            };
            constexpr double kInvSqrtPi = 5.64189583547756286948e-01;

            auto low_mask = at::lt(x, -2.09);
            auto high_mask = at::ge(x, 2.09);
            auto mid_mask = at::logical_not(at::logical_or(low_mask, high_mask));

            auto xl = clamp_to(x, low_mask, -3.0);
            auto root = at::sqrt(-xl);
            auto zl = 1.0 / (xl * root * (-2.0 / 3.0));
            auto zz = zl * zl;
            auto t = xl * root * (-2.0 / 3.0) + 0.25 * M_PI;
            auto low =
                (kInvSqrtPi / at::sqrt(root) *
                 (at::sin(t) * (1.0 + zz * horner(zz, afn) / horner(zz, afd)) -
                  at::cos(t) * (zl * horner(zz, agn) / horner(zz, agd))));

            auto xh = clamp_to(x, at::logical_and(high_mask, at::le(x, 103.892)), 3.0);
            auto zeta = xh * at::sqrt(xh) * (2.0 / 3.0);
            auto r = 1.0 / zeta;
            auto high =
                (kInvSqrtPi * (horner(r, an) / horner(r, ad)) /
                 (at::sqrt(at::sqrt(xh)) * 2.0 * at::exp(zeta)));
            high = at::where(at::gt(x, 103.892), 0.0, high);

            auto xm = clamp_to(x, mid_mask, 0.0);
            auto z3 = xm * xm * xm;
            auto f = at::ones_like(xm), g = xm;
            auto m = at::ones_like(xm), n = xm;
            double kk = 1.0;
            for (int i = 0; i < 16; ++i) {
                m = m * z3 / (kk + 1.0);
                n = n * z3 / (kk + 2.0);
                m = m / (kk + 2.0);
                f = f + m;
                n = n / (kk + 3.0);
                g = g + n;
                kk += 3.0;
            }
            auto mid = f * 0.355028053887817239260 - g * 0.258819403792806798405;

            auto value = at::where(low_mask, low, at::where(high_mask, high, mid));
            return at::where(at::isinf(x), kNan, value);
        }

        using Fn = at::Tensor (*)(const at::Tensor&);

        template <Fn F>
        at::Tensor& special_out(const at::Tensor& self, at::Tensor& out) {
            return into("special_bessel.out", out, F(self.to(at::kFloat)));
        }

        at::Tensor k0(const at::Tensor& x) {
            return k(x, 0, false);
        }

        at::Tensor k1(const at::Tensor& x) {
            return k(x, 1, false);
        }

        at::Tensor k0e(const at::Tensor& x) {
            return k(x, 0, true);
        }

        at::Tensor k1e(const at::Tensor& x) {
            return k(x, 1, true);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("special_bessel_j0.out", &special_out<bessel_j0>);
        m.impl("special_bessel_j1.out", &special_out<bessel_j1>);
        m.impl("special_bessel_y0.out", &special_out<bessel_y0>);
        m.impl("special_bessel_y1.out", &special_out<bessel_y1>);
        m.impl("special_modified_bessel_i0.out", &special_out<modified_i0>);
        m.impl("special_modified_bessel_i1.out", &special_out<modified_i1>);
        m.impl("special_modified_bessel_k0.out", &special_out<k0>);
        m.impl("special_modified_bessel_k1.out", &special_out<k1>);
        m.impl("special_scaled_modified_bessel_k0.out", &special_out<k0e>);
        m.impl("special_scaled_modified_bessel_k1.out", &special_out<k1e>);
        m.impl("special_spherical_bessel_j0.out", &special_out<spherical_j0>);
        m.impl("special_airy_ai.out", &special_out<airy_ai>);
    }

}  // namespace ktt
