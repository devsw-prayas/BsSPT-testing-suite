# spectra_library.py

import torch


class SpectraLibrary:

    @staticmethod
    def _to_tensor(value, lbda):
        return torch.tensor(value, device=lbda.device, dtype=lbda.dtype)

    @staticmethod
    def _gaussian(lbda, center, width):
        c = SpectraLibrary._to_tensor(center, lbda)
        w = SpectraLibrary._to_tensor(width, lbda)
        two = SpectraLibrary._to_tensor(2.0, lbda)

        return torch.exp(-((lbda - c) ** 2) / (two * w ** 2))

    @staticmethod
    def _blackbody(lbda, T_value):

        device = lbda.device
        dtype = lbda.dtype

        T = torch.tensor(T_value, device=device, dtype=dtype)

        lbda_m = lbda * torch.tensor(1e-9, device=device, dtype=dtype)

        h = torch.tensor(6.62607015e-34, device=device, dtype=dtype)
        c = torch.tensor(2.99792458e8, device=device, dtype=dtype)
        k = torch.tensor(1.380649e-23, device=device, dtype=dtype)
        two = torch.tensor(2.0, device=device, dtype=dtype)
        one = torch.tensor(1.0, device=device, dtype=dtype)

        numerator = two * h * c ** 2
        exponent = (h * c) / (lbda_m * k * T)
        denominator = (lbda_m ** 5) * (torch.exp(exponent) - one)

        return numerator / denominator

    @staticmethod
    def build_all(domain):

        lbda = domain.m_lambda
        one = SpectraLibrary._to_tensor(1.0, lbda)
        half = SpectraLibrary._to_tensor(0.5, lbda)
        spectra = {}

        # =====================================================
        # IRL
        # =====================================================

        spectra["spectrum_00_IRL"] = SpectraLibrary._blackbody(lbda, 6500.0)
        spectra["spectrum_01_IRL"] = SpectraLibrary._blackbody(lbda, 2800.0)

        spectra["spectrum_02_IRL"] = (
            SpectraLibrary._gaussian(lbda, 450.0, 18.0) +
            SpectraLibrary._to_tensor(1.5, lbda) *
            SpectraLibrary._gaussian(lbda, 580.0, 40.0)
        )

        spectra["spectrum_03_IRL"] = SpectraLibrary._gaussian(lbda, 630.0, 15.0)
        spectra["spectrum_04_IRL"] = SpectraLibrary._gaussian(lbda, 540.0, 18.0)
        spectra["spectrum_05_IRL"] = SpectraLibrary._gaussian(lbda, 460.0, 15.0)

        spectra["spectrum_06_IRL"] = (
            SpectraLibrary._to_tensor(0.3, lbda) +
            half * SpectraLibrary._gaussian(lbda, 520.0, 80.0)
        )

        spectra["spectrum_07_IRL"] = SpectraLibrary._gaussian(lbda, 532.0, 3.0)

        # =====================================================
        # HOSTILE
        # =====================================================

        spectra["spectrum_08_HOSTILE"] = (
            SpectraLibrary._gaussian(lbda, 430.0, 10.0) +
            SpectraLibrary._gaussian(lbda, 610.0, 12.0)
        )

        spectra["spectrum_09_HOSTILE"] = (
            one + SpectraLibrary._to_tensor(0.8, lbda) *
            torch.cos(SpectraLibrary._to_tensor(0.05, lbda) * lbda)
        )

        spectra["spectrum_10_HOSTILE"] = (
            SpectraLibrary._to_tensor(0.2, lbda) +
            SpectraLibrary._gaussian(lbda, 500.0, 5.0)
        )

        spectra["spectrum_11_HOSTILE"] = (
            one + half *
            torch.sin(SpectraLibrary._to_tensor(0.12, lbda) * lbda)
        )

        spectra["spectrum_12_HOSTILE"] = SpectraLibrary._gaussian(lbda, 390.0, 20.0)
        spectra["spectrum_13_HOSTILE"] = SpectraLibrary._gaussian(lbda, 760.0, 25.0)

        spectra["spectrum_14_HOSTILE"] = (
            SpectraLibrary._gaussian(lbda, 450.0, 10.0) +
            SpectraLibrary._gaussian(lbda, 520.0, 15.0) +
            SpectraLibrary._gaussian(lbda, 610.0, 12.0)
        )

        spectra["spectrum_15_HOSTILE"] = SpectraLibrary._gaussian(lbda, 555.0, 2.0)

        # -----------------------------------------------------
        # Normalize safely
        # -----------------------------------------------------

        for key in spectra:

            integral = domain.integrate(spectra[key])

            eps = SpectraLibrary._to_tensor(1e-16, lbda)
            spectra[key] = spectra[key] / (integral + eps)

        return spectra
