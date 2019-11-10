import os
import pandas as pd
import json
import time
import pickle

import datetime
import math

import numpy as np
import re

import torch

from bootstrap.lib.logger import Logger

from .spectra import Scan
from .spectra import SpectraFound
from .spectra import MGF


class PXD000561:

    ADULT_ADRENALGLAND_GEL_ELITE_FILES = {
        'b01' : 'Adult_Adrenalgland_Gel_Elite_49_f01.mgf',
        'b02' : 'Adult_Adrenalgland_Gel_Elite_49_f02.mgf',
        'b03' : 'Adult_Adrenalgland_Gel_Elite_49_f03.mgf',
        'b04' : 'Adult_Adrenalgland_Gel_Elite_49_f04.mgf',
        'b05' : 'Adult_Adrenalgland_Gel_Elite_49_f05.mgf',
        'b06' : 'Adult_Adrenalgland_Gel_Elite_49_f06.mgf',
        'b07' : 'Adult_Adrenalgland_Gel_Elite_49_f07.mgf',
        'b08' : 'Adult_Adrenalgland_Gel_Elite_49_f08.mgf',
        'b09' : 'Adult_Adrenalgland_Gel_Elite_49_f09.mgf',
        'b10' : 'Adult_Adrenalgland_Gel_Elite_49_f10.mgf',
        'b11' : 'Adult_Adrenalgland_Gel_Elite_49_f11.mgf',
        'b12' : 'Adult_Adrenalgland_Gel_Elite_49_f12.mgf',
        'b13' : 'Adult_Adrenalgland_Gel_Elite_49_f13.mgf',
        'b14' : 'Adult_Adrenalgland_Gel_Elite_49_f14.mgf',
        'b15' : 'Adult_Adrenalgland_Gel_Elite_49_f15.mgf',
        'b16' : 'Adult_Adrenalgland_Gel_Elite_49_f16.mgf',
        'b17' : 'Adult_Adrenalgland_Gel_Elite_49_f17.mgf',
        'b18' : 'Adult_Adrenalgland_Gel_Elite_49_f18.mgf',
        'b19' : 'Adult_Adrenalgland_Gel_Elite_49_f19.mgf',
        'b20' : 'Adult_Adrenalgland_Gel_Elite_49_f20.mgf',
        'b21' : 'Adult_Adrenalgland_Gel_Elite_49_f21.mgf',
        'b22' : 'Adult_Adrenalgland_Gel_Elite_49_f22.mgf',
        'b23' : 'Adult_Adrenalgland_Gel_Elite_49_f23.mgf',
        'b24' : 'Adult_Adrenalgland_Gel_Elite_49_f24.mgf'
    }


    ADULT_ADRENALGLAND_GEL_VELOS_FILES = {
        'D01' : 'Adult_Adrenalgland_Gel_Velos_2_f01.mgf',
        'D02' : 'Adult_Adrenalgland_Gel_Velos_2_f02.mgf',
        'D03' : 'Adult_Adrenalgland_Gel_Velos_2_f03.mgf',
        'D04' : 'Adult_Adrenalgland_Gel_Velos_2_f04.mgf',
        'D05' : 'Adult_Adrenalgland_Gel_Velos_2_f05.mgf',
        'D06' : 'Adult_Adrenalgland_Gel_Velos_2_f06.mgf',
        'D07' : 'Adult_Adrenalgland_Gel_Velos_2_f07.mgf',
        'D08' : 'Adult_Adrenalgland_Gel_Velos_2_f08.mgf',
        'D09' : 'Adult_Adrenalgland_Gel_Velos_2_f09.mgf',
        'D10' : 'Adult_Adrenalgland_Gel_Velos_2_f10.mgf',
        'D11' : 'Adult_Adrenalgland_Gel_Velos_2_f11.mgf',
        'D12' : 'Adult_Adrenalgland_Gel_Velos_2_f12.mgf',
        'E01' : 'Adult_Adrenalgland_Gel_Velos_2_f13.mgf',
        'E01Re' : 'Adult_Adrenalgland_Gel_Velos_2_f14.mgf',
        'E02' : 'Adult_Adrenalgland_Gel_Velos_2_f15.mgf',
        'E03' : 'Adult_Adrenalgland_Gel_Velos_2_f16.mgf',
        'E04' : 'Adult_Adrenalgland_Gel_Velos_2_f17.mgf',
        'E05' : 'Adult_Adrenalgland_Gel_Velos_2_f18.mgf',
        'E06' : 'Adult_Adrenalgland_Gel_Velos_2_f19.mgf',
        'E06-110108181623' : 'Adult_Adrenalgland_Gel_Velos_2_f20.mgf',
        'E07' : 'Adult_Adrenalgland_Gel_Velos_2_f21.mgf',
        'E08' : 'Adult_Adrenalgland_Gel_Velos_2_f22.mgf',
        'E09' : 'Adult_Adrenalgland_Gel_Velos_2_f23.mgf',
        'E10' : 'Adult_Adrenalgland_Gel_Velos_2_f24.mgf',
        'E11' : 'Adult_Adrenalgland_Gel_Velos_2_f25.mgf',
        'E12' : 'Adult_Adrenalgland_Gel_Velos_2_f26.mgf'
    }


    ADULT_ADRENALGLAND_BRP_VELOS_FILES = {
        'A01' : 'Adult_Adrenalgland_bRP_Velos_1_f01.mgf',
        'A02' : 'Adult_Adrenalgland_bRP_Velos_1_f02.mgf',
        'A03' : 'Adult_Adrenalgland_bRP_Velos_1_f03.mgf',
        'A04' : 'Adult_Adrenalgland_bRP_Velos_1_f04.mgf',
        'A05' : 'Adult_Adrenalgland_bRP_Velos_1_f05.mgf',
        'A06' : 'Adult_Adrenalgland_bRP_Velos_1_f06.mgf',
        'A07' : 'Adult_Adrenalgland_bRP_Velos_1_f07.mgf',
        'A08' : 'Adult_Adrenalgland_bRP_Velos_1_f08.mgf',
        'A09' : 'Adult_Adrenalgland_bRP_Velos_1_f09.mgf',
        'A10' : 'Adult_Adrenalgland_bRP_Velos_1_f10.mgf',
        'A11' : 'Adult_Adrenalgland_bRP_Velos_1_f11.mgf',
        'A12' : 'Adult_Adrenalgland_bRP_Velos_1_f12.mgf',
        'B01' : 'Adult_Adrenalgland_bRP_Velos_1_f13.mgf',
        'B02' : 'Adult_Adrenalgland_bRP_Velos_1_f14.mgf',
        'B03' : 'Adult_Adrenalgland_bRP_Velos_1_f15.mgf',
        'B04' : 'Adult_Adrenalgland_bRP_Velos_1_f16.mgf',
        'B05' : 'Adult_Adrenalgland_bRP_Velos_1_f17.mgf',
        'B06' : 'Adult_Adrenalgland_bRP_Velos_1_f18.mgf',
        'B07' : 'Adult_Adrenalgland_bRP_Velos_1_f19.mgf',
        'B08' : 'Adult_Adrenalgland_bRP_Velos_1_f20.mgf',
        'B09' : 'Adult_Adrenalgland_bRP_Velos_1_f21.mgf',
        'B10' : 'Adult_Adrenalgland_bRP_Velos_1_f22.mgf',
        'B11' : 'Adult_Adrenalgland_bRP_Velos_1_f23.mgf',
        'B12' : 'Adult_Adrenalgland_bRP_Velos_1_f24.mgf',
        'C01' : 'Adult_Adrenalgland_bRP_Velos_1_f25.mgf',
        'C02' : 'Adult_Adrenalgland_bRP_Velos_1_f26.mgf',
        'C03' : 'Adult_Adrenalgland_bRP_Velos_1_f27.mgf',
        'C04' : 'Adult_Adrenalgland_bRP_Velos_1_f28.mgf',
        'C05' : 'Adult_Adrenalgland_bRP_Velos_1_f29.mgf',
        'C06' : 'Adult_Adrenalgland_bRP_Velos_1_f30.mgf',
        'C07' : 'Adult_Adrenalgland_bRP_Velos_1_f31.mgf',
        'C08' : 'Adult_Adrenalgland_bRP_Velos_1_f32.mgf',
        'C09' : 'Adult_Adrenalgland_bRP_Velos_1_f33.mgf',
        'C10' : 'Adult_Adrenalgland_bRP_Velos_1_f34.mgf',
        'C11' : 'Adult_Adrenalgland_bRP_Velos_1_f35.mgf',
        'C12' : 'Adult_Adrenalgland_bRP_Velos_1_f36.mgf'
    }


    ADULT_HEART_BRP_ELITE_FILES = {
        'f01' : 'Adult_Heart_bRP_Elite_81_f01.mgf',
        'f02' : 'Adult_Heart_bRP_Elite_81_f02.mgf',
        'f03' : 'Adult_Heart_bRP_Elite_81_f03.mgf',
        'f04' : 'Adult_Heart_bRP_Elite_81_f04.mgf',
        'f05' : 'Adult_Heart_bRP_Elite_81_f05.mgf',
        'f06' : 'Adult_Heart_bRP_Elite_81_f06.mgf',
        'f07' : 'Adult_Heart_bRP_Elite_81_f07.mgf',
        'f08' : 'Adult_Heart_bRP_Elite_81_f08.mgf',
        'f09' : 'Adult_Heart_bRP_Elite_81_f09.mgf',
        'f10' : 'Adult_Heart_bRP_Elite_81_f10.mgf',
        'f11' : 'Adult_Heart_bRP_Elite_81_f11.mgf',
        'f12' : 'Adult_Heart_bRP_Elite_81_f12.mgf',
        'f13' : 'Adult_Heart_bRP_Elite_81_f13.mgf',
        'f14' : 'Adult_Heart_bRP_Elite_81_f14.mgf',
        'f15' : 'Adult_Heart_bRP_Elite_81_f15.mgf',
        'f16' : 'Adult_Heart_bRP_Elite_81_f16.mgf',
        'f17' : 'Adult_Heart_bRP_Elite_81_f17.mgf',
        'f18' : 'Adult_Heart_bRP_Elite_81_f18.mgf',
        'f19' : 'Adult_Heart_bRP_Elite_81_f19.mgf',
        'f20' : 'Adult_Heart_bRP_Elite_81_f20.mgf',
        'f21' : 'Adult_Heart_bRP_Elite_81_f21.mgf',
        'f22' : 'Adult_Heart_bRP_Elite_81_f22.mgf',
        'f23' : 'Adult_Heart_bRP_Elite_81_f23.mgf',
        'f24' : 'Adult_Heart_bRP_Elite_81_f24.mgf',
    }


    ADULT_PLATELETS_GEL_ELITE_FILES = {
        'b01' : 'Adult_Platelets_Gel_Elite_48_f01.mgf',
        'b02' : 'Adult_Platelets_Gel_Elite_48_f02.mgf',
        'b03' : 'Adult_Platelets_Gel_Elite_48_f03.mgf',
        'b04' : 'Adult_Platelets_Gel_Elite_48_f04.mgf',
        'b05' : 'Adult_Platelets_Gel_Elite_48_f05.mgf',
        'b06' : 'Adult_Platelets_Gel_Elite_48_f06.mgf',
        'b07' : 'Adult_Platelets_Gel_Elite_48_f07.mgf',
        'b08' : 'Adult_Platelets_Gel_Elite_48_f08.mgf',
        'b09' : 'Adult_Platelets_Gel_Elite_48_f09.mgf',
        'b10' : 'Adult_Platelets_Gel_Elite_48_f10.mgf',
        'b11' : 'Adult_Platelets_Gel_Elite_48_f11.mgf',
        'b12' : 'Adult_Platelets_Gel_Elite_48_f12.mgf',
        'b13' : 'Adult_Platelets_Gel_Elite_48_f13.mgf',
        'b14' : 'Adult_Platelets_Gel_Elite_48_f14.mgf',
        'b15' : 'Adult_Platelets_Gel_Elite_48_f15.mgf',
        'b16' : 'Adult_Platelets_Gel_Elite_48_f16.mgf',
        'b17' : 'Adult_Platelets_Gel_Elite_48_f17.mgf',
        'b18' : 'Adult_Platelets_Gel_Elite_48_f18.mgf',
        'b19' : 'Adult_Platelets_Gel_Elite_48_f19.mgf',
        'b20' : 'Adult_Platelets_Gel_Elite_48_f20.mgf',
        'b21' : 'Adult_Platelets_Gel_Elite_48_f21.mgf',
        'b22' : 'Adult_Platelets_Gel_Elite_48_f22.mgf',
        'b23' : 'Adult_Platelets_Gel_Elite_48_f23.mgf',
        'b24' : 'Adult_Platelets_Gel_Elite_48_f24.mgf',
    }


    ADULT_URINARYBLADDER_GEL_ELITE_FILES = {
        'b01' : 'Adult_Urinarybladder_Gel_Elite_70_f01.mgf',
        'b02' : 'Adult_Urinarybladder_Gel_Elite_70_f02.mgf',
        'b03' : 'Adult_Urinarybladder_Gel_Elite_70_f03.mgf',
        'b04' : 'Adult_Urinarybladder_Gel_Elite_70_f04.mgf',
        'b05' : 'Adult_Urinarybladder_Gel_Elite_70_f05.mgf',
        'b06' : 'Adult_Urinarybladder_Gel_Elite_70_f06.mgf',
        'b07' : 'Adult_Urinarybladder_Gel_Elite_70_f07.mgf',
        'b08' : 'Adult_Urinarybladder_Gel_Elite_70_f08.mgf',
        'b09' : 'Adult_Urinarybladder_Gel_Elite_70_f09.mgf',
        'b09r' : 'Adult_Urinarybladder_Gel_Elite_70_f10.mgf',
        'b10' : 'Adult_Urinarybladder_Gel_Elite_70_f11.mgf',
        'b11' : 'Adult_Urinarybladder_Gel_Elite_70_f12.mgf',
        'b12' : 'Adult_Urinarybladder_Gel_Elite_70_f13.mgf',
        'b13' : 'Adult_Urinarybladder_Gel_Elite_70_f14.mgf',
        'b14' : 'Adult_Urinarybladder_Gel_Elite_70_f15.mgf',
        'b15' : 'Adult_Urinarybladder_Gel_Elite_70_f16.mgf',
        'b16' : 'Adult_Urinarybladder_Gel_Elite_70_f17.mgf',
        'b17' : 'Adult_Urinarybladder_Gel_Elite_70_f18.mgf',
        'b18' : 'Adult_Urinarybladder_Gel_Elite_70_f19.mgf',
        'b19' : 'Adult_Urinarybladder_Gel_Elite_70_f20.mgf',
        'b20' : 'Adult_Urinarybladder_Gel_Elite_70_f21.mgf',
        'b21' : 'Adult_Urinarybladder_Gel_Elite_70_f22.mgf',
        'b22' : 'Adult_Urinarybladder_Gel_Elite_70_f23.mgf',
        'b23' : 'Adult_Urinarybladder_Gel_Elite_70_f24.mgf',
        'b24' : 'Adult_Urinarybladder_Gel_Elite_70_f25.mgf',
    }


    FETAL_BRAIN_GEL_VELOS_FILES = {
		"E01" : "Fetal_Brain_Gel_Velos_16_f01.mgf",
		"E01rep" : "Fetal_Brain_Gel_Velos_16_f02.mgf",
		"E02" : "Fetal_Brain_Gel_Velos_16_f03.mgf",
		"E03" : "Fetal_Brain_Gel_Velos_16_f04.mgf",
		"E03rep" : "Fetal_Brain_Gel_Velos_16_f05.mgf",
		"E04" : "Fetal_Brain_Gel_Velos_16_f06.mgf",
		"E05" : "Fetal_Brain_Gel_Velos_16_f07.mgf",
		"E06" : "Fetal_Brain_Gel_Velos_16_f08.mgf",
		"E07" : "Fetal_Brain_Gel_Velos_16_f09.mgf",
		"E08" : "Fetal_Brain_Gel_Velos_16_f10.mgf",
		"E09" : "Fetal_Brain_Gel_Velos_16_f11.mgf",
		"E10" : "Fetal_Brain_Gel_Velos_16_f12.mgf",
		"E11" : "Fetal_Brain_Gel_Velos_16_f13.mgf",
		"E12" : "Fetal_Brain_Gel_Velos_16_f14.mgf",
		"F01" : "Fetal_Brain_Gel_Velos_16_f15.mgf",
		"F02" : "Fetal_Brain_Gel_Velos_16_f16.mgf",
		"F03" : "Fetal_Brain_Gel_Velos_16_f17.mgf",
		"F04" : "Fetal_Brain_Gel_Velos_16_f18.mgf",
		"F05" : "Fetal_Brain_Gel_Velos_16_f19.mgf",
		"F06" : "Fetal_Brain_Gel_Velos_16_f20.mgf",
		"F07" : "Fetal_Brain_Gel_Velos_16_f21.mgf",
		"F08" : "Fetal_Brain_Gel_Velos_16_f22.mgf",
		"F09" : "Fetal_Brain_Gel_Velos_16_f23.mgf",
		"F10" : "Fetal_Brain_Gel_Velos_16_f24.mgf",
		"F11" : "Fetal_Brain_Gel_Velos_16_f25.mgf",
		"F12" : "Fetal_Brain_Gel_Velos_16_f26.mgf",
		"G01" : "Fetal_Brain_Gel_Velos_16_f27.mgf",
		"G02" : "Fetal_Brain_Gel_Velos_16_f28.mgf",
		"G03" : "Fetal_Brain_Gel_Velos_16_f29.mgf"
    }


    FETAL_LIVER_BRP_ELITE_FILES = {
        '01' : 'Fetal_Liver_bRP_Elite_22_f01.mgf',
        '02' : 'Fetal_Liver_bRP_Elite_22_f02.mgf',
        '03' : 'Fetal_Liver_bRP_Elite_22_f03.mgf',
        '04' : 'Fetal_Liver_bRP_Elite_22_f04.mgf',
        '05' : 'Fetal_Liver_bRP_Elite_22_f05.mgf',
        '06' : 'Fetal_Liver_bRP_Elite_22_f06.mgf',
        '07' : 'Fetal_Liver_bRP_Elite_22_f07.mgf',
        '08' : 'Fetal_Liver_bRP_Elite_22_f08.mgf',
        '08-NCE27' : 'Fetal_Liver_bRP_Elite_22_f09.mgf',
        '08-NCE27-1' : 'Fetal_Liver_bRP_Elite_22_f10.mgf',
        '08-NCE27-32' : 'Fetal_Liver_bRP_Elite_22_f11.mgf',
        '08-NCE27-32-1' : 'Fetal_Liver_bRP_Elite_22_f12.mgf',
        '08-1' : 'Fetal_Liver_bRP_Elite_22_f13.mgf',
        '09' : 'Fetal_Liver_bRP_Elite_22_f14.mgf',
        '10' : 'Fetal_Liver_bRP_Elite_22_f15.mgf',
        '11' : 'Fetal_Liver_bRP_Elite_22_f16.mgf',
        '12' : 'Fetal_Liver_bRP_Elite_22_f17.mgf',
        '13-120502235957' : 'Fetal_Liver_bRP_Elite_22_f18.mgf',
        '13' : 'Fetal_Liver_bRP_Elite_22_f19.mgf',
        '14-120503014940' : 'Fetal_Liver_bRP_Elite_22_f20.mgf',
        '14' : 'Fetal_Liver_bRP_Elite_22_f21.mgf',
        '15-120503033941' : 'Fetal_Liver_bRP_Elite_22_f22.mgf',
        '15' : 'Fetal_Liver_bRP_Elite_22_f23.mgf',
        '16-120503052933' : 'Fetal_Liver_bRP_Elite_22_f24.mgf',
        '16' : 'Fetal_Liver_bRP_Elite_22_f25.mgf',
        '17-120503071920' : 'Fetal_Liver_bRP_Elite_22_f26.mgf',
        '17' : 'Fetal_Liver_bRP_Elite_22_f27.mgf',
        '18' : 'Fetal_Liver_bRP_Elite_22_f28.mgf',
        '19' : 'Fetal_Liver_bRP_Elite_22_f29.mgf',
        '20' : 'Fetal_Liver_bRP_Elite_22_f30.mgf',
        '21' : 'Fetal_Liver_bRP_Elite_22_f31.mgf',
        '22' : 'Fetal_Liver_bRP_Elite_22_f32.mgf',
        '23401' : 'Fetal_Liver_bRP_Elite_22_f33.mgf',
        '23' : 'Fetal_Liver_bRP_Elite_22_f34.mgf',
        '24' : 'Fetal_Liver_bRP_Elite_22_f35.mgf',
    }


    FETAL_OVARY_BRP_VELOS_FILES = {
        'E01' : 'Fetal_Ovary_bRP_Velos_41_f01.mgf',
        'E02' : 'Fetal_Ovary_bRP_Velos_41_f02.mgf',
        'E03' : 'Fetal_Ovary_bRP_Velos_41_f03.mgf',
        'E04' : 'Fetal_Ovary_bRP_Velos_41_f04.mgf',
        'E05' : 'Fetal_Ovary_bRP_Velos_41_f05.mgf',
        'E06' : 'Fetal_Ovary_bRP_Velos_41_f06.mgf',
        'E06-E1-E6' : 'Fetal_Ovary_bRP_Velos_41_f07.mgf',
        'E07' : 'Fetal_Ovary_bRP_Velos_41_f08.mgf',
        'E08' : 'Fetal_Ovary_bRP_Velos_41_f09.mgf',
        'E09' : 'Fetal_Ovary_bRP_Velos_41_f26.mgf',
        'E11' : 'Fetal_Ovary_bRP_Velos_41_f10.mgf',
        'E12' : 'Fetal_Ovary_bRP_Velos_41_f11.mgf',
        'E12-E7-E12' : 'Fetal_Ovary_bRP_Velos_41_f12.mgf',
        'F01' : 'Fetal_Ovary_bRP_Velos_41_f13.mgf',
        'F02' : 'Fetal_Ovary_bRP_Velos_41_f14.mgf',
        'F03' : 'Fetal_Ovary_bRP_Velos_41_f15.mgf',
        'F04' : 'Fetal_Ovary_bRP_Velos_41_f16.mgf',
        'F05' : 'Fetal_Ovary_bRP_Velos_41_f17.mgf',
        'F06' : 'Fetal_Ovary_bRP_Velos_41_f18.mgf',
        'F06-F1-F6' : 'Fetal_Ovary_bRP_Velos_41_f19.mgf',
        'F07' : 'Fetal_Ovary_bRP_Velos_41_f20.mgf',
        'F08' : 'Fetal_Ovary_bRP_Velos_41_f21.mgf',
        'F09' : 'Fetal_Ovary_bRP_Velos_41_f22.mgf',
        'F10' : 'Fetal_Ovary_bRP_Velos_41_f23.mgf',
        'F11' : 'Fetal_Ovary_bRP_Velos_41_f24.mgf',
        'F12' : 'Fetal_Ovary_bRP_Velos_41_f25.mgf',
    }


    FETAL_OVARY_BRP_ELITE_FILES = {
        '01' : 'Fetal_Ovary_bRP_Elite_25_f01.mgf',
        '02' : 'Fetal_Ovary_bRP_Elite_25_f02.mgf',
        '03' : 'Fetal_Ovary_bRP_Elite_25_f03.mgf',
        '04' : 'Fetal_Ovary_bRP_Elite_25_f04.mgf',
        '05' : 'Fetal_Ovary_bRP_Elite_25_f05.mgf',
        '06' : 'Fetal_Ovary_bRP_Elite_25_f06.mgf',
        '07' : 'Fetal_Ovary_bRP_Elite_25_f07.mgf',
        '08' : 'Fetal_Ovary_bRP_Elite_25_f08.mgf',
        '09' : 'Fetal_Ovary_bRP_Elite_25_f09.mgf',
        '10' : 'Fetal_Ovary_bRP_Elite_25_f10.mgf',
        '11' : 'Fetal_Ovary_bRP_Elite_25_f11.mgf',
        '12' : 'Fetal_Ovary_bRP_Elite_25_f12.mgf',
        '13' : 'Fetal_Ovary_bRP_Elite_25_f13.mgf',
        '14' : 'Fetal_Ovary_bRP_Elite_25_f14.mgf',
        '15' : 'Fetal_Ovary_bRP_Elite_25_f15.mgf',
        '16' : 'Fetal_Ovary_bRP_Elite_25_f16.mgf',
        '17' : 'Fetal_Ovary_bRP_Elite_25_f17.mgf',
        '18' : 'Fetal_Ovary_bRP_Elite_25_f18.mgf',
        '19' : 'Fetal_Ovary_bRP_Elite_25_f19.mgf',
        '20' : 'Fetal_Ovary_bRP_Elite_25_f20.mgf',
        '21' : 'Fetal_Ovary_bRP_Elite_25_f21.mgf',
        '22' : 'Fetal_Ovary_bRP_Elite_25_f22.mgf',
        '23' : 'Fetal_Ovary_bRP_Elite_25_f23.mgf',
        '24' : 'Fetal_Ovary_bRP_Elite_25_f24.mgf',
    }


    MATCHES_TO_FILES_LIST = {
        "adult_adrenalgland_gel_elite.csv" : ADULT_ADRENALGLAND_GEL_ELITE_FILES,
        "adult_adrenalgland_gel_velos.csv" : ADULT_ADRENALGLAND_GEL_VELOS_FILES,
        "adult_adrenalgland_bRP_velos.csv" : ADULT_ADRENALGLAND_BRP_VELOS_FILES,
        "adult_heart_brp_elite.csv" : ADULT_HEART_BRP_ELITE_FILES,
        "adult_platelets_gel_elite.csv" : ADULT_PLATELETS_GEL_ELITE_FILES,
        "adult_urinarybladder_gel_elite.csv" : ADULT_URINARYBLADDER_GEL_ELITE_FILES,
        "fetal_brain_gel_velos.csv" : FETAL_BRAIN_GEL_VELOS_FILES,
        "fetal_ovary_brp_velos.csv" : FETAL_OVARY_BRP_VELOS_FILES,
        "fetal_ovary_brp_elite.csv" : FETAL_OVARY_BRP_ELITE_FILES
    }


    def __init__(self, identificationsFilename = None, spectraFilename = None):
    
        self.identificationsFilename = identificationsFilename
        self.spectraFilename = spectraFilename
    
        self.totalSpectra = SpectraFound(False, 'sequences')
    
    
    def load_identifications(self, verbose = False, filteredFilesList = None):

        #
        # First check if the spectra has already been loaded
        #
        
        self.totalSpectra.load_spectra(self.spectraFilename)
        
        if self.totalSpectra.spectra: 
            return
        
        print('Loading file: {}. dir:{}'.format(self.identificationsFilename, os.getcwd()))

        matches_file = pd.read_csv(self.identificationsFilename)

        if verbose:
            print(matches_file)

        print('Number of unique sequences found: {}'.format(len(matches_file['Sequence'].unique())))

        #
        # Inject new columns to hold the scan sequence within the file and the file name, recovered from the 
        # "Spectrum Title" information.
        #

        matches_file['File Sequence'] = \
            matches_file['Spectrum Title'].str.split("_", expand = True).iloc[:, 3].str.zfill(MGF.SCAN_SEQUENCE_MAX_DIGITS)
        
        matches_file['File'] = matches_file['Spectrum Title'].str.split("_", expand = True).iloc[:, 2]

        ordered = matches_file.sort_values(['File', 'File Sequence'])


        #
        # Select only unique "Sequence" + "First Scan" combination:
        #
        # - Same spectrum can contain different sequences
        # - Different sequences can be read in different scans

        duplicatesIndication = ordered.duplicated(['Spectrum Title', 'Sequence', 'First Scan'])

        self.uniqueCombination = ordered[~duplicatesIndication]
        print('Unique combinations found: {}'.format(self.uniqueCombination.shape))

        # print('Unique combinations file {}: {}'.format('b01', 
        #                                                self.uniqueCombination[self.uniqueCombination['File'] == 'b01'].shape))

        if filteredFilesList:
            self.uniqueCombination = self.uniqueCombination[self.uniqueCombination['File'].isin(filteredFilesList)]


    def read_spectra(self, spectraParser, storeUnrecognized = True):
        currentFileName           = ''
        currentScanFileNamePrefix = ''

        currentFile  = None
        lastScan     = None

        startTime = time.time()

        spectraFiles = PXD000561.MATCHES_TO_FILES_LIST[self.identificationsFilename]

        for index, row in self.uniqueCombination.iterrows():
            try:
                if (spectraFiles[row['File']] != currentFileName):
                    if (currentFile != None):

                        if storeUnrecognized:
                            #
                            # Dumps the rest of current file as unrecognized spectra
                            #

                            spectraParser.read_spectrum(currentFile, currentFileNamePrefix + '_', 
                                                        MGF.SCAN_SEQUENCE_OVER_LIMIT, 
                                                        '', 
                                                        self.totalSpectra)

                            print('File {}. Processing time of {} seconds'.format(currentFile.name, 
                                                                                time.time() - startTime))

                        currentFile.close()

                        # break

                    currentFileNamePrefix = row['File']
                    currentFileName = spectraFiles[currentFileNamePrefix]

                    print('Will open file \"{}\"'.format(currentFileName))

                    currentFile = open(currentFileName, 'r')
                    lastScan    = None

                _, lastScan, _ = spectraParser.read_spectrum(currentFile, 
                                                             currentFileNamePrefix + '_', 
                                                             row['First Scan'], 
                                                             row['Sequence'], 
                                                             self.totalSpectra, 
                                                             currentScan = lastScan,
                                                             storeUnrecognized = storeUnrecognized)
            except KeyError:
                Logger()("- Identification file key {} has not corresponding spectra file.")

        Logger()('{} Total processing time {} seconds'.format(self.identificationsFilename, time.time() - startTime))
        
        self.totalSpectra.save_spectra(self.spectraFilename)


